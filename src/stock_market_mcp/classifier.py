"""Classifier transport boundary: get_signal's verdict comes from an injectable seam.

The Protocol mirrors DataFetcher's pattern: tests inject a fake, and the real
transport below is the only OpenRouter-aware code in the server.
"""

import os
from typing import Protocol

import requests

DEFAULT_CLASSIFIER_MODEL = "typesafe/jev-1.13"

# Endpoint pinned as a constant: https only, and requests.post with a timeout
# (never urllib) so no file: or custom scheme can sneak into the transport.
# The default model (typesafe/jev-1.13) is a Decisions model: it rejects
# chat/completions with 400, so the transport targets the Decisions API.
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# Stored with each Signal; bump by hand whenever the prompt or the criteria below change.
PROMPT_VERSION = 3

_BUY = "Bullish momentum or trend evidence in the supplied data."
_HOLD = "Mixed or insufficient evidence in the supplied data."

# The single question the verdict hinges on: _NOT_HELD_QUESTIONS without a Position, _HELD_QUESTIONS with one.
# The state supplied by the caller is the prompt builder's serialized OHLCV+indicators
# payload (the API accepts a string state verbatim).
_NOT_HELD_QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "Given the supplied price history and indicators, is now the right "
            "time to buy this stock, or to hold off? The signal is advisory."
        ),
        "criteria": {"buy": _BUY, "hold": _HOLD},
    },
}
_HELD_QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "A long position in this stock is supplied. Given the price history, "
            "indicators and the position, should the holder buy more, hold, or cut "
            "(exit the long now)? The signal is advisory."
        ),
        "criteria": {
            "buy": _BUY,
            "hold": _HOLD,
            "cut": "Bearish evidence in the supplied data: exit the long now, at a gain or a loss.",
        },
    },
}


def _questions(held: bool) -> dict:
    """The Decisions questions for a prompt with (held) or without a Position."""
    return _HELD_QUESTIONS if held else _NOT_HELD_QUESTIONS


def _verdicts(held: bool) -> list[str]:
    """The Verdicts a Signal may have: buy or hold, plus cut on a held Position."""
    return list(_questions(held)["verdict"]["criteria"])


def classifier_model_id() -> str:
    """Model id for the signal classifier; SMM_CLASSIFIER_MODEL overrides the default."""
    # Empty string counts as unset so compose's `${SMM_CLASSIFIER_MODEL:-}`
    # passthrough falls back to the default instead of shipping an empty id.
    return os.environ.get("SMM_CLASSIFIER_MODEL") or DEFAULT_CLASSIFIER_MODEL


class SignalClassifier(Protocol):
    def classify(self, prompt: str, held: bool) -> dict:
        """The Signal for prompt; held says the prompt supplies a Position, which allows a cut."""
        ...


class SignalContractError(Exception):
    """A classifier reply violates the strict verdict contract (retryable once)."""


_VERDICT_KEYS = frozenset({"verdict", "confidence", "rationale"})


def validate_signal(payload: object, held: bool) -> dict:
    """Enforce the strict verdict schema; raise SignalContractError naming the violation."""
    if not isinstance(payload, dict):
        raise SignalContractError(f"reply is not a JSON object: {type(payload).__name__}")
    missing = _VERDICT_KEYS - payload.keys()
    if missing:
        raise SignalContractError(f"reply missing key(s): {', '.join(sorted(missing))}")
    extra = payload.keys() - _VERDICT_KEYS
    if extra:
        raise SignalContractError(f"reply has unexpected key(s): {', '.join(sorted(extra))}")
    if payload["verdict"] not in _verdicts(held):
        raise SignalContractError(
            f"verdict must be one of {'/'.join(_verdicts(held))}, got: {payload['verdict']!r}"
        )
    confidence = payload["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        raise SignalContractError(
            f"confidence must be a number in [0, 1], got: {confidence!r}"
        )
    if not isinstance(payload["rationale"], str):
        raise SignalContractError(f"rationale must be a string, got: {payload['rationale']!r}")
    return payload


class EnforcingClassifier:
    """SignalClassifier decorator enforcing the strict schema with exactly one retry.

    A malformed reply gets one second attempt; a persistent violation is a loud
    SignalContractError naming the violation. Non-contract failures (provider
    errors, missing key) propagate immediately and are never retried.
    """

    def __init__(self, classifier: SignalClassifier) -> None:
        self._classifier = classifier

    def classify(self, prompt: str, held: bool) -> dict:
        try:
            return validate_signal(self._classifier.classify(prompt, held), held)
        except SignalContractError as first_error:
            try:
                return validate_signal(self._classifier.classify(prompt, held), held)
            except SignalContractError as second_error:
                raise SignalContractError(
                    f"classifier reply violated the signal contract twice, "
                    f"giving up: {second_error} (first attempt: {first_error})"
                ) from second_error


class OpenRouterClassifier:
    """Decisions-API transport against OpenRouter's /api/alpha/decisions endpoint."""

    def classify(self, prompt: str, held: bool) -> dict:
        # Key is read at call time and never stored; a missing key is a loud
        # error naming the cause, before any HTTP request is attempted.
        api_key = os.environ.get("SMM_OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "SMM_OPENROUTER_API_KEY is not set; get_signal requires an "
                "OpenRouter API key (the other tools work without it)"
            )
        request = {
            "model": classifier_model_id(),
            "state": prompt,
            "questions": _questions(held),
        }
        response = requests.post(
            OPENROUTER_DECISIONS_URL,
            json=request,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        # Non-200 is a provider failure (#12 semantics): the HTTPError propagates
        # loudly and is never retried or converted into a contract violation.
        response.raise_for_status()
        body = response.json()
        answers = body.get("answers")
        if not isinstance(answers, dict) or "verdict" not in answers:
            raise SignalContractError("decisions reply missing answer for 'verdict'")
        answer = answers["verdict"]
        # The type field must be checked before reading; an unexpected type is an
        # error, never a default.
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise SignalContractError(f"answer for 'verdict' has unexpected type: {answer!r}")
        # The Verdict itself is checked once, by validate_signal in EnforcingClassifier.
        choice = answer.get("choice")
        probabilities = answer.get("probabilities")
        confidence = answer.get("confidence")
        # confidence and probabilities are schema-optional; read through presence
        # checks, and with neither there is no schema-valid confidence.
        if confidence is None:
            # A non-string choice (e.g. a list) is unhashable, so check it before the lookup.
            if not isinstance(choice, str) or not isinstance(probabilities, dict) or choice not in probabilities:
                raise SignalContractError(
                    f"answer has neither confidence nor a probability for '{choice}'"
                )
            confidence = probabilities[choice]
        probs_desc = (
            ", ".join(f"{key}={value}" for key, value in probabilities.items())
            if isinstance(probabilities, dict)
            else "unavailable"
        )
        return {
            "verdict": choice,
            "confidence": confidence,
            "rationale": f"Decisions API chose {choice} (probabilities: {probs_desc})",
        }
