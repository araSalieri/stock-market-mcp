"""Classifier: the injectable seam that gives get_signal its Verdict and Confidence.

The Protocol mirrors DataFetcher's pattern: tests inject a fake, and the real
transport below is the only OpenRouter-aware code in the server.
"""

import os
from typing import Protocol

import requests

from stock_market_mcp.prompt import level_questions

DEFAULT_CLASSIFIER_MODEL = "typesafe/jev-1.13"

# Endpoint pinned as a constant: https only, and requests.post with a timeout
# (never urllib) so no file: or custom scheme can sneak into the transport.
# The default model (typesafe/jev-1.13) is a Decisions model: it rejects
# chat/completions with 400, so the transport targets the Decisions API.
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

def classifier_model_id() -> str:
    """Model id for the signal classifier; SMM_CLASSIFIER_MODEL overrides the default."""
    # Empty string counts as unset so compose's `${SMM_CLASSIFIER_MODEL:-}`
    # passthrough falls back to the default instead of shipping an empty id.
    return os.environ.get("SMM_CLASSIFIER_MODEL") or DEFAULT_CLASSIFIER_MODEL


class SignalClassifier(Protocol):
    def classify(self, prompt: str, questions: dict) -> dict:
        """The Signal for prompt, answering questions; the verdict question's criteria are the Verdicts allowed."""
        ...


class SignalContractError(Exception):
    """A classifier reply violates the strict verdict contract (retryable once)."""


_VERDICT_KEYS = frozenset({"verdict", "confidence", "rationale"})


def _check_confidence(confidence: object, name: str) -> None:
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise SignalContractError(f"{name} must be a number in [0, 1], got: {confidence!r}")


def validate_signal(payload: object, questions: dict) -> dict:
    """Enforce the strict verdict schema; raise SignalContractError naming the violation.

    When questions ask for levels, the reply also has picks: each level question's choice, one of its offered
    options, and confidence."""
    if not isinstance(payload, dict):
        raise SignalContractError(f"reply is not a JSON object: {type(payload).__name__}")
    asked = level_questions(questions)
    keys = _VERDICT_KEYS | {"picks"} if asked else _VERDICT_KEYS
    missing = keys - payload.keys()
    if missing:
        raise SignalContractError(f"reply missing key(s): {', '.join(sorted(missing))}")
    extra = payload.keys() - keys
    if extra:
        raise SignalContractError(f"reply has unexpected key(s): {', '.join(sorted(extra))}")
    verdicts = list(questions["verdict"]["criteria"])
    if payload["verdict"] not in verdicts:
        raise SignalContractError(
            f"verdict must be one of {'/'.join(verdicts)}, got: {payload['verdict']!r}"
        )
    _check_confidence(payload["confidence"], "confidence")
    if not isinstance(payload["rationale"], str):
        raise SignalContractError(f"rationale must be a string, got: {payload['rationale']!r}")
    if asked:
        picks = payload["picks"]
        if not isinstance(picks, dict) or sorted(picks) != sorted(asked):
            raise SignalContractError(f"picks must answer {'/'.join(asked)}, got: {picks!r}")
        for question in asked:
            answer = picks[question]
            offered = list(questions[question]["criteria"])
            if not isinstance(answer, dict) or answer.get("choice") not in offered:
                raise SignalContractError(f"{question} must be one of {'/'.join(offered)}, got: {answer!r}")
            _check_confidence(answer.get("confidence"), f"{question} confidence")
    return payload


class EnforcingClassifier:
    """SignalClassifier decorator enforcing the strict schema with exactly one retry.

    A malformed reply gets one second attempt; a persistent violation is a loud
    SignalContractError naming the violation. Non-contract failures (provider
    errors, missing key) propagate immediately and are never retried.
    """

    def __init__(self, classifier: SignalClassifier) -> None:
        self._classifier = classifier

    def classify(self, prompt: str, questions: dict) -> dict:
        try:
            return validate_signal(self._classifier.classify(prompt, questions), questions)
        except SignalContractError as first_error:
            try:
                return validate_signal(self._classifier.classify(prompt, questions), questions)
            except SignalContractError as second_error:
                raise SignalContractError(
                    f"classifier reply violated the signal contract twice, "
                    f"giving up: {second_error} (first attempt: {first_error})"
                ) from second_error


class OpenRouterClassifier:
    """Decisions-API transport against OpenRouter's /api/alpha/decisions endpoint."""

    def classify(self, prompt: str, questions: dict) -> dict:
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
            "questions": questions,
        }
        response = requests.post(
            OPENROUTER_DECISIONS_URL,
            json=request,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        # Non-200 is a provider failure, not a bad answer: the HTTPError propagates
        # loudly and is never retried or converted into a contract violation.
        response.raise_for_status()
        answers = response.json().get("answers")
        if not isinstance(answers, dict):
            raise SignalContractError("decisions reply has no answers")
        choice, confidence, probabilities = _choice(answers, "verdict")
        probs_desc = (
            ", ".join(f"{key}={value}" for key, value in probabilities.items())
            if isinstance(probabilities, dict)
            else "unavailable"
        )
        signal: dict = {
            "verdict": choice,
            "confidence": confidence,
            "rationale": f"Decisions API chose {choice} (probabilities: {probs_desc})",
        }
        asked = level_questions(questions)
        if asked:
            signal["picks"] = {}
            for question in asked:
                choice, confidence, _ = _choice(answers, question)
                signal["picks"][question] = {"choice": choice, "confidence": confidence}
        return signal


def _choice(answers: dict, question: str) -> tuple[object, object, object]:
    """The choice, confidence and probabilities of one choice answer; the choice itself is checked by validate_signal."""
    if question not in answers:
        raise SignalContractError(f"decisions reply missing answer for '{question}'")
    answer = answers[question]
    # The type field must be checked before reading; an unexpected type is an
    # error, never a default.
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise SignalContractError(f"answer for '{question}' has unexpected type: {answer!r}")
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
    return choice, confidence, probabilities
