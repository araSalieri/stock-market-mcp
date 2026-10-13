"""Classifier transport boundary: get_signal's verdict comes from an injectable seam.

The Protocol mirrors DataFetcher's pattern: tests inject a fake, and the real
transport below is the only OpenRouter-aware code in the server.
"""

import json
import os
from typing import Protocol

import requests

DEFAULT_CLASSIFIER_MODEL = "typesafe/jev-1.13"

# Endpoint pinned as a constant: https only, and requests.post with a timeout
# (never urllib) so no file: or custom scheme can sneak into the transport.
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def classifier_model_id() -> str:
    """Model id for the signal classifier; SMM_CLASSIFIER_MODEL overrides the default."""
    return os.environ.get("SMM_CLASSIFIER_MODEL", DEFAULT_CLASSIFIER_MODEL)


class SignalClassifier(Protocol):
    def classify(self, prompt: str) -> dict: ...


class SignalContractError(Exception):
    """A classifier reply violates the strict verdict contract (retryable once)."""


_VERDICT_KEYS = frozenset({"verdict", "confidence", "rationale", "data_date"})
_VALID_VERDICTS = frozenset({"buy", "sell", "hold"})


def validate_signal(payload: object) -> dict:
    """Enforce the strict verdict schema; raise SignalContractError naming the violation."""
    if not isinstance(payload, dict):
        raise SignalContractError(f"reply is not a JSON object: {type(payload).__name__}")
    missing = _VERDICT_KEYS - payload.keys()
    if missing:
        raise SignalContractError(f"reply missing key(s): {', '.join(sorted(missing))}")
    extra = payload.keys() - _VERDICT_KEYS
    if extra:
        raise SignalContractError(f"reply has unexpected key(s): {', '.join(sorted(extra))}")
    if payload["verdict"] not in _VALID_VERDICTS:
        raise SignalContractError(
            f"verdict must be one of buy/sell/hold, got: {payload['verdict']!r}"
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
    for key in ("rationale", "data_date"):
        if not isinstance(payload[key], str):
            raise SignalContractError(f"{key} must be a string, got: {payload[key]!r}")
    return payload


class EnforcingClassifier:
    """SignalClassifier decorator enforcing the strict schema with exactly one retry.

    A malformed reply gets one second attempt; a persistent violation is a loud
    SignalContractError naming the violation. Non-contract failures (provider
    errors, missing key) propagate immediately and are never retried.
    """

    def __init__(self, classifier: SignalClassifier) -> None:
        self._classifier = classifier

    def classify(self, prompt: str) -> dict:
        try:
            return validate_signal(self._classifier.classify(prompt))
        except SignalContractError as first_error:
            try:
                return validate_signal(self._classifier.classify(prompt))
            except SignalContractError as second_error:
                raise SignalContractError(
                    f"classifier reply violated the signal contract twice, "
                    f"giving up: {second_error} (first attempt: {first_error})"
                ) from second_error


class OpenRouterClassifier:
    """Chat-completion transport against the OpenRouter endpoint, strict JSON contract."""

    def classify(self, prompt: str) -> dict:
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
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a market timing classifier. Judge the supplied "
                        "price history and indicators. The signal is advisory. "
                        "Answer with strict JSON only: "
                        '{"verdict": "buy"|"sell"|"hold", "confidence": 0-1, '
                        '"rationale": string grounded in the data, '
                        '"data_date": the date of the last data row}.'
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "signal_verdict",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "verdict": {
                                "type": "string",
                                "enum": ["buy", "sell", "hold"],
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "rationale": {"type": "string"},
                            "data_date": {"type": "string"},
                        },
                        "required": ["verdict", "confidence", "rationale", "data_date"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        response = requests.post(
            OPENROUTER_CHAT_URL,
            json=request,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        content = body["choices"][0]["message"]["content"]
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            # A chatty/malformed model reply is a formatting slip, so it goes
            # through the same one-retry contract as a schema violation.
            raise SignalContractError(f"reply is not valid JSON: {exc}") from exc
