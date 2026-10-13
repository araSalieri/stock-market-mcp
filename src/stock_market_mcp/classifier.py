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


class OpenRouterClassifier:
    """Chat-completion transport against the OpenRouter endpoint, strict JSON contract."""

    def classify(self, prompt: str) -> dict:
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
            # Key is read at call time and never stored; the missing-key loud
            # error is the hardening ticket's scope, not this one.
            headers={
                "Authorization": f"Bearer {os.environ.get('SMM_OPENROUTER_API_KEY', '')}",
            },
            timeout=120,
        )
        response.raise_for_status()
        body = response.json()
        return json.loads(body["choices"][0]["message"]["content"])
