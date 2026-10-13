"""MCP-surface tests for get_signal: verdict round-trip through a fake classifier."""

import json

from test_get_history import fake_history
from test_get_quote import call_tool, text_of

from stock_market_mcp.classifier import classifier_model_id


class SignalFakeFetcher:
    """Fake data layer: history only; quote/fundamentals seams must stay untouched."""

    def __init__(self):
        self.history_calls = []

    def fetch_quote(self, ticker):
        raise AssertionError("get_signal tests never fetch quotes")

    def fetch_fundamentals(self, ticker):
        raise AssertionError("get_signal tests never fetch fundamentals")

    def fetch_history(self, ticker, period):
        self.history_calls.append((ticker, period))
        return fake_history(60)


class FakeClassifier:
    """Fake transport below the classifier boundary; no network, records prompts."""

    def __init__(self, verdict=None, error=None):
        self.verdict = verdict or {
            "verdict": "buy",
            "confidence": 0.7,
            "rationale": "uptrend with rising RSI",
            "data_date": "2024-01-03T11:00:00",
        }
        self.error = error
        self.prompts = []

    def classify(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.verdict


def test_classifier_model_id_default_and_env_override(monkeypatch):
    monkeypatch.delenv("SMM_CLASSIFIER_MODEL", raising=False)
    assert classifier_model_id() == "typesafe/jev-1.13"

    monkeypatch.setenv("SMM_CLASSIFIER_MODEL", "acme/other-2")
    assert classifier_model_id() == "acme/other-2"


async def test_get_signal_returns_strict_json_verdict(server_url):
    classifier = FakeClassifier()
    fetcher = SignalFakeFetcher()
    result = await call_tool(
        server_url(fetcher, classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert not result.is_error
    payload = json.loads(text_of(result))
    assert payload == {
        "verdict": "buy",
        "confidence": 0.7,
        "rationale": "uptrend with rising RSI",
        "data_date": "2024-01-03T11:00:00",
    }
    assert fetcher.history_calls == [("AAPL", "1mo")]
    assert len(classifier.prompts) == 1
    prompt_payload = json.loads(classifier.prompts[0])
    assert prompt_payload["ticker"] == "AAPL"
    assert prompt_payload["period"] == "1mo"
    last_row = prompt_payload["rows"][-1]
    assert last_row["Date"].startswith("2024-01-03T11:00:00")
    assert "RSI_14" in last_row  # the computed indicator set is in the prompt
