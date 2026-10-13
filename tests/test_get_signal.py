"""MCP-surface tests for get_signal: verdict round-trip through a fake classifier,
plus the 5-minute TTL cache asserted via classifier call counts (ticket #13)."""

import json
import time

import pytest
from test_get_history import fake_history
from test_get_quote import FakeFetcher, call_tool, text_of

from stock_market_mcp.classifier import (
    EnforcingClassifier,
    OpenRouterClassifier,
    SignalContractError,
    classifier_model_id,
)

# Short TTL so tests can cross the expiry window without waiting on real timeouts.
TEST_SIGNAL_TTL = 0.2


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
    """Fake transport below the classifier boundary; no network, records prompts.

    replies (optional) is a sequence of verdicts handed out one per classify call;
    without it every call returns verdict.
    """

    def __init__(self, verdict=None, error=None, replies=None):
        self.verdict = verdict or {
            "verdict": "buy",
            "confidence": 0.7,
            "rationale": "uptrend with rising RSI",
            "data_date": "2024-01-03T11:00:00",
        }
        self.error = error
        self.replies = list(replies) if replies is not None else None
        self.prompts = []

    def classify(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        if self.replies is not None:
            return self.replies.pop(0)
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


async def test_malformed_reply_retried_once_then_loud_error(server_url):
    # Both attempts violate the contract; exactly one retry, then a loud tool
    # error naming the violation. Agents never see a fabricated success.
    bad_verdict = {"verdict": "maybe", "confidence": 0.7,
                   "rationale": "uptrend", "data_date": "2024-01-03"}
    missing_key = {"verdict": "hold", "confidence": 0.5, "data_date": "2024-01-03"}
    classifier = FakeClassifier(replies=[bad_verdict, missing_key])
    result = await call_tool(
        server_url(fetcher := SignalFakeFetcher(), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert result.is_error
    message = text_of(result)
    assert "verdict" in message  # names the violation
    assert len(classifier.prompts) == 2  # exactly one retry
    assert fetcher.history_calls == [("AAPL", "1mo")]


async def test_single_formatting_slip_recovers_after_one_retry(server_url):
    # One malformed reply, then a valid one: the call succeeds on the retry.
    bad = {"verdict": "maybe", "confidence": 0.7,
           "rationale": "uptrend", "data_date": "2024-01-03"}
    good = {"verdict": "sell", "confidence": 0.9,
            "rationale": "overbought", "data_date": "2024-01-03"}
    classifier = FakeClassifier(replies=[bad, good])
    result = await call_tool(
        server_url(SignalFakeFetcher(), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert not result.is_error
    assert json.loads(text_of(result)) == good
    assert len(classifier.prompts) == 2  # the retry happened, not a third call


async def test_provider_failure_is_loud_and_never_retried(server_url):
    # Transport errors are not formatting slips: loud immediately, no retry call.
    classifier = FakeClassifier(error=RuntimeError("openrouter timeout"))
    result = await call_tool(
        server_url(SignalFakeFetcher(), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert result.is_error
    assert "openrouter timeout" in text_of(result)
    assert len(classifier.prompts) == 1


async def test_missing_openrouter_key_is_a_loud_tool_error(server_url, monkeypatch):
    # Real transport (no fake): the key is checked at call time, before any HTTP.
    # The patched requests.post proves the error fires with zero network I/O.
    monkeypatch.delenv("SMM_OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        "stock_market_mcp.classifier.requests.post",
        lambda *a, **k: pytest.fail("network I/O attempted despite missing key"),
    )
    result = await call_tool(
        server_url(SignalFakeFetcher()),  # no classifier injected: real transport
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert result.is_error
    assert "SMM_OPENROUTER_API_KEY" in text_of(result)


async def test_other_tools_keep_working_without_openrouter_key(server_url, monkeypatch):
    monkeypatch.delenv("SMM_OPENROUTER_API_KEY", raising=False)
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})

    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    assert not result.is_error
    assert "190.0" in text_of(result)


async def test_second_signal_within_ttl_is_served_from_cache(server_url):
    classifier = FakeClassifier()
    fetcher = SignalFakeFetcher()
    url = server_url(fetcher, classifier=classifier, signal_ttl=TEST_SIGNAL_TTL)

    first = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})
    second = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})

    assert not first.is_error
    assert not second.is_error
    assert text_of(second) == text_of(first)  # identical verdict inside the window
    assert len(classifier.prompts) == 1  # second call never touched the classifier


async def test_signal_after_ttl_expiry_reinvokes_classifier(server_url):
    classifier = FakeClassifier()
    fetcher = SignalFakeFetcher()
    url = server_url(fetcher, classifier=classifier, signal_ttl=TEST_SIGNAL_TTL)

    first = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})
    time.sleep(TEST_SIGNAL_TTL + 0.1)
    second = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})

    assert not first.is_error
    assert not second.is_error
    assert len(classifier.prompts) == 2  # window expired: classifier ran again
    assert text_of(second) == text_of(first)


class FakeResponse:
    """Minimal requests.Response stub: raise_for_status no-op, json returns the body."""

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def _openrouter_body(content):
    return {"choices": [{"message": {"content": content}}]}


GOOD_VERDICT = {
    "verdict": "sell",
    "confidence": 0.9,
    "rationale": "overbought",
    "data_date": "2024-01-03",
}


def _real_transport_with_bodies(monkeypatch, bodies):
    """Real OpenRouterClassifier transport with requests.post serving bodies in order."""
    posts = []

    def fake_post(*args, **kwargs):
        posts.append(args)
        return FakeResponse(bodies.pop(0))

    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", fake_post)
    return EnforcingClassifier(OpenRouterClassifier()), posts


def test_none_content_retried_once_then_valid_verdict(monkeypatch):
    # A non-string message content is a formatting slip like bad JSON: it goes
    # through the one-retry contract, and the retry's valid verdict succeeds.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [_openrouter_body(None), _openrouter_body(json.dumps(GOOD_VERDICT))],
    )

    assert classifier.classify("prompt") == GOOD_VERDICT
    assert len(posts) == 2  # exactly one retry


def test_none_content_twice_is_loud_contract_error(monkeypatch):
    # Two non-string contents exhaust the retry: a loud SignalContractError
    # naming the violation, never a fabricated success.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch, [_openrouter_body(None), _openrouter_body(None)]
    )

    with pytest.raises(SignalContractError) as exc_info:
        classifier.classify("prompt")

    assert "twice" in str(exc_info.value)
    assert "JSON" in str(exc_info.value)
    assert len(posts) == 2


async def test_failed_signal_is_never_cached(server_url):
    classifier = FakeClassifier(error=RuntimeError("openrouter 500"))
    fetcher = SignalFakeFetcher()
    url = server_url(fetcher, classifier=classifier, signal_ttl=TEST_SIGNAL_TTL)

    first = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})
    second = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})

    assert first.is_error
    assert "openrouter 500" in text_of(first)
    assert second.is_error  # failure is retried, not pinned to the ticker
    assert "openrouter 500" in text_of(second)
    assert len(classifier.prompts) == 2  # the failed call was not cached
