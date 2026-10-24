"""MCP-surface tests for get_signal: verdict round-trip through a fake classifier,
plus the 5-minute TTL cache asserted via classifier call counts (ticket #13)."""

import json
import time

import pandas as pd
import pytest
from fakes import fake_history
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
    assert payload.pop("trade_plan")["direction"] == "long"
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
    payload = json.loads(text_of(result))
    assert payload.pop("trade_plan")["direction"] == "short"
    assert payload == good
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
    """Minimal requests.Response stub: raise_for_status no-op, json returns the body.

    Used only by the Decisions-transport tests below; the fake_post in
    _real_transport_with_bodies raises before a body is needed for HTTP errors.
    """

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


PROMPT = json.dumps(
    {"ticker": "AAPL", "period": "1y", "rows": [{"Date": "2024-01-03T00:00:00"}]}
)


def _decisions_body(answer):
    return {"answers": {"verdict": answer}, "usage": {"cost": 0.001}, "model": "typesafe/jev-1.13"}


def _choice_answer(
    choice="sell",
    probabilities=None,
    confidence: float | None = 0.9,
):
    answer = {
        "type": "choice",
        "choice": choice,
        "probabilities": probabilities
        if probabilities is not None
        else {"buy": 0.1, "sell": 0.7, "hold": 0.2},
    }
    if confidence is not None:
        answer["confidence"] = confidence
    return answer


def _real_transport_with_bodies(monkeypatch, bodies):
    """Real OpenRouterClassifier transport with requests.post serving bodies in order."""
    posts = []

    def fake_post(url, **kwargs):
        posts.append((url, kwargs))
        return FakeResponse(bodies.pop(0))

    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", fake_post)
    return EnforcingClassifier(OpenRouterClassifier()), posts


def test_decisions_answer_maps_to_strict_verdict(monkeypatch):
    # The Decisions answer becomes the settled verdict dict: confidence straight
    # from the answer, rationale citing the probabilities, data_date from the
    # prompt's last row (the Decisions API returns no date of its own).
    classifier, posts = _real_transport_with_bodies(
        monkeypatch, [_decisions_body(_choice_answer())]
    )

    verdict = classifier.classify(PROMPT)

    assert verdict == {
        "verdict": "sell",
        "confidence": 0.9,
        "rationale": "Decisions API chose sell (probabilities: buy=0.1, sell=0.7, hold=0.2)",
        "data_date": "2024-01-03T00:00:00",
    }
    assert len(posts) == 1
    url, kwargs = posts[0]
    assert url == "https://openrouter.ai/api/alpha/decisions"
    request = kwargs["json"]
    assert request["model"] == classifier_model_id()
    assert request["state"] == PROMPT
    question = request["questions"]["verdict"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"buy", "sell", "hold"}
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_missing_confidence_falls_back_to_chosen_probability(monkeypatch):
    # confidence and probabilities are schema-optional; with confidence absent
    # the chosen option's probability stands in for it.
    classifier, _ = _real_transport_with_bodies(
        monkeypatch,
        [_decisions_body(_choice_answer(confidence=None))],
    )

    verdict = classifier.classify(PROMPT)

    assert verdict["verdict"] == "sell"
    assert verdict["confidence"] == 0.7


def test_missing_verdict_answer_retried_once_then_valid(monkeypatch):
    # A reply without answers.verdict is a contract violation like bad JSON:
    # exactly one retry, and the retry's valid answer succeeds.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [
            {"answers": {}, "usage": {}, "model": "typesafe/jev-1.13"},
            _decisions_body(_choice_answer()),
        ],
    )

    assert classifier.classify(PROMPT)["verdict"] == "sell"
    assert len(posts) == 2  # exactly one retry


def test_unexpected_answer_type_retried_once_then_loud(monkeypatch):
    # Per the Decisions contract the answer's type field is checked before
    # reading; an unexpected type is an error, never a default.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [
            _decisions_body({"type": "text", "text": "I think sell"}),
            _decisions_body({"type": "text", "text": "I still think sell"}),
        ],
    )

    with pytest.raises(SignalContractError) as exc_info:
        classifier.classify(PROMPT)

    assert "twice" in str(exc_info.value)
    assert "type" in str(exc_info.value)
    assert len(posts) == 2


def test_choice_outside_buy_sell_hold_retried_once_then_loud(monkeypatch):
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [
            _decisions_body(_choice_answer(choice="maybe")),
            _decisions_body(_choice_answer(choice="maybe")),
        ],
    )

    with pytest.raises(SignalContractError) as exc_info:
        classifier.classify(PROMPT)

    assert "twice" in str(exc_info.value)
    assert "maybe" in str(exc_info.value)
    assert len(posts) == 2


def test_missing_confidence_and_probability_is_a_contract_error(monkeypatch):
    # With neither confidence nor a probability for the choice there is no
    # schema-valid confidence: a contract violation, retried once then loud.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch,
        [
            _decisions_body(_choice_answer(confidence=None, probabilities={})),
            _decisions_body(_choice_answer(confidence=None, probabilities={})),
        ],
    )

    with pytest.raises(SignalContractError) as exc_info:
        classifier.classify(PROMPT)

    assert "twice" in str(exc_info.value)
    assert len(posts) == 2


def test_provider_http_error_is_loud_and_never_retried(monkeypatch):
    # Non-200 responses are provider failures (#12 semantics): loud HTTPError,
    # never retried, never wrapped as a contract violation.
    import requests

    classifier, posts = _real_transport_with_bodies(monkeypatch, [])

    def failing_post(url, **kwargs):
        posts.append((url, kwargs))
        raise requests.HTTPError("503 service unavailable")

    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", failing_post)

    with pytest.raises(requests.HTTPError):
        classifier.classify(PROMPT)

    assert len(posts) == 1


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


# --- Trade Plan attached to the Verdict (ticket #21) ---


def flat_history(close: float = 100.0, rows: int = 60) -> pd.DataFrame:
    """Flat close with High/Low one unit away: True Range is 2 every row, so ATR(14) is 2."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", name="Date")
    return pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000},
        index=idx,
    )


class FlatFakeFetcher(SignalFakeFetcher):
    def __init__(self, close: float = 100.0):
        super().__init__()
        self.close = close

    def fetch_history(self, ticker, period):
        self.history_calls.append((ticker, period))
        return flat_history(self.close)


def _verdict(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence,
            "rationale": "test", "data_date": "2024-02-29T00:00:00"}


async def test_buy_carries_a_long_trade_plan(server_url):
    # Entry 100, ATR 2: Stop = 100 - 2*2 = 96, Target = 100 + 2*(100 - 96) = 108.
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=FakeClassifier(_verdict("buy"))),
        "get_signal",
        {"ticker": "AAPL"},
    )

    assert not result.is_error, text_of(result)
    assert json.loads(text_of(result)) == {
        **_verdict("buy"),
        "trade_plan": {
            "direction": "long",
            "entry": 100.0,
            "stop": pytest.approx(96.0),
            "target": pytest.approx(108.0),
            "atr": pytest.approx(2.0),
            "atr_multiple": 2.0,
            "reward_ratio": 2.0,
        },
    }


async def test_sell_carries_a_short_trade_plan(server_url):
    # Entry 100, ATR 2: Stop = 100 + 2*2 = 104, Target = 100 - 2*(104 - 100) = 92.
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=FakeClassifier(_verdict("sell"))),
        "get_signal",
        {"ticker": "AAPL"},
    )

    assert not result.is_error, text_of(result)
    plan = json.loads(text_of(result))["trade_plan"]
    assert plan["direction"] == "short"
    assert plan["entry"] == 100.0
    assert plan["stop"] == pytest.approx(104.0)
    assert plan["target"] == pytest.approx(92.0)


async def test_hold_has_no_trade_plan(server_url):
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=FakeClassifier(_verdict("hold"))),
        "get_signal",
        {"ticker": "AAPL"},
    )

    assert not result.is_error, text_of(result)
    assert json.loads(text_of(result)) == {**_verdict("hold"), "trade_plan": None}


async def test_plan_is_attached_even_at_very_low_confidence(server_url):
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=FakeClassifier(_verdict("buy", 0.01))),
        "get_signal",
        {"ticker": "AAPL"},
    )

    assert not result.is_error, text_of(result)
    payload = json.loads(text_of(result))
    assert payload["confidence"] == 0.01
    assert payload["trade_plan"]["direction"] == "long"


# Entry 100, ATR 2, atr_multiple 1.5, reward_ratio 3:
# long Stop = 100 - 1.5*2 = 97, Target = 100 + 3*(100-97) = 109;
# short Stop = 100 + 1.5*2 = 103, Target = 100 - 3*(103-100) = 91.
@pytest.mark.parametrize(("verdict", "stop", "target"), [("buy", 97.0, 109.0), ("sell", 103.0, 91.0)])
async def test_custom_atr_multiple_and_reward_ratio_change_the_plan_and_are_echoed(
    server_url, verdict, stop, target
):
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=FakeClassifier(_verdict(verdict))),
        "get_signal",
        {"ticker": "AAPL", "atr_multiple": 1.5, "reward_ratio": 3},
    )

    assert not result.is_error, text_of(result)
    plan = json.loads(text_of(result))["trade_plan"]
    assert plan["stop"] == pytest.approx(stop)
    assert plan["target"] == pytest.approx(target)
    assert plan["atr_multiple"] == 1.5
    assert plan["reward_ratio"] == 3


async def test_exactly_ten_is_accepted_and_ints_echo_as_floats(server_url):
    # Entry 100, ATR 2: Stop = 100 - 10*2 = 80; Target = 100 + 10*(100-80) = 300.
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=FakeClassifier(_verdict("buy"))),
        "get_signal",
        {"ticker": "AAPL", "atr_multiple": 10, "reward_ratio": 10},
    )

    assert not result.is_error, text_of(result)
    plan = json.loads(text_of(result))["trade_plan"]
    assert plan["stop"] == pytest.approx(80.0)
    assert plan["target"] == pytest.approx(300.0)
    assert isinstance(plan["atr_multiple"], float)
    assert isinstance(plan["reward_ratio"], float)


@pytest.mark.parametrize("arg", ["atr_multiple", "reward_ratio"])
@pytest.mark.parametrize("value", [0, -1, 10.5])
async def test_out_of_bounds_plan_argument_fails_before_the_classifier(server_url, arg, value):
    classifier = FakeClassifier()
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", arg: value},
    )

    assert result.is_error
    assert arg in text_of(result)
    assert "(0, 10]" in text_of(result)
    assert classifier.prompts == []


@pytest.mark.parametrize("arg", ["atr_multiple", "reward_ratio"])
@pytest.mark.parametrize("value", [True, "2"])
async def test_non_number_plan_argument_fails_before_the_classifier(server_url, arg, value):
    classifier = FakeClassifier()
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", arg: value},
    )

    assert result.is_error
    assert arg in text_of(result)
    assert classifier.prompts == []


# Entry 1, ATR 2: long Stop = 1 - 2*2 = -3; short Stop = 5, Target = 1 - 2*(5 - 1) = -7.
@pytest.mark.parametrize("verdict", ["buy", "sell"])
async def test_non_positive_plan_price_is_a_loud_tool_error(server_url, verdict):
    result = await call_tool(
        server_url(FlatFakeFetcher(close=1.0), classifier=FakeClassifier(_verdict(verdict))),
        "get_signal",
        {"ticker": "AAPL"},
    )

    assert result.is_error
    assert "non-positive" in text_of(result)


async def test_cached_verdict_gets_a_recomputed_plan_for_new_arguments(server_url):
    # atr_multiple 1: Stop = 100 - 1*2 = 98, Target = 100 + 2*(100 - 98) = 104.
    classifier = FakeClassifier(_verdict("buy"))
    url = server_url(FlatFakeFetcher(), classifier=classifier, signal_ttl=TEST_SIGNAL_TTL)

    first = await call_tool(url, "get_signal", {"ticker": "AAPL"})
    second = await call_tool(url, "get_signal", {"ticker": "AAPL", "atr_multiple": 1})

    assert not first.is_error and not second.is_error
    assert json.loads(text_of(first))["trade_plan"]["stop"] == pytest.approx(96.0)
    plan = json.loads(text_of(second))["trade_plan"]
    assert plan["stop"] == pytest.approx(98.0)
    assert plan["target"] == pytest.approx(104.0)
    assert len(classifier.prompts) == 1
