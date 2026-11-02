"""Tests for get_signal over MCP: one happy path, argument checks, error wrapping, the Decisions transport, and the tool surface; Signal rules are tested on SignalMaker."""

import json
from datetime import UTC, datetime

import pytest
from fakes import (
    FakeClassifier,
    FakeFetcher,
    call_tool,
    daily_bars,
    fake_history,
    five_minute_bars,
    flat_history,
    intraday_history,
    text_of,
)
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from stock_market_mcp.classifier import (
    EnforcingClassifier,
    OpenRouterClassifier,
    SignalContractError,
    classifier_model_id,
)

def test_classifier_model_id_default_and_env_override(monkeypatch):
    monkeypatch.delenv("SMM_CLASSIFIER_MODEL", raising=False)
    assert classifier_model_id() == "typesafe/jev-1.13"

    monkeypatch.setenv("SMM_CLASSIFIER_MODEL", "acme/other-2")
    assert classifier_model_id() == "acme/other-2"


async def test_get_signal_returns_strict_json_signal(server_url):
    classifier = FakeClassifier()
    fetcher = FakeFetcher(history=fake_history())
    result = await call_tool(
        server_url(fetcher, classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert not result.is_error
    payload = json.loads(text_of(result))
    plan = payload.pop("trade_plan")
    assert plan["direction"] == "long"
    assert payload.pop("price") == plan["entry"]
    assert payload.pop("hit_rate") is None
    assert payload.pop("min_confidence") == 0.6
    assert payload == {
        "verdict": "buy",
        "confidence": 0.7,
        "rationale": "uptrend with rising RSI",
        "data_date": "2024-01-03T11:00:00",
    }
    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False)]
    assert len(classifier.prompts) == 1
    prompt_payload = json.loads(classifier.prompts[0])
    assert prompt_payload["ticker"] == "AAPL"
    assert prompt_payload["period"] == "1mo"
    last_row = prompt_payload["rows"][-1]
    assert last_row["Date"].startswith("2024-01-03T11:00:00")
    assert "RSI_14" in last_row  # the computed indicator set is in the prompt


async def test_unknown_ticker_is_a_loud_tool_error(server_url):
    # Spec #19 user story 17: no empty data, and nothing reaches the classifier.
    classifier = FakeClassifier()
    result = await call_tool(
        server_url(FakeFetcher(history={}), classifier=classifier),
        "get_signal",
        {"ticker": "NOPE"},
    )

    assert result.is_error
    assert "NOPE" in text_of(result)
    assert classifier.prompts == []


async def test_malformed_reply_retried_once_then_loud_error(server_url):
    # Both attempts violate the contract; exactly one retry, then a loud tool
    # error naming the violation. Agents never see a fabricated success.
    bad_verdict = {"verdict": "maybe", "confidence": 0.7,
                   "rationale": "uptrend", "data_date": "2024-01-03"}
    missing_key = {"verdict": "hold", "confidence": 0.5, "data_date": "2024-01-03"}
    classifier = FakeClassifier(replies=[bad_verdict, missing_key])
    result = await call_tool(
        server_url(fetcher := FakeFetcher(history=fake_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert result.is_error
    message = text_of(result)
    assert "verdict" in message  # names the violation
    assert len(classifier.prompts) == 2  # exactly one retry
    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False)]


async def test_single_formatting_slip_recovers_after_one_retry(server_url):
    # One malformed reply, then a valid one: the call succeeds on the retry.
    bad = {"verdict": "maybe", "confidence": 0.7,
           "rationale": "uptrend", "data_date": "2024-01-03"}
    good = {"verdict": "sell", "confidence": 0.9,
            "rationale": "overbought", "data_date": "2024-01-03"}
    classifier = FakeClassifier(replies=[bad, good])
    result = await call_tool(
        server_url(FakeFetcher(history=fake_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "period": "1mo"},
    )

    assert not result.is_error
    payload = json.loads(text_of(result))
    plan = payload.pop("trade_plan")
    assert plan["direction"] == "short"
    assert payload.pop("price") == plan["entry"]
    assert payload.pop("hit_rate") is None
    assert payload.pop("min_confidence") == 0.6
    assert payload == good
    assert len(classifier.prompts) == 2  # the retry happened, not a third call


async def test_provider_failure_is_loud_and_never_retried(server_url):
    # Transport errors are not formatting slips: loud immediately, no retry call.
    classifier = FakeClassifier(error=RuntimeError("openrouter timeout"))
    result = await call_tool(
        server_url(FakeFetcher(history=fake_history()), classifier=classifier),
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
        server_url(FakeFetcher(history=fake_history())),  # no classifier injected: real transport
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


def test_decisions_answer_maps_to_strict_signal(monkeypatch):
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
    assert question["criteria"] == {
        "buy": "Bullish momentum or trend evidence in the supplied data.",
        "sell": "Bearish momentum or trend evidence in the supplied data.",
        "hold": "Mixed or insufficient evidence in the supplied data.",
    }
    assert question["instructions"].endswith(
        "If a position is supplied, judge it too: sell on a held long or buy on a held short means close it."
    )
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


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "choice", "choice": "maybe", "confidence": 0.9},
        {"type": "choice", "choice": "maybe", "probabilities": {"buy": 0.1, "sell": 0.7, "hold": 0.2}},
        {"type": "choice", "choice": "maybe"},
        {"type": "choice", "choice": ["maybe"], "probabilities": {"buy": 0.1, "sell": 0.7, "hold": 0.2}},
    ],
    ids=["confidence", "probabilities-only", "neither", "non-string"],
)
def test_choice_outside_buy_sell_hold_retried_once_then_loud(monkeypatch, answer):
    # Called directly: a malformed Decisions reply is a transport detail no fake classifier behind a tool can send.
    classifier, posts = _real_transport_with_bodies(
        monkeypatch, [_decisions_body(answer), _decisions_body(answer)]
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


# --- Trade Plan arguments (ticket #21) ---


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence,
            "rationale": "test", "data_date": "2024-02-29T00:00:00"}


async def test_exactly_ten_is_accepted_and_ints_echo_as_floats(server_url):
    # Entry 100, ATR 2: Stop = 100 - 10*2 = 80; Target = 100 + 10*(100-80) = 300.
    result = await call_tool(
        server_url(FakeFetcher(history=flat_history()), classifier=FakeClassifier(_signal("buy"))),
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
        server_url(FakeFetcher(history=flat_history()), classifier=classifier),
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
        server_url(FakeFetcher(history=flat_history()), classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", arg: value},
    )

    assert result.is_error
    assert arg in text_of(result)
    assert classifier.prompts == []


# --- Tool surface (ticket #22) ---


async def test_server_lists_its_tools_and_get_signal_says_advisory_and_price(server_url):
    async with (
        streamable_http_client(server_url(FakeFetcher(history=fake_history()))) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = {t.name: t for t in (await session.list_tools()).tools}

    assert sorted(tools) == [
        "get_fundamentals",
        "get_portfolio",
        "get_quote",
        "get_signal",
        "get_watchlists",
        "scan",
        "set_position",
        "set_watchlist",
    ]
    description = (tools["get_signal"].description or "").lower()
    assert "advisory" in description
    assert "price (last close)" in description
    assert "min_confidence null means no band beats break-even" in description
    assert "position" in description and "unrealised p&l" in description


@pytest.mark.parametrize("name", ["get_plan", "get_history"])
async def test_removed_tools_fail_as_unknown(server_url, name):
    result = await call_tool(server_url(FakeFetcher(history=fake_history())), name, {"ticker": "AAPL"})

    assert result.is_error
    assert "unknown tool" in text_of(result).lower()


# --- Intervals (ticket #28) ---


async def test_interval_outside_the_allowlist_is_a_tool_error(server_url):
    classifier = FakeClassifier()
    fetcher = FakeFetcher(history=intraday_history())
    result = await call_tool(
        server_url(fetcher, classifier=classifier),
        "get_signal",
        {"ticker": "AAPL", "interval": "30m"},
    )

    assert result.is_error
    assert "interval" in text_of(result)
    assert fetcher.history_calls == []
    assert classifier.prompts == []


async def test_intraday_data_date_is_the_latest_bars_timestamp(server_url, monkeypatch):
    # Real transport: data_date comes from the prompt's last row. The fake's 60th
    # 5m bar opens 2024-03-01 13:55 WIB, given in the exchange's offset as scan does.
    posts = []

    def fake_post(url, **kwargs):
        posts.append(kwargs)
        return FakeResponse(_decisions_body(_choice_answer(choice="hold")))

    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr("stock_market_mcp.classifier.requests.post", fake_post)
    result = await call_tool(
        server_url(FakeFetcher(history=intraday_history())),
        "get_signal",
        {"ticker": "BBCA.JK", "interval": "5m"},
    )

    assert not result.is_error, text_of(result)
    data_date = json.loads(text_of(result))["data_date"]
    assert data_date == "2024-03-01T13:55:00+07:00"


async def test_daily_data_date_matches_scans_for_the_same_last_bar(server_url, monkeypatch):
    # Real transport: data_date comes from the prompt's last row; the 5m bars stop the day
    # before, so scan's data_date is the daily bar's too.
    monkeypatch.setenv("SMM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        "stock_market_mcp.classifier.requests.post",
        lambda url, **kwargs: FakeResponse(_decisions_body(_choice_answer(choice="hold"))),
    )
    daily = daily_bars([100.0 + i for i in range(60)], last="2024-03-15")
    intraday = five_minute_bars({"2024-03-14 16:10": (158.0, 1000)}, [], tz="Asia/Jakarta")
    fetcher = FakeFetcher(history={"BBCA.JK": {"1d": daily, "5m": intraday}})
    # 2024-03-15 10:00 WIB, during the IDX session.
    url = server_url(fetcher, now=lambda: datetime(2024, 3, 15, 3, 0, tzinfo=UTC))

    signal = await call_tool(url, "get_signal", {"ticker": "BBCA.JK", "interval": "1d"})
    scan = await call_tool(url, "scan", {"tickers": ["BBCA.JK"]})

    assert not signal.is_error, text_of(signal)
    assert scan.structured_content is not None
    assert json.loads(text_of(signal))["data_date"] == scan.structured_content["result"][0]["data_date"]
    assert json.loads(text_of(signal))["data_date"] == "2024-03-15T00:00:00+07:00"
