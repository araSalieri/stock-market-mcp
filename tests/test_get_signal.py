"""Tests for get_signal over MCP: one happy path, argument checks, error wrapping and the tool surface; Signal rules are tested on SignalMaker, the classifier transport in test_classifier.py."""

import json

import pytest
from fakes import (
    FakeClassifier,
    FakeFetcher,
    call_tool,
    fake_history,
    flat_history,
    intraday_history,
    text_of,
)
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


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
        "get_track_record",
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


