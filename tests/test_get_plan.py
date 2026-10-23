"""MCP-surface tests for get_plan: Trade Plan (Entry, Stop, Target) from ATR(14)."""

import json

import pandas as pd
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from test_get_quote import call_tool, text_of

from stock_market_mcp.data import DataFetcher


def flat_history(close: float, rows: int = 20) -> pd.DataFrame:
    """Flat close with High/Low one unit away: True Range is 2 every row, so ATR(14) is 2."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1000,
        },
        index=idx,
    )


class PlanFakeFetcher(DataFetcher):
    def __init__(self, close: float = 100.0, rows: int = 20):
        self.history_calls = []
        self.close = close
        self.rows = rows

    def fetch_quote(self, ticker):
        raise AssertionError("get_plan tests never fetch quotes")

    def fetch_fundamentals(self, ticker):
        raise AssertionError("get_plan tests never fetch fundamentals")

    def fetch_history(self, ticker, period):
        self.history_calls.append((ticker, period))
        if ticker.upper() != "AAPL":
            raise LookupError(f"no history data for ticker: {ticker}")
        return flat_history(self.close, self.rows)


async def test_long_plan_with_defaults(server_url):
    result = await call_tool(
        server_url(PlanFakeFetcher()), "get_plan", {"ticker": "aapl", "direction": "long"}
    )

    assert not result.is_error, text_of(result)
    plan = json.loads(text_of(result))
    assert plan == {
        "ticker": "AAPL",
        "direction": "long",
        "entry": 100.0,
        "stop": pytest.approx(96.0),
        "target": pytest.approx(108.0),
        "atr": pytest.approx(2.0),
        "atr_multiple": 2,
        "reward_ratio": 2,
        "data_date": "2024-01-20",
    }


async def test_get_plan_is_the_fifth_tool_and_says_advisory(server_url):
    url = server_url(PlanFakeFetcher())
    async with (
        streamable_http_client(url) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = {t.name: t for t in (await session.list_tools()).tools}

    assert sorted(tools) == [
        "get_fundamentals", "get_history", "get_plan", "get_quote", "get_signal"
    ]
    assert "advisory" in (tools["get_plan"].description or "").lower()


async def test_unknown_ticker_is_a_loud_tool_error(server_url):
    result = await call_tool(
        server_url(PlanFakeFetcher()), "get_plan", {"ticker": "NOPE", "direction": "long"}
    )

    assert result.is_error
    assert "NOPE" in text_of(result)


async def test_too_few_rows_for_atr_is_a_loud_tool_error(server_url):
    result = await call_tool(
        server_url(PlanFakeFetcher(rows=5)),
        "get_plan",
        {"ticker": "AAPL", "direction": "long"},
    )

    assert result.is_error
    assert "not enough history rows" in text_of(result)


async def test_non_positive_stop_is_a_loud_tool_error_not_clamped(server_url):
    # Entry 1, ATR 2: Stop = 1 - 2*2 = -3.
    result = await call_tool(
        server_url(PlanFakeFetcher(close=1.0)),
        "get_plan",
        {"ticker": "AAPL", "direction": "long"},
    )

    assert result.is_error
    assert "non-positive" in text_of(result)


async def test_works_with_no_classifier_key(server_url, monkeypatch):
    monkeypatch.delenv("SMM_OPENROUTER_API_KEY", raising=False)
    result = await call_tool(
        server_url(PlanFakeFetcher()), "get_plan", {"ticker": "AAPL", "direction": "long"}
    )

    assert not result.is_error, text_of(result)
    assert json.loads(text_of(result))["entry"] == 100.0


async def test_second_call_within_history_ttl_does_not_refetch(server_url):
    fetcher = PlanFakeFetcher()
    url = server_url(fetcher)
    first = await call_tool(url, "get_plan", {"ticker": "AAPL", "direction": "long"})
    second = await call_tool(url, "get_plan", {"ticker": "AAPL", "direction": "long"})

    assert not first.is_error and not second.is_error
    assert text_of(first) == text_of(second)
    assert len(fetcher.history_calls) == 1
