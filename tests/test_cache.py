"""MCP-surface tests for the TTL cache: asserted via fetcher call counts, no network I/O."""

import time

import pandas as pd
from test_get_quote import call_tool, text_of

from stock_market_mcp.data import DataFetcher

# Short TTLs so tests can cross the expiry window without waiting on real timeouts.
TEST_QUOTE_TTL = 0.2
TEST_HISTORY_TTL = 0.2
TEST_FUNDAMENTALS_TTL = 0.2


class CountingFakeFetcher(DataFetcher):
    """Fake data layer that records every fetch so tests can count invocations."""

    def __init__(self):
        self.quote_calls = []
        self.history_calls = []
        self.fundamentals_calls = []

    def fetch_quote(self, ticker):
        self.quote_calls.append(ticker)
        return {"price": 190.0, "currency": "USD", "ticker": ticker.upper()}

    def fetch_fundamentals(self, ticker):
        self.fundamentals_calls.append(ticker)
        return {"ticker": ticker.upper(), "long_name": "Apple Inc."}

    def fetch_history(self, ticker, period):
        self.history_calls.append((ticker, period))
        idx = pd.date_range("2024-01-01", periods=5, freq="h", name="Date")
        close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=idx)
        return pd.DataFrame(
            {
                "Open": close - 0.1,
                "High": close + 0.1,
                "Low": close - 0.2,
                "Close": close,
                "Volume": [1000] * 5,
            }
        )


async def test_second_quote_within_ttl_is_served_from_cache(server_url):
    fetcher = CountingFakeFetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    first = await call_tool(url, "get_quote", {"ticker": "AAPL"})
    second = await call_tool(url, "get_quote", {"ticker": "AAPL"})

    assert not first.is_error
    assert not second.is_error
    assert text_of(second) == text_of(first)
    assert fetcher.quote_calls == ["AAPL"]  # second call hit the cache


async def test_quote_after_ttl_expiry_refetches(server_url):
    fetcher = CountingFakeFetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    await call_tool(url, "get_quote", {"ticker": "AAPL"})
    time.sleep(TEST_QUOTE_TTL + 0.1)
    await call_tool(url, "get_quote", {"ticker": "AAPL"})

    assert fetcher.quote_calls == ["AAPL", "AAPL"]


async def test_second_history_within_ttl_is_served_from_cache(server_url):
    fetcher = CountingFakeFetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    first = await call_tool(url, "get_history", {"ticker": "AAPL", "period": "1mo"})
    second = await call_tool(url, "get_history", {"ticker": "AAPL", "period": "1mo"})

    assert not first.is_error
    assert not second.is_error
    assert fetcher.history_calls == [("AAPL", "1mo")]  # second call hit the cache


async def test_history_after_ttl_expiry_refetches(server_url):
    fetcher = CountingFakeFetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    await call_tool(url, "get_history", {"ticker": "AAPL", "period": "1mo"})
    time.sleep(TEST_HISTORY_TTL + 0.1)
    await call_tool(url, "get_history", {"ticker": "AAPL", "period": "1mo"})

    assert fetcher.history_calls == [("AAPL", "1mo"), ("AAPL", "1mo")]


async def test_second_fundamentals_within_ttl_is_served_from_cache(server_url):
    fetcher = CountingFakeFetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    first = await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})
    second = await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})

    assert not first.is_error
    assert not second.is_error
    assert fetcher.fundamentals_calls == ["AAPL"]  # second call hit the cache


async def test_fundamentals_after_ttl_expiry_refetches(server_url):
    fetcher = CountingFakeFetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})
    time.sleep(TEST_FUNDAMENTALS_TTL + 0.1)
    await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})

    assert fetcher.fundamentals_calls == ["AAPL", "AAPL"]
