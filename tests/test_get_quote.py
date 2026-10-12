import pytest
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from stock_market_mcp.data import DataFetcher


class FakeFetcher(DataFetcher):
    def __init__(self, quotes=None, error=None):
        self.quotes = quotes or {}
        self.error = error
        self.quote_calls = []  # asserted on by the TTL-cache ticket

    def fetch_quote(self, ticker):
        self.quote_calls.append(ticker)
        if self.error is not None:
            raise self.error
        quote = self.quotes.get(ticker.upper())
        if quote is None:
            raise LookupError(f"unknown ticker: {ticker}")
        return quote

    def fetch_fundamentals(self, ticker):
        raise AssertionError("get_quote tests never fetch fundamentals")
    # fetch_history intentionally not overridden: relies on the Protocol default,
    # which the loud-failure test below exercises end to end.


async def call_tool(url: str, name: str, args: dict) -> CallToolResult:
    async with (
        streamable_http_client(url) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.call_tool(name, args)


def text_of(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


async def test_get_quote_unknown_ticker_is_a_loud_tool_error(server_url):
    fetcher = FakeFetcher()  # no quotes at all
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "NOPE"})

    assert result.is_error
    assert "NOPE" in text_of(result)


async def test_get_quote_provider_failure_is_a_loud_tool_error(server_url):
    fetcher = FakeFetcher(error=RuntimeError("yahoo is down"))
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    assert result.is_error
    assert "yahoo is down" in text_of(result)


def test_datafetcher_default_fetch_history_fails_loudly_not_empty():
    """The Protocol default must raise, never return empty data (spec: never empty)."""
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})

    with pytest.raises(NotImplementedError, match="fetch_history"):
        fetcher.fetch_history("AAPL", "1mo")


async def test_quote_only_fake_leaking_into_history_is_a_loud_tool_error(server_url):
    # A quote-only fake leaking into get_history must surface the loud failure,
    # not an empty frame (spec: never empty data).
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})
    result = await call_tool(
        server_url(fetcher), "get_history", {"ticker": "AAPL", "period": "1mo"}
    )

    assert result.is_error
    assert "fetch_history" in text_of(result)


async def test_get_quote_returns_price_for_known_ticker(server_url):
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    assert not result.is_error
    assert "190.0" in text_of(result)
