import threading
import time

import pytest
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult
from uvicorn import Config, Server

from stock_market_mcp.data import DataFetcher
from stock_market_mcp.server import build_server


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


@pytest.fixture
def server_url():
    urls = []

    def start(fetcher):
        app = build_server(fetcher).streamable_http_app()
        config = Config(app, host="127.0.0.1", port=0, log_level="warning")
        server = Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        for _ in range(100):
            if server.started and server.servers:
                break
            time.sleep(0.05)
        url = f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}/mcp"
        urls.append(url)
        return url

    yield start
    assert urls  # harness sanity: server was started


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


async def test_get_quote_returns_price_for_known_ticker(server_url):
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    assert not result.is_error
    assert "190.0" in text_of(result)
