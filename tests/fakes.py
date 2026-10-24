"""Shared test fakes and MCP client helpers; test modules import from here, never from each other."""

import pandas as pd
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from stock_market_mcp.data import DataFetcher


async def call_tool(url: str, name: str, args: dict) -> CallToolResult:
    async with (
        streamable_http_client(url) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.call_tool(name, args)


def text_of(result: CallToolResult) -> str:
    return "".join(c.text for c in result.content if isinstance(c, types.TextContent))


def fake_history(rows: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", name="Date")
    close = pd.Series([10.0 + i * 0.5 + (i % 3) for i in range(rows)], index=idx)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": [1000 + i for i in range(rows)],
        }
    )


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
    # which test_get_quote.py's loud-failure test exercises end to end.


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


def flat_history(close: float = 100.0, rows: int = 60) -> pd.DataFrame:
    """Flat close with High/Low one unit away: True Range is 2 every row, so ATR(14) is 2."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", name="Date")
    return pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000},
        index=idx,
    )


class FlatFakeFetcher(SignalFakeFetcher):
    def __init__(self, close: float = 100.0, rows: int = 60, last_close: float | None = None):
        super().__init__()
        self.close = close
        self.rows = rows
        self.last_close = last_close

    def fetch_history(self, ticker, period):
        self.history_calls.append((ticker, period))
        df = flat_history(self.close, self.rows)
        if self.last_close is not None:
            df.loc[df.index[-1], "Close"] = self.last_close
        return df


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
