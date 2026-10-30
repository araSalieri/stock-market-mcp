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


def audit_lines(caplog) -> list:
    return [r for r in caplog.records if " outcome=" in r.getMessage()]


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

    def fetch_history(self, ticker, period, interval="1d", prepost=False):
        self.history_calls.append((ticker, period, interval, prepost))
        return fake_history(60)


class NoCloseFakeFetcher(SignalFakeFetcher):
    """History without a Close column: get_signal hits a plain KeyError, a bug path no ToolError wraps."""

    def fetch_history(self, ticker, period, interval="1d", prepost=False):
        self.history_calls.append((ticker, period, interval, prepost))
        return fake_history(60).drop(columns="Close")


def flat_history(close: float = 100.0, rows: int = 60) -> pd.DataFrame:
    """Flat close with High/Low one unit away: True Range is 2 every row, so ATR(14) is 2."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", name="Date")
    return pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000},
        index=idx,
    )


def intraday_history(
    close: float = 100.0, rows: int = 60, spread: float = 0.25, tz: str = "Asia/Jakarta"
) -> pd.DataFrame:
    """5m bars in an exchange timezone, Yahoo's `Datetime` index, flat close with High/Low
    `spread` away: True Range is 2 * spread every row, so ATR(14) is 2 * spread."""
    idx = pd.date_range("2024-03-01 09:00", periods=rows, freq="5min", tz=tz, name="Datetime")
    return pd.DataFrame(
        {"Open": close, "High": close + spread, "Low": close - spread, "Close": close, "Volume": 1000},
        index=idx,
    )


class IntradayFakeFetcher(SignalFakeFetcher):
    def fetch_history(self, ticker, period, interval="1d", prepost=False):
        self.history_calls.append((ticker, period, interval, prepost))
        return intraday_history()


class FlatFakeFetcher(SignalFakeFetcher):
    def __init__(self, close: float = 100.0, rows: int = 60, last_close: float | None = None):
        super().__init__()
        self.close = close
        self.rows = rows
        self.last_close = last_close

    def fetch_history(self, ticker, period, interval="1d", prepost=False):
        self.history_calls.append((ticker, period, interval, prepost))
        df = flat_history(self.close, self.rows)
        if self.last_close is not None:
            df.loc[df.index[-1], "Close"] = self.last_close
        return df


class SequenceFakeFetcher(SignalFakeFetcher):
    """History frames handed out one per fetch_history call; the last one repeats."""

    def __init__(self, frames: list[pd.DataFrame]):
        super().__init__()
        self.frames = frames

    def fetch_history(self, ticker, period, interval="1d", prepost=False):
        self.history_calls.append((ticker, period, interval, prepost))
        return self.frames[min(len(self.history_calls), len(self.frames)) - 1]


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


def daily_bars(
    closes: list[float], last: str = "2024-03-15", tz: str = "Asia/Jakarta"
) -> pd.DataFrame:
    """Daily bars at local midnight in the exchange timezone, as Yahoo returns them; High/Low one unit from Close."""
    idx = pd.date_range(end=last, periods=len(closes), freq="D", tz=tz, name="Date")
    close = pd.Series(closes, index=idx)
    return pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000}
    )


def five_minute_bars(
    bars: dict[str, tuple[float, float]], regular: list[tuple[str, str]], tz: str = "America/New_York"
) -> pd.DataFrame:
    """5m bars as Yahoo returns them with prepost=True: `Datetime` index in the exchange timezone.

    `bars` maps a local time to (close, volume), with Open/High/Low at the close. `regular` is the
    regular trading periods as (start, end) local times, carried in attrs as YfinanceFetcher does.
    """
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz=tz) for t in bars], name="Datetime")
    close = [c for c, _ in bars.values()]
    df = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": [v for _, v in bars.values()]},
        index=idx,
    )
    df.attrs["regular_periods"] = [(pd.Timestamp(s, tz=tz), pd.Timestamp(e, tz=tz)) for s, e in regular]
    return df


class ScanFakeFetcher:
    """History by ticker, then by Interval ("1d" or "5m"); a value that is an exception is raised."""

    def __init__(self, frames: dict):
        self.frames = frames
        self.history_calls = []

    def fetch_quote(self, ticker):
        raise AssertionError("scan never fetches quotes")

    def fetch_fundamentals(self, ticker):
        raise AssertionError("scan never fetches fundamentals")

    def fetch_history(self, ticker, period, interval="1d", prepost=False):
        self.history_calls.append((ticker, period, interval, prepost))
        frame = self.frames.get(ticker, {}).get(interval)
        if frame is None:
            raise LookupError(f"no history data for ticker: {ticker}")
        if isinstance(frame, Exception):
            raise frame
        return frame
