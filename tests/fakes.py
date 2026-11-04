"""Shared test fakes and MCP client helpers; test modules import from here, never from each other."""

import sqlite3
from contextlib import closing

import pandas as pd
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from stock_market_mcp.prompt import PROMPT_VERSION
from stock_market_mcp.signals import SignalLog


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


def seed_signals(db, outcomes, *, verdict="buy", confidence=0.7, prompt_version=PROMPT_VERSION, ticker="AAPL", interval="1d", cut_loss=None, long_plan=True, short_plan=True, re_entry=None):
    """Store one scored Signal per Outcome (None for pending): scoring dozens of Signals over MCP would take dozens of calls.

    long_plan and short_plan False store that default Trade Plan as nulls; re_entry tags them as the first buy after a lift by that path.

    The only test code that knows the signals schema; SignalLog(db).track_record creates the table first.
    """
    SignalLog(db).track_record(None, None)
    long = (96.0, 108.0) if long_plan else (None, None)
    short = (104.0, 92.0) if short_plan else (None, None)
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.executemany(
            "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, outcome, "
            "cut_loss, long_stop, long_target, short_stop, short_target, re_entry) "
            "VALUES (?, ?, ?, '2023-01-01T00:00:00', ?, ?, 100.0, ?, ?, ?, ?, ?, ?, ?)",
            [
                (ticker, interval, prompt_version, verdict, confidence, outcome, cut_loss, *long, *short, re_entry)
                for outcome in outcomes
            ],
        )


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


class FakeFetcher:
    """Fake data layer that records every call.

    quotes and fundamentals map an upper-cased ticker to its reply; a missing ticker is a LookupError.
    Without them, fetching one fails loudly, so tests can prove a tool never quotes.
    history is a frame for every call, a list of frames handed out one per call (the last repeats),
    or a dict of ticker to {Interval: frame}, where a missing entry is a LookupError and an
    exception value is raised. Without history, fetch_history fails loudly. error is raised by every fetch.
    """

    def __init__(self, quotes=None, fundamentals=None, history=None, error=None):
        self.quotes = quotes
        self.fundamentals = fundamentals
        self.history = history
        self.error = error
        self.quote_calls = []
        self.fundamentals_calls = []
        self.history_calls = []

    def fetch_quote(self, ticker):
        self.quote_calls.append(ticker)
        return self._reply(self.quotes, "fetch_quote", ticker)

    def fetch_fundamentals(self, ticker):
        self.fundamentals_calls.append(ticker)
        return self._reply(self.fundamentals, "fetch_fundamentals", ticker)

    def fetch_history(self, ticker, period, interval="1d", prepost=False):
        self.history_calls.append((ticker, period, interval, prepost))
        if self.error is not None:
            raise self.error
        if self.history is None:
            raise AssertionError("this fake has no history: fetch_history was not expected")
        if isinstance(self.history, list):
            frame = self.history[min(len(self.history_calls), len(self.history)) - 1]
        elif isinstance(self.history, dict):
            frame = self.history.get(ticker, {}).get(interval)
            if frame is None:
                raise LookupError(f"no history data for ticker: {ticker}")
            if isinstance(frame, Exception):
                raise frame
        else:
            frame = self.history
        return frame

    def _reply(self, replies, method, ticker):
        if self.error is not None:
            raise self.error
        if replies is None:
            raise AssertionError(f"this fake has no replies: {method} was not expected")
        reply = replies.get(ticker.upper())
        if reply is None:
            raise LookupError(f"unknown ticker: {ticker}")
        return reply


def flat_history(close: float = 100.0, rows: int = 60, last_close: float | None = None) -> pd.DataFrame:
    """Flat close with High/Low one unit away: True Range is 2 every row, so ATR(14) is 2.

    last_close (optional) replaces the last row's Close only."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", name="Date")
    df = pd.DataFrame(
        {"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000},
        index=idx,
    )
    if last_close is not None:
        df.loc[df.index[-1], "Close"] = last_close
    return df


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
        }
        self.error = error
        self.replies = list(replies) if replies is not None else None
        self.prompts = []

    def classify(self, prompt, held):
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
