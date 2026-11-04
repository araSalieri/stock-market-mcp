"""Price history for one Interval: its period rules and the indicators joined onto the bars."""

import math
from datetime import timedelta
from typing import Literal, NamedTuple, cast

import pandas as pd
import pandas_ta as ta

from stock_market_mcp.data import DataFetcher
from stock_market_mcp.errors import InputError

Interval = Literal["1m", "5m", "15m", "1h", "1d"]

# Every period Yahoo knows, as yfinance lists them.
_YAHOO_PERIODS = ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")


class _Periods(NamedTuple):
    default: str
    # Yahoo's history limit and the periods that fit inside it; daily bars have none and pass any period on.
    limit: str | None
    allowed: tuple[str, ...]


_PERIODS: dict[Interval, _Periods] = {
    "1m": _Periods("5d", "7 days", ("1d", "5d")),
    "5m": _Periods("1mo", "60 days", ("1d", "5d", "1mo")),
    "15m": _Periods("1mo", "60 days", ("1d", "5d", "1mo")),
    "1h": _Periods("6mo", "730 days", ("1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "ytd")),
    "1d": _Periods("1y", None, ()),
}


# How long one bar of each Interval lasts: a bar is completed once its start plus this is at or before the clock.
BAR_LENGTH: dict[Interval, timedelta] = {
    "1m": timedelta(minutes=1),
    "5m": timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "1h": timedelta(hours=1),
    "1d": timedelta(days=1),
}


def fetch_period(interval: Interval, period: str | None) -> str:
    """The period to fetch: the Interval's default, or the caller's once it fits the Interval."""
    periods = _PERIODS[interval]
    if period is None:
        return periods.default
    if periods.limit is None or period in periods.allowed:
        return period
    if period in _YAHOO_PERIODS:
        reason = f"is longer than Yahoo keeps for {interval} bars ({periods.limit})"
    else:
        reason = f"is not a valid period for {interval} bars"
    raise InputError(f"period {period!r} {reason}; use one of {', '.join(periods.allowed)}")


def _suffixed_frame(ind: pd.Series | None) -> pd.DataFrame | None:
    """Frame a single-output indicator; pandas-ta returns None on insufficient rows."""
    if ind is None:
        return None
    return ind.to_frame()


# Bars in the EMA, shared by the ema indicator and the Re-entry Lock.
EMA_LENGTH = 10


# Dispatch table: indicator name -> handler producing the indicator frame(s),
# or None when the history is too short for that indicator (loud error path).
_INDICATOR_DISPATCH = {
    "sma": lambda df: _suffixed_frame(ta.sma(cast("pd.Series", df["Close"]))),
    "ema": lambda df: _suffixed_frame(ta.ema(cast("pd.Series", df["Close"]), length=EMA_LENGTH)),
    "rsi": lambda df: _suffixed_frame(ta.rsi(cast("pd.Series", df["Close"]))),
    "macd": lambda df: ta.macd(cast("pd.Series", df["Close"])),
    "bbands": lambda df: ta.bbands(cast("pd.Series", df["Close"])),
    "atr": lambda df: _suffixed_frame(
        ta.atr(
            cast("pd.Series", df["High"]),
            cast("pd.Series", df["Low"]),
            cast("pd.Series", df["Close"]),
        )
    ),
    "adx": lambda df: ta.adx(
        cast("pd.Series", df["High"]),
        cast("pd.Series", df["Low"]),
        cast("pd.Series", df["Close"]),
    ),
    "vwap": lambda df: _suffixed_frame(
        ta.vwap(
            cast("pd.Series", df["High"]),
            cast("pd.Series", df["Low"]),
            cast("pd.Series", df["Close"]),
            cast("pd.Series", df["Volume"]),
        )
    ),
}


def last_value(name: str, df: pd.DataFrame) -> float:
    """Last value of a single-column indicator over df."""
    return float(_INDICATOR_DISPATCH[name](df).iloc[-1, 0])


def finite(value: float) -> float | None:
    """JSON has no NaN; a non-finite number is null."""
    return value if math.isfinite(value) else None


def history(fetcher: DataFetcher, ticker: str, interval: Interval, period: str | None) -> pd.DataFrame:
    """Bars of one Interval over the period (the Interval's default when None) with every indicator joined.

    The ticker is upper-cased before the fetch, so any casing shares one cache entry.
    """
    df = fetcher.fetch_history(ticker.upper(), fetch_period(interval, period), interval)
    for name, handler in sorted(_INDICATOR_DISPATCH.items()):
        ind = handler(df)
        if ind is None:
            raise InputError(f"not enough history rows for indicator: {name}")
        df = df.join(ind)
    return df
