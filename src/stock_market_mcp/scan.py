"""Scan: the Scan numbers for one ticker at a given time, from its daily and intraday bars."""

from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd

from stock_market_mcp.bars import finite, last_value
from stock_market_mcp.data import DataFetcher


def scan_ticker(fetcher: DataFetcher, ticker: str, now: datetime) -> dict:
    """The Price and its time, the previous day's high, low and close, daily ATR % and RSI, the Gap, Relative Volume and VWAP Distance."""
    daily = fetcher.fetch_history(ticker, "3mo", interval="1d")
    intraday = fetcher.fetch_history(ticker, "1mo", interval="5m", prepost=True)
    entry = _scan_daily(daily, now)
    # Pre-market or intraday bars give the Price when they're newer than the daily bar.
    if len(intraday) and intraday.index[-1] >= daily.index[-1]:
        entry["price"] = float(intraday["Close"].iloc[-1])
        entry["data_date"] = cast("pd.Timestamp", intraday.index[-1]).isoformat()
    return entry | _scan_intraday(intraday, now, entry["price"], entry["prev_close"])


def _scan_daily(df: pd.DataFrame, now: datetime) -> dict:
    """Daily Scan numbers; the previous day is the last bar dated before today in the exchange timezone."""
    index = cast("pd.DatetimeIndex", df.index)
    session_start = now.astimezone(index.tz).replace(hour=0, minute=0, second=0, microsecond=0)
    before = df.loc[index < session_start]
    if before.empty:
        raise LookupError("no daily bar before the current session")
    prev = before.iloc[-1]
    prev_close = float(prev["Close"])
    # Completed bars only, so today's forming bar can't move ATR or RSI.
    atr = last_value("atr", before)
    rsi = last_value("rsi", before)
    return {
        "price": float(df["Close"].iloc[-1]),
        "data_date": cast("pd.Timestamp", index[-1]).isoformat(),
        "prev_high": float(prev["High"]),
        "prev_low": float(prev["Low"]),
        "prev_close": prev_close,
        "atr_pct": finite(atr / prev_close * 100),
        "rsi": finite(rsi),
    }


RELATIVE_VOLUME_SESSIONS = 10


def _pct(value: float, base: float) -> float:
    return (value / base - 1) * 100


def _scan_intraday(df: pd.DataFrame, now: datetime, price: float, prev_close: float) -> dict:
    """Gap, Relative Volume and VWAP distance from 5m bars with pre-market.

    A bar is regular-session when it falls inside one of Yahoo's regular trading periods;
    everything else is pre- or post-market.
    """
    periods = df.attrs.get("regular_periods")
    if periods is None:
        raise LookupError("5m history has no regular trading periods")
    index = cast("pd.DatetimeIndex", df.index)
    today = now.astimezone(index.tz).date()
    days = np.array([t.date() for t in index])
    times = np.array([t.time() for t in index])
    regular = np.array([any(start <= t < end for start, end in periods) for t in index], dtype=bool)
    in_session = (days == today) & regular
    close = df["Close"].to_numpy(dtype=float)
    volume = df["Volume"].to_numpy(dtype=float)
    if not in_session.any():
        pre_market = close[(days == today) & ~regular]
        gap = finite(_pct(float(pre_market[-1]), prev_close)) if len(pre_market) else None
        return {"gap_pct": gap, "relative_volume": None, "vwap_distance_pct": None}

    session_volume = float(volume[in_session].sum())
    so_far = times <= times[in_session][-1]
    prior_days = set(sorted(set(days[regular & (days < today)]))[-RELATIVE_VOLUME_SESSIONS:])
    prior = regular & so_far & np.array([d in prior_days for d in days], dtype=bool)
    average = float(volume[prior].sum()) / len(prior_days) if prior_days else 0.0
    typical = (df["High"].to_numpy(dtype=float) + df["Low"].to_numpy(dtype=float) + close) / 3
    vwap = float((typical * volume)[in_session].sum()) / session_volume if session_volume else None
    return {
        "gap_pct": finite(_pct(float(df["Open"].to_numpy(dtype=float)[in_session][0]), prev_close)),
        "relative_volume": finite(session_volume / average) if average else None,
        "vwap_distance_pct": finite(_pct(price, vwap)) if vwap else None,
    }
