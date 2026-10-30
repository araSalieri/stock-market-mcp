"""MCP server assembly: tools registered against an injectable data fetcher."""

import functools
import inspect
import logging
import math
import os
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Annotated, Literal, NamedTuple, cast

import numpy as np
import pandas as pd
import pandas_ta as ta
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Strict

from stock_market_mcp.cache import (
    FUNDAMENTALS_TTL_SECONDS,
    HISTORY_TTL_SECONDS,
    QUOTE_TTL_SECONDS,
    SIGNAL_TTL_SECONDS,
    CachingClassifier,
    CachingFetcher,
)
from stock_market_mcp.classifier import (
    PROMPT_VERSION,
    EnforcingClassifier,
    OpenRouterClassifier,
    SignalClassifier,
)
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.db import DEFAULT_DB_PATH
from stock_market_mcp.prompt import build_signal_prompt
from stock_market_mcp.signals import SignalLog
from stock_market_mcp.watchlists import (
    WATCHLIST_MAX_TICKERS,
    WatchlistStore,
)

log = logging.getLogger(__name__)


def _error_line(tool: str, exc: BaseException) -> str:
    return f"tool={tool} error={exc}"


def _audited[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """Log one audit line per call: tool name, arguments, ms and outcome, even when the tool raises."""
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        fields = " ".join(f"{k}={v}" for k, v in bound.arguments.items())
        start = time.perf_counter()
        outcome = "error"
        try:
            result = tool(*args, **kwargs)
            outcome = "ok"
            return result
        except ToolError as exc:
            # A __cause__ means the tool wrapped a dependency failure; without one it rejected the input or data itself.
            if exc.__cause__ is not None:
                log.error(_error_line(tool.__name__, exc), exc_info=exc.__cause__)
            else:
                log.warning(_error_line(tool.__name__, exc))
            raise
        except Exception as exc:
            log.exception(_error_line(tool.__name__, exc))
            raise
        finally:
            ms = round((time.perf_counter() - start) * 1000)
            log.info(" ".join(filter(None, (f"tool={tool.__name__}", fields, f"ms={ms}", f"outcome={outcome}"))))

    return wrapper


def _suffixed_frame(ind: pd.Series | None) -> pd.DataFrame | None:
    """Frame a single-output indicator; pandas-ta returns None on insufficient rows."""
    if ind is None:
        return None
    return ind.to_frame()


# Dispatch table: indicator name -> handler producing the indicator frame(s),
# or None when the history is too short for that indicator (loud error path).
_INDICATOR_DISPATCH = {
    "sma": lambda df: _suffixed_frame(ta.sma(cast("pd.Series", df["Close"]))),
    "ema": lambda df: _suffixed_frame(ta.ema(cast("pd.Series", df["Close"]))),
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

# Curated allowlist: the indicators get_signal joins for the classifier.
# Derived from the dispatch table so adding a branch extends the allowlist too.
INDICATOR_ALLOWLIST = frozenset(_INDICATOR_DISPATCH)


PLAN_ATR_MULTIPLE = 2.0
PLAN_REWARD_RATIO = 2.0
PLAN_ARG_MAX = 10
MIN_CONFIDENCE = 0.6

# Strict: JSON numbers only; true and "2" are rejected, not coerced.
StrictNumber = Annotated[float, Strict()]

Direction = Literal["long", "short"]

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


def _period(interval: Interval, period: str | None) -> str:
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
    raise ToolError(f"period {period!r} {reason}; use one of {', '.join(periods.allowed)}")


def _tickers(tickers: list[str]) -> list[str]:
    """Upper-cased and de-duplicated in order; one Scan or Watchlist takes at most WATCHLIST_MAX_TICKERS."""
    names = list(dict.fromkeys(t.upper() for t in tickers))
    if len(names) > WATCHLIST_MAX_TICKERS:
        raise ToolError(f"at most {WATCHLIST_MAX_TICKERS} tickers, got {len(names)}")
    return names


# Verdict -> Trade Plan Direction; hold has no plan.
_VERDICT_DIRECTION: dict[str, Direction] = {"buy": "long", "sell": "short"}


def _trade_plan(
    df: pd.DataFrame, entry: float, direction: Direction, atr_multiple: float, reward_ratio: float
) -> dict:
    """Stop and Target around entry (the Price) from the last ATR(14) of df."""
    atr = float(_indicator_frame("atr", df).iloc[-1, 0])
    if not math.isfinite(atr):
        raise ToolError(f"latest row has a non-finite ATR (atr={atr})")
    sign = 1 if direction == "long" else -1
    stop = entry - sign * atr_multiple * atr
    # Same line for both Directions: short Target = Entry - ratio * (Stop - Entry).
    target = entry + reward_ratio * (entry - stop)
    if stop <= 0 or target <= 0:
        raise ToolError(
            f"plan has non-positive prices (stop={stop}, target={target}); "
            "ATR is too large relative to the last close"
        )
    return {
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "atr_multiple": atr_multiple,
        "reward_ratio": reward_ratio,
    }


def _default_plan(df: pd.DataFrame, entry: float, direction: Direction) -> tuple[float | None, float | None]:
    """Stop and Target of the default Trade Plan stored with a Signal; nulls when that plan can't be made."""
    try:
        plan = _trade_plan(df, entry, direction, PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO)
    except ToolError:
        return None, None
    return plan["stop"], plan["target"]


def _indicator_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Indicator values for one allowed name, keyed by pandas-ta's own column names."""
    handler = _INDICATOR_DISPATCH.get(name)
    if handler is None:
        raise ToolError(f"unsupported indicator: {name}")
    return handler(df)



def _finite(value: float) -> float | None:
    """JSON has no NaN; an indicator without a value is null."""
    return value if math.isfinite(value) else None


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
    atr = float(_indicator_frame("atr", before).iloc[-1, 0])
    rsi = float(_indicator_frame("rsi", before).iloc[-1, 0])
    return {
        "price": float(df["Close"].iloc[-1]),
        "data_date": cast("pd.Timestamp", index[-1]).isoformat(),
        "prev_high": float(prev["High"]),
        "prev_low": float(prev["Low"]),
        "prev_close": prev_close,
        "atr_pct": _finite(atr / prev_close * 100),
        "rsi": _finite(rsi),
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
        gap = _finite(_pct(float(pre_market[-1]), prev_close)) if len(pre_market) else None
        return {"gap_pct": gap, "relative_volume": None, "vwap_distance_pct": None}

    session_volume = float(volume[in_session].sum())
    so_far = times <= times[in_session][-1]
    prior_days = set(sorted(set(days[regular & (days < today)]))[-RELATIVE_VOLUME_SESSIONS:])
    prior = regular & so_far & np.array([d in prior_days for d in days], dtype=bool)
    average = float(volume[prior].sum()) / len(prior_days) if prior_days else 0.0
    typical = (df["High"].to_numpy(dtype=float) + df["Low"].to_numpy(dtype=float) + close) / 3
    vwap = float((typical * volume)[in_session].sum()) / session_volume if session_volume else None
    return {
        "gap_pct": _finite(_pct(float(df["Open"].to_numpy(dtype=float)[in_session][0]), prev_close)),
        "relative_volume": _finite(session_volume / average) if average else None,
        "vwap_distance_pct": _finite(_pct(price, vwap)) if vwap else None,
    }


def build_server(
    fetcher: DataFetcher,
    classifier: SignalClassifier | None = None,
    *,
    quote_ttl: float = QUOTE_TTL_SECONDS,
    history_ttl: float = HISTORY_TTL_SECONDS,
    fundamentals_ttl: float = FUNDAMENTALS_TTL_SECONDS,
    signal_ttl: float = SIGNAL_TTL_SECONDS,
    db_path: str | os.PathLike | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> MCPServer:
    """Assemble the MCP server over an injectable fetcher and classifier.

    The TTL parameters exist so tests can shrink the cache windows; production
    callers use the module-level defaults (60s quotes and history, 300s the rest).
    `now` is the clock the Scan uses to tell which session is current.
    """
    fetcher = CachingFetcher(
        fetcher,
        quote_ttl=quote_ttl,
        history_ttl=history_ttl,
        fundamentals_ttl=fundamentals_ttl,
    )
    signal_classifier = CachingClassifier(
        EnforcingClassifier(classifier if classifier is not None else OpenRouterClassifier()),
        signal_ttl=signal_ttl,
    )
    db_path = db_path or os.environ.get("SMM_DB_PATH", DEFAULT_DB_PATH)
    watchlists = WatchlistStore(db_path)
    signals = SignalLog(db_path)
    mcp = MCPServer("stock-market-mcp")

    def audited_tool[F: Callable](fn: F) -> F:
        mcp.tool()(_audited(fn))
        return fn

    @audited_tool
    def get_quote(ticker: str) -> dict:
        """Current quote for a stock ticker (price and currency)."""
        try:
            return fetcher.fetch_quote(ticker)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @audited_tool
    def get_fundamentals(ticker: str) -> dict:
        """Key company fundamentals (name, sector, industry, market cap)."""
        try:
            return fetcher.fetch_fundamentals(ticker)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @audited_tool
    def get_signal(
        ticker: str,
        interval: Interval = "5m",
        period: str | None = None,
        atr_multiple: StrictNumber = PLAN_ATR_MULTIPLE,
        reward_ratio: StrictNumber = PLAN_REWARD_RATIO,
        min_confidence: StrictNumber = MIN_CONFIDENCE,
    ) -> dict:
        """Advisory buy/sell/hold Signal from price history and indicators on bars of one Interval, with the Price (last close), the Hit Rate of past Signals of its kind (buy/sell or hold) for this ticker and Interval (null until enough are scored), and for buy or sell a Trade Plan (Entry, Stop, Target); buy or sell below min_confidence gets no Trade Plan (null); input only, never acts on it."""
        for name, value in (("atr_multiple", atr_multiple), ("reward_ratio", reward_ratio)):
            if not 0 < value <= PLAN_ARG_MAX:
                raise ToolError(f"{name} must be in (0, {PLAN_ARG_MAX}], got {value}")
        if not 0 <= min_confidence <= 1:
            raise ToolError(f"min_confidence must be in [0, 1], got {min_confidence}")
        period = _period(interval, period)
        try:
            df = fetcher.fetch_history(ticker, period, interval)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        for name in sorted(INDICATOR_ALLOWLIST):
            ind = _indicator_frame(name, df)
            if ind is None:
                raise ToolError(f"not enough history rows for indicator: {name}")
            df = df.join(ind)
        # Read outside the classifier cache, so cached Signals get a fresh Price and plan.
        price = float(df["Close"].iloc[-1])
        if not math.isfinite(price):
            raise ToolError(f"latest row has a non-finite close (close={price})")
        ticker = ticker.upper()
        try:
            signals.score(ticker, interval, df, now().isoformat())
            previous = signals.latest(ticker, interval)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        # Cached by the prompt without the previous Signal, so storing a Signal doesn't defeat the cache.
        key = build_signal_prompt(ticker, period, df, interval=interval)
        prompt = build_signal_prompt(ticker, period, df, interval=interval, previous_signal=previous)
        try:
            signal, fresh = signal_classifier.classify(prompt, key)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        if fresh:
            long_stop, long_target = _default_plan(df, price, "long")
            short_stop, short_target = _default_plan(df, price, "short")
            try:
                signals.add(
                    ticker=ticker,
                    interval=interval,
                    prompt_version=PROMPT_VERSION,
                    data_date=cast("pd.Timestamp", df.index[-1]).isoformat(),
                    verdict=signal["verdict"],
                    confidence=signal["confidence"],
                    price=price,
                    long_stop=long_stop,
                    long_target=long_target,
                    short_stop=short_stop,
                    short_target=short_target,
                )
            except Exception as exc:
                raise ToolError(str(exc)) from exc
        # After the cache, so each call's own min_confidence applies to a cached Signal.
        direction = _VERDICT_DIRECTION.get(signal["verdict"]) if signal["confidence"] >= min_confidence else None
        plan = _trade_plan(df, price, direction, atr_multiple, reward_ratio) if direction else None
        try:
            hit_rate = signals.hit_rate(ticker, interval, PROMPT_VERSION, signal["verdict"])
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        return {**signal, "price": price, "hit_rate": hit_rate, "trade_plan": plan}

    @audited_tool
    def scan(watchlist: str | None = None, tickers: list[str] | None = None) -> list[dict]:
        """Scan a Watchlist or an ad-hoc ticker list: per ticker the Price and its time, the previous day's high, low and close, daily ATR % and RSI, the Gap, Relative Volume and distance from this session's VWAP. No Verdict, no classifier call; a failing ticker gets its own error entry."""
        if (watchlist is None) == (tickers is None):
            raise ToolError("give exactly one of watchlist or tickers")
        if watchlist is not None:
            try:
                lists = watchlists.load()
            except Exception as exc:
                raise ToolError(str(exc)) from exc
            if watchlist not in lists:
                raise ToolError(f"no Watchlist named {watchlist!r}")
            tickers = lists[watchlist]
        names = _tickers(cast("list[str]", tickers))
        results = []
        for ticker in names:
            try:
                daily = fetcher.fetch_history(ticker, "3mo", interval="1d")
                intraday = fetcher.fetch_history(ticker, "1mo", interval="5m", prepost=True)
                clock = now()
                entry = _scan_daily(daily, clock)
                # Pre-market or intraday bars give the Price when they're newer than the daily bar.
                if len(intraday) and intraday.index[-1] >= daily.index[-1]:
                    entry["price"] = float(intraday["Close"].iloc[-1])
                    entry["data_date"] = cast("pd.Timestamp", intraday.index[-1]).isoformat()
                entry |= _scan_intraday(intraday, clock, entry["price"], entry["prev_close"])
                results.append({"ticker": ticker, **entry})
            except Exception as exc:
                log.exception(_error_line(f"scan ticker={ticker}", exc))
                results.append({"ticker": ticker, "error": str(exc)})
        return results

    @audited_tool
    def set_watchlist(name: str, tickers: list[str]) -> dict:
        """Save a named Watchlist (each ticker checked with a quote), replacing its tickers; an empty list deletes it."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):
            raise ToolError("Watchlist name must be 1-32 letters, digits, '-' or '_'")
        tickers = _tickers(tickers)
        bad = []
        for ticker in tickers:
            try:
                fetcher.fetch_quote(ticker)
            except Exception:
                bad.append(ticker)
        if bad:
            raise ToolError(f"unknown tickers, nothing saved: {', '.join(bad)}")
        try:
            watchlists.set(name, tickers)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        return {"name": name, "tickers": tickers}

    @audited_tool
    def get_watchlists() -> dict:
        """Every saved Watchlist as {name: [tickers]}."""
        try:
            return watchlists.load()
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    return mcp
