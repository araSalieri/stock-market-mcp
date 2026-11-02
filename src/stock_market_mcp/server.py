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
from typing import Annotated, Literal, cast

import numpy as np
import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Strict

from stock_market_mcp.bars import Interval, fetch_period, history, last_value
from stock_market_mcp.cache import (
    FUNDAMENTALS_TTL_SECONDS,
    HISTORY_TTL_SECONDS,
    QUOTE_TTL_SECONDS,
    SIGNAL_TTL_SECONDS,
    CachingFetcher,
    TTLCache,
)
from stock_market_mcp.classifier import (
    PROMPT_VERSION,
    EnforcingClassifier,
    OpenRouterClassifier,
    SignalClassifier,
)
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.db import DEFAULT_DB_PATH
from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.prompt import build_signal_prompt, with_previous_signal
from stock_market_mcp.signals import SignalLog
from stock_market_mcp.watchlists import (
    WATCHLIST_MAX_TICKERS,
    WatchlistStore,
)

log = logging.getLogger(__name__)


def _error_line(tool: str, exc: BaseException) -> str:
    return f"tool={tool} error={exc}"


def _audited[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """Log one audit line per call: tool name, arguments, ms and outcome, even when the tool raises.

    An InputError becomes a ToolError with no cause, logged like one the tool raised itself.
    Any other exception becomes a ToolError with it as the cause, so the caller sees its message.
    """
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
        except InputError as exc:
            log.warning(_error_line(tool.__name__, exc))
            raise ToolError(str(exc)) from None
        except Exception as exc:
            log.exception(_error_line(tool.__name__, exc))
            raise ToolError(str(exc)) from exc
        finally:
            ms = round((time.perf_counter() - start) * 1000)
            log.info(" ".join(filter(None, (f"tool={tool.__name__}", fields, f"ms={ms}", f"outcome={outcome}"))))

    return wrapper


PLAN_ATR_MULTIPLE = 2.0
PLAN_REWARD_RATIO = 2.0
PLAN_ARG_MAX = 10
# Hit Rate a default Trade Plan must beat to break even: a Target gains PLAN_REWARD_RATIO times what a Stop loses.
BREAK_EVEN_HIT_RATE = 1 / (1 + PLAN_REWARD_RATIO)

# Strict: JSON numbers only; true and "2" are rejected, not coerced.
StrictNumber = Annotated[float, Strict()]

Direction = Literal["long", "short"]

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
    atr = last_value("atr", df)
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


def _finite(value: float) -> float | None:
    """JSON has no NaN; an indicator without a value is null."""
    return value if math.isfinite(value) else None


def _unrealised_pnl(position: dict, price: float) -> dict:
    """Unrealised P&L of a Position at price, as an amount and a percent of cost; null when non-finite."""
    sign = 1 if position["direction"] == "long" else -1
    pnl = sign * (price - position["avg_cost"]) * position["quantity"]
    return {
        "unrealised_pnl": _finite(pnl),
        "unrealised_pnl_pct": _finite(pnl / (position["avg_cost"] * position["quantity"]) * 100),
    }


def _losing(pnl: dict) -> bool:
    """Whether an Unrealised P&L is a loss; break-even and a non-finite (null) amount are not."""
    return pnl["unrealised_pnl"] is not None and pnl["unrealised_pnl"] < 0


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
    signal_classifier = EnforcingClassifier(classifier if classifier is not None else OpenRouterClassifier())
    signal_cache = TTLCache()
    db_path = db_path or os.environ.get("SMM_DB_PATH", DEFAULT_DB_PATH)
    watchlists = WatchlistStore(db_path)
    signals = SignalLog(db_path)
    portfolio = PortfolioStore(db_path)
    mcp = MCPServer("stock-market-mcp")

    def audited_tool[F: Callable](fn: F) -> F:
        mcp.tool()(_audited(fn))
        return fn

    @audited_tool
    def get_quote(ticker: str) -> dict:
        """Current quote for a stock ticker (price and currency)."""
        return fetcher.fetch_quote(ticker)

    @audited_tool
    def get_fundamentals(ticker: str) -> dict:
        """Key company fundamentals (name, sector, industry, market cap)."""
        return fetcher.fetch_fundamentals(ticker)

    @audited_tool
    def get_signal(
        ticker: str,
        interval: Interval = "5m",
        period: str | None = None,
        atr_multiple: StrictNumber = PLAN_ATR_MULTIPLE,
        reward_ratio: StrictNumber = PLAN_REWARD_RATIO,
    ) -> dict:
        """Advisory buy/sell/hold Signal from price history and indicators on bars of one Interval, with the Price (last close), the Hit Rate of past Signals of its kind (buy/sell or hold) for this ticker and Interval (null until enough are scored), the Minimum Confidence applied as min_confidence (set by the server from buy/sell Hit Rates per 0.1 Confidence band, a fixed default until enough are scored), and for buy or sell at or above min_confidence a Trade Plan (Entry, Stop, Target), else null; min_confidence null means no band beats break-even, so no buy or sell gets a Trade Plan; for a held ticker a position key with the Position, its Unrealised P&L (amount and percent of cost) at the Price, and cut_loss: true when the Verdict points against the Position (sell on a long, buy on a short) at or above min_confidence while its Unrealised P&L is negative, else false (the Trade Plan is unchanged), and cut_loss_hit_rate: Targets / (Targets + Stops) over scored past Signals that pointed against a losing Position at any Confidence, across all tickers and Intervals (null until enough are scored); position is absent when not held; input only, never acts on it."""
        for name, value in (("atr_multiple", atr_multiple), ("reward_ratio", reward_ratio)):
            if not 0 < value <= PLAN_ARG_MAX:
                raise ToolError(f"{name} must be in (0, {PLAN_ARG_MAX}], got {value}")
        df = history(fetcher, ticker, interval, period)
        # Read outside the classifier cache, so cached Signals get a fresh Price and plan.
        price = float(df["Close"].iloc[-1])
        if not math.isfinite(price):
            raise ToolError(f"latest row has a non-finite close (close={price})")
        ticker = ticker.upper()
        period = fetch_period(interval, period)
        # Read before the classifier, so a broken Portfolio file fails before a classifier call.
        position = portfolio.load().get(ticker)
        # Quantity stays out, so size doesn't sway the Verdict or bust the cache (ADR-0005).
        held = pnl = None
        if position is not None:
            pnl = _unrealised_pnl(position, price)
            pct = pnl["unrealised_pnl_pct"]
            held = {"direction": position["direction"], "avg_cost": position["avg_cost"], "unrealised_pnl_pct": pct}
        # The prompt without the previous Signal is the cache key, so storing a Signal doesn't defeat the cache.
        prompt = build_signal_prompt(ticker, period, df, interval=interval, position=held)
        signals.score(ticker, interval, df, now().isoformat())
        signal = signal_cache.get(prompt, signal_ttl)
        fresh = signal is None
        if fresh:
            previous = signals.latest(ticker, interval)
            signal = signal_classifier.classify(prompt if previous is None else with_previous_signal(prompt, previous))
        # After the cache, so a cached Signal gets the current Minimum Confidence; a new pending Signal changes neither read.
        min_confidence = signals.min_confidence(PROMPT_VERSION, BREAK_EVEN_HIT_RATE)
        hit_rate = signals.hit_rate(ticker, interval, PROMPT_VERSION, signal["verdict"])
        confident = min_confidence is not None and signal["confidence"] >= min_confidence
        direction = _VERDICT_DIRECTION.get(signal["verdict"]) if confident else None
        # Before the Signal is stored, so a plan error doesn't fail a call that already logged.
        plan = _trade_plan(df, price, direction, atr_multiple, reward_ratio) if direction else None
        # Stored at any Confidence for the Cut Loss Hit Rate; the reply's cut_loss also needs the Minimum Confidence.
        opposes_losing = None
        if position is not None and pnl is not None:
            opposes = _VERDICT_DIRECTION.get(signal["verdict"]) not in (None, position["direction"])
            opposes_losing = opposes and _losing(pnl)
        if fresh:
            long_stop, long_target = _default_plan(df, price, "long")
            short_stop, short_target = _default_plan(df, price, "short")
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
                cut_loss=opposes_losing,
            )
            # Cached only once stored, so a failed store is retried instead of served from the cache.
            signal_cache.put(prompt, signal)
        reply = {**signal, "price": price, "hit_rate": hit_rate, "min_confidence": min_confidence, "trade_plan": plan}
        if position is not None and pnl is not None:
            reply["position"] = {
                **position,
                **pnl,
                "cut_loss": confident and opposes_losing,
                "cut_loss_hit_rate": signals.cut_loss_hit_rate(PROMPT_VERSION),
            }
        return reply

    @audited_tool
    def scan(watchlist: str | None = None, tickers: list[str] | None = None) -> list[dict]:
        """Scan a Watchlist or an ad-hoc ticker list: per ticker the Price and its time, the previous day's high, low and close, daily ATR % and RSI, the Gap, Relative Volume and distance from this session's VWAP. No Verdict, no classifier call; a failing ticker gets its own error entry."""
        if (watchlist is None) == (tickers is None):
            raise ToolError("give exactly one of watchlist or tickers")
        if watchlist is not None:
            lists = watchlists.load()
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
        watchlists.set(name, tickers)
        return {"name": name, "tickers": tickers}

    @audited_tool
    def get_watchlists() -> dict:
        """Every saved Watchlist as {name: [tickers]}."""
        return watchlists.load()

    @audited_tool
    def set_position(ticker: str, direction: Direction, quantity: StrictNumber, avg_cost: StrictNumber) -> dict:
        """Save the Position for a ticker (checked with a quote), replacing any held one; quantity 0 deletes it and replies with just the ticker and quantity 0. Direction long or short, quantity at least 0, avg_cost above 0; as many Positions as a Watchlist has tickers. Records only, never places orders."""
        ticker = ticker.upper()
        if not (math.isfinite(quantity) and quantity >= 0):
            raise ToolError(f"quantity must be a finite number of at least 0, got {quantity}")
        if quantity == 0:
            portfolio.delete(ticker)
            return {"ticker": ticker, "quantity": 0}
        if not (math.isfinite(avg_cost) and avg_cost > 0):
            raise ToolError(f"avg_cost must be a finite number above 0, got {avg_cost}")
        held = portfolio.load()
        if ticker not in held and len(held) >= WATCHLIST_MAX_TICKERS:
            raise ToolError(f"at most {WATCHLIST_MAX_TICKERS} Positions; delete one before adding {ticker}")
        try:
            fetcher.fetch_quote(ticker)
        except Exception:
            # Bad input, not a dependency failure: no cause, so the audit logs a warning like set_watchlist.
            raise ToolError(f"unknown ticker, nothing saved: {ticker}") from None
        portfolio.set(ticker, direction, quantity, avg_cost)
        return {"ticker": ticker, "direction": direction, "quantity": quantity, "avg_cost": avg_cost}

    @audited_tool
    def get_portfolio() -> list[dict]:
        """Every Position with its latest price and Unrealised P&L (amount and percent of cost) in the ticker's own currency; no total across currencies. A Position whose quote fails gets its own error entry."""
        held = portfolio.load()
        results = []
        for ticker, position in held.items():
            entry = {"ticker": ticker, **position}
            try:
                quote = fetcher.fetch_quote(ticker)
                price = float(quote["price"])
                entry |= {"price": _finite(price), "currency": quote["currency"], **_unrealised_pnl(position, price)}
            except Exception as exc:
                log.exception(_error_line(f"get_portfolio ticker={ticker}", exc))
                entry["error"] = str(exc)
            results.append(entry)
        return results

    return mcp
