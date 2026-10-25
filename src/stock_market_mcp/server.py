"""MCP server assembly: tools registered against an injectable data fetcher."""

import math
from typing import Annotated, Literal, cast

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
    EnforcingClassifier,
    OpenRouterClassifier,
    SignalClassifier,
)
from stock_market_mcp.data import DataFetcher
from stock_market_mcp.prompt import build_signal_prompt


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

# Strict: JSON numbers only; true and "2" are rejected, not coerced.
PlanArg = Annotated[float, Strict()]

Direction = Literal["long", "short"]

Interval = Literal["1m", "5m", "15m", "1h", "1d"]

# Period when the caller gives none, per Interval.
_DEFAULT_PERIOD: dict[str, str] = {"1m": "5d", "5m": "1mo", "15m": "1mo", "1h": "6mo", "1d": "1y"}

# Intraday Intervals: Yahoo's history limit and the periods that fit inside it.
# Daily bars have no limit worth checking, so 1d is absent.
_PERIOD_LIMITS: dict[str, tuple[str, frozenset[str]]] = {
    "1m": ("7 days", frozenset({"1d", "5d"})),
    "5m": ("60 days", frozenset({"1d", "5d", "1mo"})),
    "15m": ("60 days", frozenset({"1d", "5d", "1mo"})),
    "1h": ("730 days", frozenset({"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "ytd"})),
}

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


def _indicator_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Indicator values for one allowed name, keyed by pandas-ta's own column names."""
    handler = _INDICATOR_DISPATCH.get(name)
    if handler is None:
        raise ToolError(f"unsupported indicator: {name}")
    return handler(df)


def build_server(
    fetcher: DataFetcher,
    classifier: SignalClassifier | None = None,
    *,
    quote_ttl: float = QUOTE_TTL_SECONDS,
    history_ttl: float = HISTORY_TTL_SECONDS,
    fundamentals_ttl: float = FUNDAMENTALS_TTL_SECONDS,
    signal_ttl: float = SIGNAL_TTL_SECONDS,
) -> MCPServer:
    """Assemble the MCP server over an injectable fetcher and classifier.

    The TTL parameters exist so tests can shrink the cache windows; production
    callers use the module-level defaults (60s quotes and history, 300s the rest).
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
    mcp = MCPServer("stock-market-mcp")

    @mcp.tool()
    def get_quote(ticker: str) -> dict:
        """Current quote for a stock ticker (price and currency)."""
        try:
            return fetcher.fetch_quote(ticker)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def get_fundamentals(ticker: str) -> dict:
        """Key company fundamentals (name, sector, industry, market cap)."""
        try:
            return fetcher.fetch_fundamentals(ticker)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def get_signal(
        ticker: str,
        interval: Interval = "1d",
        period: str | None = None,
        atr_multiple: PlanArg = PLAN_ATR_MULTIPLE,
        reward_ratio: PlanArg = PLAN_REWARD_RATIO,
    ) -> dict:
        """Advisory buy/sell/hold Signal from price history and indicators on bars of one Interval (1m, 5m, 15m, 1h, 1d), with the Price (last close) and, for buy or sell, a Trade Plan (Entry, Stop, Target); input only, never acts on it."""
        for name, value in (("atr_multiple", atr_multiple), ("reward_ratio", reward_ratio)):
            if not 0 < value <= PLAN_ARG_MAX:
                raise ToolError(f"{name} must be in (0, {PLAN_ARG_MAX}], got {value}")
        if period is None:
            period = _DEFAULT_PERIOD[interval]
        elif interval in _PERIOD_LIMITS:
            limit, allowed = _PERIOD_LIMITS[interval]
            if period not in allowed:
                raise ToolError(
                    f"period {period!r} is longer than Yahoo keeps for {interval} bars "
                    f"({limit}); use one of {', '.join(sorted(allowed))}"
                )
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
        prompt = build_signal_prompt(ticker.upper(), period, df, interval=interval)
        try:
            signal = signal_classifier.classify(prompt)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        direction = _VERDICT_DIRECTION.get(signal["verdict"])
        plan = _trade_plan(df, price, direction, atr_multiple, reward_ratio) if direction else None
        return {**signal, "price": price, "trade_plan": plan}

    return mcp
