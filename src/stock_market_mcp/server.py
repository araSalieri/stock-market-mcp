"""MCP server assembly: tools registered against an injectable data fetcher."""

import math
from collections.abc import Callable
from datetime import UTC, datetime
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


SCAN_MAX_TICKERS = 15


def _finite(value: float) -> float | None:
    """JSON has no NaN; an indicator without a value is null."""
    return value if math.isfinite(value) else None


def _scan_daily(df: pd.DataFrame, now: datetime) -> dict:
    """Daily Scan numbers; the previous day is the last bar dated before today in the exchange timezone."""
    index = cast("pd.DatetimeIndex", df.index)
    session_start = now.astimezone(index.tz).replace(hour=0, minute=0, second=0, microsecond=0)
    before = df[index < session_start]
    if before.empty:
        raise LookupError("no daily bar before the current session")
    prev = before.iloc[-1]
    prev_close = float(prev["Close"])
    atr = float(_indicator_frame("atr", df).iloc[-1, 0])
    rsi = float(_indicator_frame("rsi", df).iloc[-1, 0])
    return {
        "price": float(df["Close"].iloc[-1]),
        "data_date": cast("pd.Timestamp", index[-1]).isoformat(),
        "prev_high": float(prev["High"]),
        "prev_low": float(prev["Low"]),
        "prev_close": prev_close,
        "atr_pct": _finite(atr / prev_close * 100),
        "rsi": _finite(rsi),
    }


def build_server(
    fetcher: DataFetcher,
    classifier: SignalClassifier | None = None,
    *,
    quote_ttl: float = QUOTE_TTL_SECONDS,
    history_ttl: float = HISTORY_TTL_SECONDS,
    fundamentals_ttl: float = FUNDAMENTALS_TTL_SECONDS,
    signal_ttl: float = SIGNAL_TTL_SECONDS,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> MCPServer:
    """Assemble the MCP server over an injectable fetcher and classifier.

    The TTL parameters exist so tests can shrink the cache windows; production
    callers use the module-level defaults (60s quotes, 300s the rest). `now` is
    the clock the Scan uses to tell which session is current.
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
        period: str = "1y",
        atr_multiple: PlanArg = PLAN_ATR_MULTIPLE,
        reward_ratio: PlanArg = PLAN_REWARD_RATIO,
    ) -> dict:
        """Advisory buy/sell/hold Signal from price history and indicators, with the Price (last close) and, for buy or sell, a Trade Plan (Entry, Stop, Target); input only, never acts on it."""
        for name, value in (("atr_multiple", atr_multiple), ("reward_ratio", reward_ratio)):
            if not 0 < value <= PLAN_ARG_MAX:
                raise ToolError(f"{name} must be in (0, {PLAN_ARG_MAX}], got {value}")
        try:
            df = fetcher.fetch_history(ticker, period)
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
        prompt = build_signal_prompt(ticker.upper(), period, df)
        try:
            signal = signal_classifier.classify(prompt)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        direction = _VERDICT_DIRECTION.get(signal["verdict"])
        plan = _trade_plan(df, price, direction, atr_multiple, reward_ratio) if direction else None
        return {**signal, "price": price, "trade_plan": plan}

    @mcp.tool()
    def scan(watchlist: str | None = None, tickers: list[str] | None = None) -> list[dict]:
        """Scan a Watchlist or an ad-hoc ticker list: per ticker the Price and its time, the previous day's high, low and close, daily ATR % and RSI. No Verdict, no classifier call; a failing ticker gets its own error entry."""
        if (watchlist is None) == (tickers is None):
            raise ToolError("give exactly one of watchlist or tickers")
        if watchlist is not None:
            raise ToolError("scan by watchlist is not supported yet; pass tickers")
        names = list(dict.fromkeys(t.upper() for t in cast("list[str]", tickers)))
        if len(names) > SCAN_MAX_TICKERS:
            raise ToolError(f"scan takes at most {SCAN_MAX_TICKERS} tickers, got {len(names)}")
        results = []
        for ticker in names:
            try:
                results.append({"ticker": ticker, **_scan_daily(fetcher.fetch_history(ticker, "3mo"), now())})
            except Exception as exc:
                results.append({"ticker": ticker, "error": str(exc)})
        return results

    return mcp
