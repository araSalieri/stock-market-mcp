"""MCP server assembly: tools registered against an injectable data fetcher."""

import json
from typing import cast

import pandas as pd
import pandas_ta as ta
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

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

# Curated allowlist: the only indicator names get_history will compute.
# Derived from the dispatch table so adding a branch extends the allowlist too.
INDICATOR_ALLOWLIST = frozenset(_INDICATOR_DISPATCH)


# get_plan inputs: a fixed ~3-month window gives ATR(14) headroom without a full year.
PLAN_PERIOD = "3mo"
PLAN_ATR_MULTIPLE = 2
PLAN_REWARD_RATIO = 2


def _indicator_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Indicator values for one allowed name, keyed by pandas-ta's own column names."""
    handler = _INDICATOR_DISPATCH.get(name)
    if handler is None:
        raise ToolError(f"unsupported indicator: {name}")
    return handler(df)


def _rows(df):
    """DataFrame rows as JSON-safe dicts; dates become ISO strings, NaN becomes null."""
    payload = df.reset_index().to_json(orient="records", date_format="iso")
    if payload is None:
        raise ValueError("could not serialize history rows")
    return json.loads(payload)


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
    callers use the module-level defaults (60s quotes, 300s the rest).
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
    def get_history(
        ticker: str, period: str = "1y", indicators: list[str] | None = None
    ) -> dict:
        """OHLCV history for a stock ticker over a period."""
        unknown = sorted(set(indicators or []) - INDICATOR_ALLOWLIST)
        if unknown:
            raise ToolError(
                f"unknown indicator(s): {', '.join(unknown)}; "
                f"allowed: {', '.join(sorted(INDICATOR_ALLOWLIST))}"
            )
        try:
            df = fetcher.fetch_history(ticker, period)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        for name in indicators or []:
            ind = _indicator_frame(name, df)
            if ind is None:
                raise ToolError(f"not enough history rows for indicator: {name}")
            df = df.join(ind)
        return {"ticker": ticker.upper(), "period": period, "rows": _rows(df)}

    @mcp.tool()
    def get_signal(ticker: str, period: str = "1y") -> dict:
        """Advisory buy/sell/hold verdict from history and indicators; input only, never acts on it."""
        try:
            df = fetcher.fetch_history(ticker, period)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        for name in sorted(INDICATOR_ALLOWLIST):
            ind = _indicator_frame(name, df)
            if ind is None:
                raise ToolError(f"not enough history rows for indicator: {name}")
            df = df.join(ind)
        prompt = build_signal_prompt(ticker.upper(), period, df)
        try:
            return signal_classifier.classify(prompt)
        except Exception as exc:
            raise ToolError(str(exc)) from exc

    @mcp.tool()
    def get_plan(
        ticker: str,
        direction: str,
        atr_multiple: float = PLAN_ATR_MULTIPLE,
        reward_ratio: float = PLAN_REWARD_RATIO,
    ) -> dict:
        """Advisory Trade Plan (Entry, Stop, Target) for a Direction from the last close and ATR(14); input only, never acts on it."""
        for name, value in (("atr_multiple", atr_multiple), ("reward_ratio", reward_ratio)):
            if not 0 < value <= 10:
                raise ToolError(f"{name} must be in (0, 10], got {value}")
        if direction not in ("long", "short"):
            raise ToolError(f"unsupported direction: {direction}; allowed: long, short")
        try:
            df = fetcher.fetch_history(ticker, PLAN_PERIOD)
        except Exception as exc:
            raise ToolError(str(exc)) from exc
        atr_frame = _indicator_frame("atr", df)
        atr = float("nan") if atr_frame is None else float(atr_frame.iloc[-1, 0])
        if atr != atr:  # None or NaN last value: too few rows for ATR(14)
            raise ToolError("not enough history rows for indicator: atr")
        entry = float(df["Close"].iloc[-1])
        side = 1 if direction == "long" else -1
        stop = entry - side * atr_multiple * atr
        # Same line for both Directions: short Target = Entry - ratio * (Stop - Entry).
        target = entry + reward_ratio * (entry - stop)
        if stop <= 0 or target <= 0:
            raise ToolError(
                f"plan has non-positive prices (stop={stop}, target={target}); "
                "ATR is too large relative to the last close"
            )
        return {
            "ticker": ticker.upper(),
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "atr": atr,
            "atr_multiple": atr_multiple,
            "reward_ratio": reward_ratio,
            # Same serialization get_signal's prompt uses, so both tools report one data_date.
            "data_date": _rows(df.tail(1))[0]["Date"],
        }

    return mcp
