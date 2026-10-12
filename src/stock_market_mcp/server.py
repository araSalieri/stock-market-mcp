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
    CachingFetcher,
)
from stock_market_mcp.data import DataFetcher


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
    *,
    quote_ttl: float = QUOTE_TTL_SECONDS,
    history_ttl: float = HISTORY_TTL_SECONDS,
    fundamentals_ttl: float = FUNDAMENTALS_TTL_SECONDS,
) -> MCPServer:
    """Assemble the MCP server over an injectable fetcher behind the TTL cache.

    The TTL parameters exist so tests can shrink the cache windows; production
    callers use the module-level defaults (60s quotes, 300s history/fundamentals).
    """
    fetcher = CachingFetcher(
        fetcher,
        quote_ttl=quote_ttl,
        history_ttl=history_ttl,
        fundamentals_ttl=fundamentals_ttl,
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

    return mcp
