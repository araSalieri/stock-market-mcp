"""MCP server assembly: tools registered against an injectable data fetcher."""

import json
from typing import cast

import pandas as pd
import pandas_ta as ta
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from stock_market_mcp.data import DataFetcher

# Curated allowlist: the only indicator names get_history will compute.
INDICATOR_ALLOWLIST = {
    "sma", "ema", "rsi", "macd", "bbands", "atr", "adx", "vwap",
}


def _indicator_frame(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Indicator values for one allowed name, keyed by pandas-ta's own column names."""
    if name == "sma":
        return ta.sma(cast("pd.Series", df["Close"])).to_frame()
    if name == "ema":
        return ta.ema(cast("pd.Series", df["Close"])).to_frame()
    if name == "rsi":
        return ta.rsi(cast("pd.Series", df["Close"])).to_frame()
    if name == "macd":
        return ta.macd(cast("pd.Series", df["Close"]))
    if name == "bbands":
        return ta.bbands(cast("pd.Series", df["Close"]))
    if name == "atr":
        return ta.atr(
            cast("pd.Series", df["High"]),
            cast("pd.Series", df["Low"]),
            cast("pd.Series", df["Close"]),
        ).to_frame()
    if name == "adx":
        return ta.adx(
            cast("pd.Series", df["High"]),
            cast("pd.Series", df["Low"]),
            cast("pd.Series", df["Close"]),
        )
    if name == "vwap":
        return ta.vwap(
            cast("pd.Series", df["High"]),
            cast("pd.Series", df["Low"]),
            cast("pd.Series", df["Close"]),
            cast("pd.Series", df["Volume"]),
        ).to_frame()
    raise ToolError(f"unsupported indicator: {name}")


def _rows(df):
    """DataFrame rows as JSON-safe dicts; dates become ISO strings, NaN becomes null."""
    payload = df.reset_index().to_json(orient="records", date_format="iso")
    if payload is None:
        raise ValueError("could not serialize history rows")
    return json.loads(payload)


def build_server(fetcher: DataFetcher) -> MCPServer:
    mcp = MCPServer("stock-market-mcp")

    @mcp.tool()
    def get_quote(ticker: str) -> dict:
        """Current quote for a stock ticker (price and currency)."""
        try:
            return fetcher.fetch_quote(ticker)
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
