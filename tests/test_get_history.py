"""MCP-surface tests for get_history: OHLCV plus server-side indicators."""

import json
import math

import pandas as pd
import pytest

from stock_market_mcp.data import DataFetcher

from test_get_quote import call_tool, server_url as server_url_fixture, text_of

# pytest resolves fixtures by parameter name; re-export the seam fixture.
server_url = server_url_fixture


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


class HistoryFakeFetcher(DataFetcher):
    def __init__(self):
        self.history_calls = []

    def fetch_quote(self, ticker):
        raise LookupError(f"unknown ticker: {ticker}")

    def fetch_fundamentals(self, ticker):
        raise AssertionError("get_history tests never fetch fundamentals")

    def fetch_history(self, ticker, period):
        self.history_calls.append((ticker, period))
        return fake_history()


async def test_get_history_returns_ohlcv_rows(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher), "get_history", {"ticker": "AAPL", "period": "1mo"}
    )

    assert not result.is_error
    payload = json.loads(text_of(result))
    assert fetcher.history_calls == [("AAPL", "1mo")]
    rows = payload["rows"]
    assert len(rows) == 60
    expected = fake_history()
    last = rows[-1]
    assert last["Close"] == expected["Close"].iloc[-1]
    assert last["Open"] == expected["Open"].iloc[-1]
    assert last["High"] == expected["High"].iloc[-1]
    assert last["Low"] == expected["Low"].iloc[-1]
    assert last["Volume"] == expected["Volume"].iloc[-1]
    assert last["Date"].startswith("2024-01-03T11:00:00")


async def test_get_history_unknown_indicator_is_a_loud_tool_error(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["sma", "badta"]},
    )

    assert result.is_error
    assert "badta" in text_of(result)
    assert "sma" in text_of(result)  # allowed names are listed, not just the bad one


async def test_get_history_computes_sma(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["sma"]},
    )

    assert not result.is_error
    rows = json.loads(text_of(result))["rows"]
    last = rows[-1]
    expected = fake_history()["Close"].iloc[-10:].mean()
    assert last["SMA_10"] == pytest.approx(expected)


async def test_get_history_computes_ema(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["ema"]},
    )

    assert not result.is_error
    last = json.loads(text_of(result))["rows"][-1]
    assert math.isfinite(last["EMA_10"])


async def test_get_history_computes_rsi(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["rsi"]},
    )

    assert not result.is_error
    last = json.loads(text_of(result))["rows"][-1]
    assert math.isfinite(last["RSI_14"]) and 0 <= last["RSI_14"] <= 100


async def test_get_history_computes_macd(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["macd"]},
    )

    assert not result.is_error
    last = json.loads(text_of(result))["rows"][-1]
    assert math.isfinite(last["MACD_12_26_9"])
    assert math.isfinite(last["MACDs_12_26_9"])
    assert math.isfinite(last["MACDh_12_26_9"])


async def test_get_history_computes_bbands(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["bbands"]},
    )

    assert not result.is_error
    last = json.loads(text_of(result))["rows"][-1]
    lower, mid, upper = last["BBL_5_2.0_2.0"], last["BBM_5_2.0_2.0"], last["BBU_5_2.0_2.0"]
    assert all(math.isfinite(v) for v in (lower, mid, upper))
    assert lower <= mid <= upper


async def test_get_history_computes_atr(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["atr"]},
    )

    assert not result.is_error
    last = json.loads(text_of(result))["rows"][-1]
    assert math.isfinite(last["ATRr_14"]) and last["ATRr_14"] > 0


async def test_get_history_computes_adx(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["adx"]},
    )

    assert not result.is_error
    last = json.loads(text_of(result))["rows"][-1]
    assert math.isfinite(last["ADX_14"]) and 0 <= last["ADX_14"] <= 100


async def test_get_history_computes_vwap(server_url):
    fetcher = HistoryFakeFetcher()
    result = await call_tool(
        server_url(fetcher),
        "get_history",
        {"ticker": "AAPL", "period": "1mo", "indicators": ["vwap"]},
    )

    assert not result.is_error
    last = json.loads(text_of(result))["rows"][-1]
    assert math.isfinite(last["VWAP_D"])
    # VWAP of the intraday fake sits between the frame-wide low and high
    expected = fake_history()
    assert expected["Low"].min() <= last["VWAP_D"] <= expected["High"].max()


