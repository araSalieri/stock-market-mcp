"""Data-fetching layer: the seam between tools and the market data provider."""

from typing import Protocol

import pandas as pd


class DataFetcher(Protocol):
    def fetch_quote(self, ticker: str) -> dict: ...

    def fetch_fundamentals(self, ticker: str) -> dict: ...

    # The frame's attrs["regular_periods"] holds the regular trading periods as (start, end)
    # timestamps; the Scan needs it to tell pre-market bars apart.
    def fetch_history(
        self, ticker: str, period: str, interval: str = "1d", prepost: bool = False
    ) -> pd.DataFrame: ...


class YfinanceFetcher:
    """Thin wrapper around yfinance; failures surface as exceptions, never empty data."""

    def fetch_quote(self, ticker: str) -> dict:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.fast_info
        price = info["last_price"]
        # fast_info returns NaN (not None) for dead tickers; NaN != NaN
        if price is None or price != price:
            raise LookupError(f"no quote data for ticker: {ticker}")
        return {"price": price, "currency": info["currency"], "ticker": ticker.upper()}

    def fetch_history(
        self, ticker: str, period: str, interval: str = "1d", prepost: bool = False
    ) -> pd.DataFrame:
        import yfinance as yf

        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, prepost=prepost)
        if df.empty:
            raise LookupError(f"no history data for ticker: {ticker}")
        periods = t.history_metadata.get("tradingPeriods")
        if isinstance(periods, pd.DataFrame):
            # Only start/end are reliable; yfinance's pre_*/post_* columns are a day off.
            df.attrs["regular_periods"] = list(zip(periods["start"], periods["end"], strict=True))
        return df

    def fetch_fundamentals(self, ticker: str) -> dict:
        import yfinance as yf

        t = yf.Ticker(ticker)
        info = t.info
        if not info or not info.get("longName"):
            raise LookupError(f"no fundamentals data for ticker: {ticker}")
        return {
            "ticker": ticker.upper(),
            "long_name": info["longName"],
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
        }
