"""Data-fetching layer: the seam between tools and the market data provider."""

from typing import Protocol

import pandas as pd


class DataFetcher(Protocol):
    def fetch_quote(self, ticker: str) -> dict: ...

    # Default body (pyright treats a raise-only body as abstract) so quote-only
    # fakes stay structurally valid; real fetchers override this.
    def fetch_history(self, ticker: str, period: str) -> pd.DataFrame:
        return pd.DataFrame()


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

    def fetch_history(self, ticker: str, period: str) -> pd.DataFrame:
        import yfinance as yf

        df = yf.Ticker(ticker).history(period=period)
        if df.empty:
            raise LookupError(f"no history data for ticker: {ticker}")
        return df
