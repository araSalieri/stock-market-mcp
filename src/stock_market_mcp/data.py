"""Data-fetching layer: the seam between tools and the market data provider."""

from typing import Protocol


class DataFetcher(Protocol):
    def fetch_quote(self, ticker: str) -> dict: ...


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
