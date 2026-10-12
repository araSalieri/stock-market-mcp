"""In-process TTL cache wrapping a DataFetcher so tools stop re-fetching every call."""

import threading
import time
from typing import Any

import pandas as pd

from stock_market_mcp.data import DataFetcher

# Production TTLs: quotes move fast, history and fundamentals are stable.
QUOTE_TTL_SECONDS = 60.0
HISTORY_TTL_SECONDS = 300.0
FUNDAMENTALS_TTL_SECONDS = 300.0


class CachingFetcher:
    """DataFetcher decorator that serves repeated calls from a TTL cache.

    Quotes are cached per ticker for QUOTE_TTL_SECONDS; history per
    (ticker, period) and fundamentals per ticker for the 5-minute TTLs.
    Failures are never cached: the delegate raises, the cache stays empty.
    """

    def __init__(
        self,
        fetcher: DataFetcher,
        quote_ttl: float = QUOTE_TTL_SECONDS,
        history_ttl: float = HISTORY_TTL_SECONDS,
        fundamentals_ttl: float = FUNDAMENTALS_TTL_SECONDS,
    ) -> None:
        self._fetcher = fetcher
        self._quote_ttl = quote_ttl
        self._history_ttl = history_ttl
        self._fundamentals_ttl = fundamentals_ttl
        self._lock = threading.Lock()
        self._quotes: dict[str, tuple[float, dict]] = {}
        self._histories: dict[tuple[str, str], tuple[float, pd.DataFrame]] = {}
        self._fundamentals: dict[str, tuple[float, dict]] = {}

    def _get(self, cache: dict[Any, tuple[float, Any]], key: Any, ttl: float) -> Any:
        with self._lock:
            entry = cache.get(key)
            if entry is not None and time.monotonic() - entry[0] < ttl:
                return entry[1]
        return None

    def _put(
        self, cache: dict[Any, tuple[float, Any]], key: Any, value: Any
    ) -> None:
        with self._lock:
            cache[key] = (time.monotonic(), value)

    def fetch_quote(self, ticker: str) -> dict:
        cached = self._get(self._quotes, ticker, self._quote_ttl)
        if cached is not None:
            return cached
        result = self._fetcher.fetch_quote(ticker)
        self._put(self._quotes, ticker, result)
        return result

    def fetch_fundamentals(self, ticker: str) -> dict:
        cached = self._get(self._fundamentals, ticker, self._fundamentals_ttl)
        if cached is not None:
            return cached
        result = self._fetcher.fetch_fundamentals(ticker)
        self._put(self._fundamentals, ticker, result)
        return result

    def fetch_history(self, ticker: str, period: str) -> pd.DataFrame:
        key = (ticker, period)
        cached = self._get(self._histories, key, self._history_ttl)
        if cached is not None:
            return cached
        result = self._fetcher.fetch_history(ticker, period)
        self._put(self._histories, key, result)
        return result
