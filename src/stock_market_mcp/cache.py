"""In-process TTL cache wrapping a DataFetcher so tools stop re-fetching every call."""

import threading
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from stock_market_mcp.data import DataFetcher

# Production TTLs: quotes and history (intraday bars) move fast, fundamentals are stable.
QUOTE_TTL_SECONDS = 60.0
HISTORY_TTL_SECONDS = 60.0
FUNDAMENTALS_TTL_SECONDS = 300.0
SIGNAL_TTL_SECONDS = 300.0


class TTLCache:
    """Per-key TTL cache (monotonic clock, lock). Expiry is checked per key at read time."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any, ttl: float) -> Any:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and time.monotonic() - entry[0] < ttl:
                return entry[1]
        return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), value)


class CachingFetcher:
    """DataFetcher decorator that serves repeated calls from a TTL cache.

    Quotes are cached per ticker for QUOTE_TTL_SECONDS; history per
    (ticker, period, interval, prepost) and fundamentals per ticker for their TTLs.
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
        self._quotes = TTLCache()
        self._histories = TTLCache()
        self._fundamentals = TTLCache()

    def fetch_quote(self, ticker: str) -> dict:
        return _cached(self._quotes, ticker, self._quote_ttl, lambda: self._fetcher.fetch_quote(ticker))

    def fetch_fundamentals(self, ticker: str) -> dict:
        return _cached(
            self._fundamentals, ticker, self._fundamentals_ttl, lambda: self._fetcher.fetch_fundamentals(ticker)
        )

    def fetch_history(
        self, ticker: str, period: str, interval: str = "1d", prepost: bool = False
    ) -> pd.DataFrame:
        return _cached(
            self._histories,
            (ticker, period, interval, prepost),
            self._history_ttl,
            lambda: self._fetcher.fetch_history(ticker, period, interval, prepost),
        )


def _cached[T](cache: TTLCache, key: Any, ttl: float, fetch: Callable[[], T]) -> T:
    """Cached value within ttl, else fetch and store; a raising fetch stores nothing."""
    cached = cache.get(key, ttl)
    if cached is not None:
        return cached
    result = fetch()
    cache.put(key, result)
    return result
