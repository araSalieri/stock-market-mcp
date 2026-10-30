"""In-process TTL cache wrapping a DataFetcher so tools stop re-fetching every call."""

import threading
import time
from typing import Any

import pandas as pd

from stock_market_mcp.classifier import SignalClassifier
from stock_market_mcp.data import DataFetcher

# Production TTLs: quotes and history (intraday bars) move fast, fundamentals are stable.
QUOTE_TTL_SECONDS = 60.0
HISTORY_TTL_SECONDS = 60.0
FUNDAMENTALS_TTL_SECONDS = 300.0
SIGNAL_TTL_SECONDS = 300.0


class _TTLCache:
    """Shared private per-key TTL cache (monotonic clock, lock) used by both
    caching decorators. Expiry is checked per key at read time."""

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
        self._quotes = _TTLCache()
        self._histories = _TTLCache()
        self._fundamentals = _TTLCache()

    def fetch_quote(self, ticker: str) -> dict:
        cached = self._quotes.get(ticker, self._quote_ttl)
        if cached is not None:
            return cached
        result = self._fetcher.fetch_quote(ticker)
        self._quotes.put(ticker, result)
        return result

    def fetch_fundamentals(self, ticker: str) -> dict:
        cached = self._fundamentals.get(ticker, self._fundamentals_ttl)
        if cached is not None:
            return cached
        result = self._fetcher.fetch_fundamentals(ticker)
        self._fundamentals.put(ticker, result)
        return result

    def fetch_history(
        self, ticker: str, period: str, interval: str = "1d", prepost: bool = False
    ) -> pd.DataFrame:
        key = (ticker, period, interval, prepost)
        cached = self._histories.get(key, self._history_ttl)
        if cached is not None:
            return cached
        result = self._fetcher.fetch_history(ticker, period, interval, prepost)
        self._histories.put(key, result)
        return result


class CachingClassifier:
    """SignalClassifier decorator that serves repeated classify calls from a TTL cache.

    Verdicts are cached per key (the prompt without the previous Signal, which
    pins ticker, period and data rows) for SIGNAL_TTL_SECONDS. Failures are never cached: the delegate
    raises, the cache stays empty.
    """

    def __init__(
        self,
        classifier: SignalClassifier,
        signal_ttl: float = SIGNAL_TTL_SECONDS,
    ) -> None:
        self._classifier = classifier
        self._signal_ttl = signal_ttl
        self._signals = _TTLCache()

    def classify(self, prompt: str, key: str) -> tuple[dict, bool]:
        """The Signal for prompt, cached under key, and whether it is a fresh classifier answer."""
        cached = self._signals.get(key, self._signal_ttl)
        if cached is not None:
            return cached, False
        result = self._classifier.classify(prompt)
        self._signals.put(key, result)
        return result, True
