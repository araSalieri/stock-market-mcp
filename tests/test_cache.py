"""MCP-surface tests for the TTL cache: asserted via fetcher call counts, no network I/O."""

import time

from fakes import FakeClassifier, FakeFetcher, call_tool, fake_history, text_of

# Short TTLs so tests can cross the expiry window without waiting on real timeouts.
TEST_QUOTE_TTL = 0.2
TEST_HISTORY_TTL = 0.2
TEST_FUNDAMENTALS_TTL = 0.2


def counting_fetcher() -> FakeFetcher:
    return FakeFetcher(
        quotes={"AAPL": {"price": 190.0, "currency": "USD", "ticker": "AAPL"}},
        fundamentals={"AAPL": {"ticker": "AAPL", "long_name": "Apple Inc."}},
        history=fake_history(),
    )


async def test_second_quote_within_ttl_is_served_from_cache(server_url):
    fetcher = counting_fetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    first = await call_tool(url, "get_quote", {"ticker": "AAPL"})
    second = await call_tool(url, "get_quote", {"ticker": "AAPL"})

    assert not first.is_error
    assert not second.is_error
    assert text_of(second) == text_of(first)
    assert fetcher.quote_calls == ["AAPL"]  # second call hit the cache


async def test_quote_after_ttl_expiry_refetches(server_url):
    fetcher = counting_fetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    await call_tool(url, "get_quote", {"ticker": "AAPL"})
    time.sleep(TEST_QUOTE_TTL + 0.1)
    await call_tool(url, "get_quote", {"ticker": "AAPL"})

    assert fetcher.quote_calls == ["AAPL", "AAPL"]


async def test_second_history_within_ttl_is_served_from_cache(server_url):
    fetcher = counting_fetcher()
    # Default history TTL: two get_signal calls can outlast the short test TTL.
    url = server_url(fetcher, classifier=FakeClassifier())

    first = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})
    second = await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})

    assert not first.is_error
    assert not second.is_error
    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False)]  # second call hit the cache


async def test_history_after_ttl_expiry_refetches(server_url):
    fetcher = counting_fetcher()
    url = server_url(
        fetcher,
        classifier=FakeClassifier(),
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})
    time.sleep(TEST_HISTORY_TTL + 0.1)
    await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})

    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False), ("AAPL", "1mo", "5m", False)]


async def test_second_fundamentals_within_ttl_is_served_from_cache(server_url):
    fetcher = counting_fetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    first = await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})
    second = await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})

    assert not first.is_error
    assert not second.is_error
    assert fetcher.fundamentals_calls == ["AAPL"]  # second call hit the cache


async def test_fundamentals_after_ttl_expiry_refetches(server_url):
    fetcher = counting_fetcher()
    url = server_url(
        fetcher,
        quote_ttl=TEST_QUOTE_TTL,
        history_ttl=TEST_HISTORY_TTL,
        fundamentals_ttl=TEST_FUNDAMENTALS_TTL,
    )

    await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})
    time.sleep(TEST_FUNDAMENTALS_TTL + 0.1)
    await call_tool(url, "get_fundamentals", {"ticker": "AAPL"})

    assert fetcher.fundamentals_calls == ["AAPL", "AAPL"]


class FakeClock:
    """Stands in for the cache module's time; tests move it forward by hand."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now


async def test_history_is_cached_for_60_seconds_by_default(server_url, monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("stock_market_mcp.cache.time", clock)
    fetcher = counting_fetcher()
    url = server_url(fetcher, classifier=FakeClassifier())
    args = {"ticker": "AAPL", "period": "1mo"}

    await call_tool(url, "get_signal", args)
    clock.now += 59
    await call_tool(url, "get_signal", args)
    assert len(fetcher.history_calls) == 1
    clock.now += 2
    await call_tool(url, "get_signal", args)
    assert len(fetcher.history_calls) == 2


async def test_history_cache_keeps_intervals_apart(server_url):
    fetcher = counting_fetcher()
    url = server_url(fetcher, classifier=FakeClassifier())

    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "5m", "period": "1mo"})
    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "15m", "period": "1mo"})

    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False), ("AAPL", "1mo", "15m", False)]


async def test_history_cache_keeps_prepost_apart(server_url):
    # get_signal fetches 5m bars without pre-market and scan fetches them with it; a second scan hits the cache.
    fetcher = counting_fetcher()
    url = server_url(fetcher, classifier=FakeClassifier())

    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "5m", "period": "1mo"})
    await call_tool(url, "scan", {"tickers": ["AAPL"]})
    await call_tool(url, "scan", {"tickers": ["AAPL"]})

    assert fetcher.history_calls == [
        ("AAPL", "1mo", "5m", False),
        ("AAPL", "3mo", "1d", False),
        ("AAPL", "1mo", "5m", True),
    ]


async def test_history_cache_ignores_ticker_case(server_url):
    fetcher = counting_fetcher()
    url = server_url(fetcher, classifier=FakeClassifier())

    await call_tool(url, "get_signal", {"ticker": "aapl", "period": "1mo"})
    await call_tool(url, "get_signal", {"ticker": "AAPL", "period": "1mo"})

    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False)]  # both casings share one cache entry
