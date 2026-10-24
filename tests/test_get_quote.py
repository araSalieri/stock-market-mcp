import pytest
from fakes import FakeFetcher, call_tool, text_of


async def test_get_quote_unknown_ticker_is_a_loud_tool_error(server_url):
    fetcher = FakeFetcher()  # no quotes at all
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "NOPE"})

    assert result.is_error
    assert "NOPE" in text_of(result)


async def test_get_quote_provider_failure_is_a_loud_tool_error(server_url):
    fetcher = FakeFetcher(error=RuntimeError("yahoo is down"))
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    assert result.is_error
    assert "yahoo is down" in text_of(result)


def test_datafetcher_default_fetch_history_fails_loudly_not_empty():
    """The Protocol default must raise, never return empty data (spec: never empty)."""
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})

    with pytest.raises(NotImplementedError, match="fetch_history"):
        fetcher.fetch_history("AAPL", "1mo")


async def test_quote_only_fake_leaking_into_history_is_a_loud_tool_error(server_url):
    # A quote-only fake leaking into get_signal's history fetch must surface the
    # loud failure, not an empty frame (spec: never empty data).
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})
    result = await call_tool(
        server_url(fetcher), "get_signal", {"ticker": "AAPL", "period": "1mo"}
    )

    assert result.is_error
    assert "fetch_history" in text_of(result)


async def test_get_quote_returns_price_for_known_ticker(server_url):
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    assert not result.is_error
    assert "190.0" in text_of(result)
