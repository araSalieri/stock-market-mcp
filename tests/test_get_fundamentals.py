from fakes import FakeFetcher, call_tool, text_of


async def test_get_fundamentals_returns_facts_for_known_ticker(server_url):
    fetcher = FakeFetcher(
        fundamentals={
            "AAPL": {
                "ticker": "AAPL",
                "long_name": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "market_cap": 3_400_000_000_000,
            }
        }
    )
    result = await call_tool(server_url(fetcher), "get_fundamentals", {"ticker": "AAPL"})

    assert not result.is_error
    text = text_of(result)
    assert "Apple Inc." in text
    assert "Technology" in text


async def test_get_fundamentals_unknown_ticker_is_a_loud_tool_error(server_url):
    fetcher = FakeFetcher(fundamentals={})  # no fundamentals at all
    result = await call_tool(server_url(fetcher), "get_fundamentals", {"ticker": "NOPE"})

    assert result.is_error
    assert "NOPE" in text_of(result)


async def test_get_fundamentals_provider_failure_is_a_loud_tool_error(server_url):
    fetcher = FakeFetcher(error=RuntimeError("yahoo is down"))
    result = await call_tool(server_url(fetcher), "get_fundamentals", {"ticker": "AAPL"})

    assert result.is_error
    assert "yahoo is down" in text_of(result)
