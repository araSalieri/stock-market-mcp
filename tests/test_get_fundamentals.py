from test_get_quote import call_tool, text_of


class FundamentalsFakeFetcher:
    """Fake data layer: fundamentals only; the quote seam is never touched."""

    def __init__(self, fundamentals=None, error=None):
        self.fundamentals = fundamentals or {}
        self.error = error

    def fetch_quote(self, ticker):
        raise AssertionError("get_fundamentals must not call fetch_quote")

    def fetch_fundamentals(self, ticker):
        if self.error is not None:
            raise self.error
        info = self.fundamentals.get(ticker.upper())
        if info is None:
            raise LookupError(f"unknown ticker: {ticker}")
        return info


async def test_get_fundamentals_returns_facts_for_known_ticker(server_url):
    fetcher = FundamentalsFakeFetcher(
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
    fetcher = FundamentalsFakeFetcher()  # no fundamentals at all
    result = await call_tool(server_url(fetcher), "get_fundamentals", {"ticker": "NOPE"})

    assert result.is_error
    assert "NOPE" in text_of(result)


async def test_get_fundamentals_provider_failure_is_a_loud_tool_error(server_url):
    fetcher = FundamentalsFakeFetcher(error=RuntimeError("yahoo is down"))
    result = await call_tool(server_url(fetcher), "get_fundamentals", {"ticker": "AAPL"})

    assert result.is_error
    assert "yahoo is down" in text_of(result)
