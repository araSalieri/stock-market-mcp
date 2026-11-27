"""get_sentiment through the MCP surface: the Reading from the Sentiment reader with the Price on the Interval."""

import json
from datetime import UTC, datetime

import pytest
from fakes import FakeFetcher, FakeSentimentReader, call_tool, daily_bars, text_of
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

NOW = datetime(2026, 11, 27, 15, 0, tzinfo=UTC)
FUNDAMENTALS = {"BBCA.JK": {"ticker": "BBCA.JK", "long_name": "PT Bank Central Asia Tbk"}}


def _fetcher(closes=None):
    bars = daily_bars(closes or [100.0] * 59 + [104.0], last="2026-11-27")
    return FakeFetcher(fundamentals=FUNDAMENTALS, history={"BBCA.JK": {"1d": bars, "5m": bars}})


async def test_the_reading_comes_back_with_the_price_on_the_interval(server_url):
    reader = FakeSentimentReader()
    url = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)

    result = await call_tool(url, "get_sentiment", {"ticker": "bbca.jk"})

    assert not result.is_error, text_of(result)
    assert json.loads(text_of(result)) == {
        "ticker": "BBCA.JK",
        "interval": "1d",
        "days": 1,
        "reading": "bullish",
        "summary": "Traders expect strong loan growth.",
        "posts": ["https://x.com/trader/status/1"],
        "posts_fetched": 42,
        "price": 104.0,
        "data_date": "2026-11-27T00:00:00+07:00",
        "note": "A Reading is not a Signal and sets nothing for Signals.",
    }


@pytest.mark.parametrize(("days", "from_date"), [(1, "2026-11-27"), (3, "2026-11-25"), (7, "2026-11-21")])
async def test_the_lookback_sets_the_utc_days_searched(server_url, days, from_date):
    reader = FakeSentimentReader()
    url = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK", "days": days})

    assert not result.is_error, text_of(result)
    [(_, sent_from, sent_to)] = reader.reads
    # to_date is exclusive, so tomorrow takes in all of today.
    assert (sent_from, sent_to) == (from_date, "2026-11-28")


async def test_the_search_uses_the_bare_cashtag_and_the_company_name(server_url):
    reader = FakeSentimentReader()
    url = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)

    await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    [(prompt, _, _)] = reader.reads
    assert "$BBCA " in prompt
    assert "$BBCA.JK" not in prompt
    assert "PT Bank Central Asia Tbk" in prompt


async def test_a_ticker_without_a_name_is_searched_by_cashtag_only(server_url):
    reader = FakeSentimentReader()
    bars = daily_bars([100.0] * 60, last="2026-11-27")
    url = server_url(FakeFetcher(fundamentals={}, history=bars), sentiment_reader=reader, now=lambda: NOW)

    result = await call_tool(url, "get_sentiment", {"ticker": "GOTO.JK"})

    assert not result.is_error, text_of(result)
    [(prompt, _, _)] = reader.reads
    assert "$GOTO" in prompt


@pytest.mark.parametrize("args", [{"days": 0}, {"days": 8}, {"interval": "1wk"}, {"ticker": "NOPE"}])
async def test_bad_arguments_fail_before_any_read(server_url, args):
    reader = FakeSentimentReader()
    url = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK", **args})

    assert result.is_error
    assert reader.reads == []
    if "days" in args:
        assert "days must be 1 to 7" in text_of(result)
    if "ticker" in args:
        assert "NOPE" in text_of(result)


async def test_a_repeat_call_within_the_ttl_reuses_the_read_with_a_fresh_price(server_url):
    reader = FakeSentimentReader()
    first, second = (daily_bars([100.0] * 59 + [close], last="2026-11-27") for close in (104.0, 106.0))
    fetcher = FakeFetcher(fundamentals=FUNDAMENTALS, history=[first, second])
    url = server_url(fetcher, sentiment_reader=reader, history_ttl=0, now=lambda: NOW)

    await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})
    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert json.loads(text_of(result))["price"] == 106.0
    assert len(reader.reads) == 1


async def test_another_lookback_or_an_expired_read_reads_again(server_url):
    reader = FakeSentimentReader()
    cached = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)
    await call_tool(cached, "get_sentiment", {"ticker": "BBCA.JK"})
    await call_tool(cached, "get_sentiment", {"ticker": "BBCA.JK", "days": 2})
    await call_tool(cached, "get_sentiment", {"ticker": "BBCA.JK", "interval": "5m"})
    expired = server_url(_fetcher(), sentiment_reader=reader, sentiment_ttl=0, now=lambda: NOW)
    await call_tool(expired, "get_sentiment", {"ticker": "BBCA.JK"})
    await call_tool(expired, "get_sentiment", {"ticker": "BBCA.JK"})

    assert len(reader.reads) == 5


GOOD = FakeSentimentReader().reply


@pytest.mark.parametrize(
    ("bad", "named"),
    [
        ({**GOOD, "reading": "euphoric"}, "euphoric"),
        ({**GOOD, "confidence": 0.9}, "confidence"),
        ({key: value for key, value in GOOD.items() if key != "summary"}, "summary"),
        ({**GOOD, "summary": None}, "summary"),
        ({**GOOD, "posts_fetched": "many"}, "posts_fetched"),
    ],
    ids=["reading", "extra-key", "missing-key", "summary-type", "posts_fetched-type"],
)
async def test_a_reply_breaking_the_contract_is_retried_once_then_named(server_url, bad, named):
    reader = FakeSentimentReader(replies=[bad, bad])
    url = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert result.is_error
    assert named in text_of(result)
    assert len(reader.reads) == 2


async def test_one_bad_reply_then_a_good_one_succeeds(server_url):
    reader = FakeSentimentReader(replies=[{**GOOD, "reading": "euphoric"}, GOOD])
    url = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert json.loads(text_of(result))["reading"] == "bullish"
    assert len(reader.reads) == 2


async def test_a_reader_failure_is_loud_and_never_retried(server_url):
    reader = FakeSentimentReader(error=RuntimeError("xai timeout"))
    url = server_url(_fetcher(), sentiment_reader=reader, now=lambda: NOW)

    result = await call_tool(url, "get_sentiment", {"ticker": "BBCA.JK"})

    assert result.is_error
    assert "xai timeout" in text_of(result)
    assert len(reader.reads) == 1


async def test_the_description_uses_the_glossary_and_sets_nothing_for_signals(server_url):
    async with streamable_http_client(server_url(FakeFetcher())) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        [tool] = [t for t in (await session.list_tools()).tools if t.name == "get_sentiment"]

    assert tool.description is not None
    for term in ("Sentiment", "Reading", "Lookback", "sets nothing for Signals"):
        assert term in tool.description
    props = tool.input_schema["properties"]
    assert {k: v.get("default") for k, v in props.items()} == {"ticker": None, "interval": "1d", "days": 1}
