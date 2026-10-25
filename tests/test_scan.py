"""Tests for scan on ad-hoc tickers: daily numbers through the MCP surface with fake daily frames and a fixed clock."""

from datetime import UTC, datetime

import pytest
from fakes import FakeClassifier, ScanFakeFetcher, call_tool, daily_bars, text_of
from mcp.types import CallToolResult

# 2024-03-15 06:00 WIB, before the IDX open; the UTC date is still the 14th.
BEFORE_OPEN = datetime(2024, 3, 14, 23, 0, tzinfo=UTC)
# 2024-03-15 10:00 WIB, during the IDX session.
IN_SESSION = datetime(2024, 3, 15, 3, 0, tzinfo=UTC)


def entries(result: CallToolResult) -> list[dict]:
    assert not result.is_error, text_of(result)
    assert result.structured_content is not None
    return result.structured_content["result"]


@pytest.mark.parametrize(
    "args",
    [{}, {"watchlist": "idx", "tickers": ["BBCA.JK"]}, {"watchlist": None, "tickers": None}],
)
async def test_scan_needs_exactly_one_of_watchlist_or_tickers(server_url, args):
    result = await call_tool(server_url(ScanFakeFetcher({})), "scan", args)

    assert result.is_error
    assert "exactly one" in text_of(result)


async def test_scan_by_watchlist_is_not_supported_yet(server_url):
    result = await call_tool(server_url(ScanFakeFetcher({})), "scan", {"watchlist": "idx"})

    assert result.is_error
    assert "not supported yet" in text_of(result)


async def test_scan_caps_tickers_at_15(server_url):
    tickers = [f"T{i}" for i in range(16)]
    fetcher = ScanFakeFetcher({})
    result = await call_tool(server_url(fetcher), "scan", {"tickers": tickers})

    assert result.is_error
    assert "15" in text_of(result)
    assert fetcher.history_calls == []


async def test_scan_upper_cases_and_dedupes_tickers_keeping_order(server_url):
    # 16 raw tickers, 15 after de-duplication, so the cap counts unique tickers.
    tickers = ["aapl", "bbca.jk", "AAPL"] + [f"t{i}" for i in range(13)]
    fetcher = ScanFakeFetcher({})
    result = await call_tool(server_url(fetcher, now=lambda: IN_SESSION), "scan", {"tickers": tickers})

    assert not result.is_error
    expected = ["AAPL", "BBCA.JK"] + [f"T{i}" for i in range(13)]
    assert [entry["ticker"] for entry in entries(result)] == expected
    assert fetcher.history_calls == [(t, "3mo") for t in expected]


def rising(n: int = 30, start: float = 100.0) -> list[float]:
    """Close up 1 a day with High/Low 1 away: True Range is 2 every bar (ATR 2) and there are no losses (RSI 100)."""
    return [start + i for i in range(n)]


async def test_scan_before_the_open_uses_the_last_completed_session(server_url):
    # Latest bar is the 14th, closes 100..129; it's the 15th in Jakarta but the 14th in UTC.
    fetcher = ScanFakeFetcher({"BBCA.JK": daily_bars(rising(), last="2024-03-14")})
    result = await call_tool(server_url(fetcher, now=lambda: BEFORE_OPEN), "scan", {"tickers": ["BBCA.JK"]})

    assert entries(result) == [
        {
            "ticker": "BBCA.JK",
            "price": 129.0,
            "data_date": "2024-03-14T00:00:00+07:00",
            "prev_high": 130.0,
            "prev_low": 128.0,
            "prev_close": 129.0,
            "atr_pct": pytest.approx(200 / 129),
            "rsi": pytest.approx(100.0),
        }
    ]


async def test_scan_in_session_excludes_the_forming_bar_from_the_previous_day(server_url):
    # Today's bar (the 15th, close 129) is still forming; the previous day is the 14th, close 128.
    fetcher = ScanFakeFetcher({"BBCA.JK": daily_bars(rising(), last="2024-03-15")})
    result = await call_tool(server_url(fetcher, now=lambda: IN_SESSION), "scan", {"tickers": ["BBCA.JK"]})

    assert entries(result) == [
        {
            "ticker": "BBCA.JK",
            "price": 129.0,
            "data_date": "2024-03-15T00:00:00+07:00",
            "prev_high": 129.0,
            "prev_low": 127.0,
            "prev_close": 128.0,
            "atr_pct": pytest.approx(1.5625),
            "rsi": pytest.approx(100.0),
        }
    ]


async def test_scan_gives_a_failing_ticker_its_own_error_entry(server_url):
    # Falling closes: no gains, so RSI is 0.
    fetcher = ScanFakeFetcher(
        {
            "AAPL": daily_bars(rising()[::-1], last="2024-03-14", tz="America/New_York"),
            "BBCA.JK": daily_bars(rising(), last="2024-03-14"),
        }
    )
    result = await call_tool(
        server_url(fetcher, now=lambda: IN_SESSION), "scan", {"tickers": ["AAPL", "NOPE", "BBCA.JK"]}
    )

    aapl, nope, bbca = entries(result)
    assert aapl["ticker"] == "AAPL"
    assert aapl["rsi"] == pytest.approx(0.0)
    assert nope == {"ticker": "NOPE", "error": "no history data for ticker: NOPE"}
    assert bbca["ticker"] == "BBCA.JK"
    assert bbca["prev_close"] == 129.0


async def test_scan_makes_no_classifier_call_and_needs_no_openrouter_key(server_url, monkeypatch):
    monkeypatch.delenv("SMM_OPENROUTER_API_KEY", raising=False)
    classifier = FakeClassifier()
    fetcher = ScanFakeFetcher({"BBCA.JK": daily_bars(rising())})
    result = await call_tool(
        server_url(fetcher, classifier=classifier, now=lambda: IN_SESSION), "scan", {"tickers": ["BBCA.JK"]}
    )

    assert "error" not in entries(result)[0]
    assert classifier.prompts == []
