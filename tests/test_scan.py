"""Tests for scan on ad-hoc tickers: daily numbers through the MCP surface with fake daily frames and a fixed clock."""

from datetime import UTC, datetime

import pytest
from fakes import FakeClassifier, ScanFakeFetcher, call_tool, daily_bars, five_minute_bars, text_of
from mcp.types import CallToolResult

# 2024-03-15 06:00 WIB, before the IDX open; the UTC date is still the 14th.
BEFORE_OPEN = datetime(2024, 3, 14, 23, 0, tzinfo=UTC)
# 2024-03-15 10:00 WIB, during the IDX session.
IN_SESSION = datetime(2024, 3, 15, 3, 0, tzinfo=UTC)
# 2024-03-15 08:00 EDT, US pre-market.
US_PRE_MARKET = datetime(2024, 3, 15, 12, 0, tzinfo=UTC)
# 2024-03-15 09:40 EDT, during the US session.
US_IN_SESSION = datetime(2024, 3, 15, 13, 40, tzinfo=UTC)

IDX_DAY = ("2024-03-14 09:00", "2024-03-14 16:15")
US_DAYS = [(f"2024-03-{d:02} 09:30", f"2024-03-{d:02} 16:00") for d in range(1, 16)]


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


async def test_scan_by_watchlist_scans_its_tickers_like_an_ad_hoc_list(server_url, tmp_path):
    path = tmp_path / "watchlists.json"
    path.write_text('{"idx": ["BBCA.JK", "NOPE"]}')
    fetcher = ScanFakeFetcher(
        {"BBCA.JK": {"1d": daily_bars(rising(), last="2024-03-14"), "5m": five_minute_bars({}, [])}}
    )
    url = server_url(fetcher, watchlist_path=path, now=lambda: IN_SESSION)

    by_name = entries(await call_tool(url, "scan", {"watchlist": "idx"}))
    ad_hoc = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK", "NOPE"]}))

    assert [entry["ticker"] for entry in by_name] == ["BBCA.JK", "NOPE"]
    assert "error" not in by_name[0]
    assert by_name == ad_hoc


async def test_scan_of_an_unknown_watchlist_names_it(server_url, tmp_path):
    path = tmp_path / "watchlists.json"
    path.write_text('{"idx": ["BBCA.JK"]}')
    fetcher = ScanFakeFetcher({})
    result = await call_tool(server_url(fetcher, watchlist_path=path), "scan", {"watchlist": "swing"})

    assert result.is_error
    assert "swing" in text_of(result)
    assert fetcher.history_calls == []


async def test_scan_of_a_corrupt_watchlist_file_errors_like_get_watchlists(server_url, tmp_path):
    path = tmp_path / "watchlists.json"
    path.write_text("{not json")
    url = server_url(ScanFakeFetcher({}), watchlist_path=path)

    scanned = await call_tool(url, "scan", {"watchlist": "idx"})
    listed = await call_tool(url, "get_watchlists", {})

    assert scanned.is_error
    # Same message after the "Error executing tool <name>:" prefix.
    assert text_of(scanned).split(": ", 1)[1] == text_of(listed).split(": ", 1)[1]
    assert path.read_text() == "{not json"


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
    assert fetcher.history_calls == [(t, "3mo", "1d", False) for t in expected]


def rising(n: int = 30, start: float = 100.0) -> list[float]:
    """Close up 1 a day with High/Low 1 away: True Range is 2 every bar (ATR 2) and there are no losses (RSI 100)."""
    return [start + i for i in range(n)]


async def test_scan_before_the_open_uses_the_last_completed_session(server_url):
    # Latest bar is the 14th, closes 100..129; it's the 15th in Jakarta but the 14th in UTC.
    # The 14th's last 5m bar is newer than its daily bar, so it gives the Price. IDX has no
    # pre-market, so there is no Gap yet, and no Relative Volume or VWAP before the open.
    intraday = five_minute_bars({"2024-03-14 16:10": (129.0, 1000)}, [IDX_DAY], tz="Asia/Jakarta")
    fetcher = ScanFakeFetcher({"BBCA.JK": {"1d": daily_bars(rising(), last="2024-03-14"), "5m": intraday}})
    result = await call_tool(server_url(fetcher, now=lambda: BEFORE_OPEN), "scan", {"tickers": ["BBCA.JK"]})

    assert entries(result) == [
        {
            "ticker": "BBCA.JK",
            "price": 129.0,
            "data_date": "2024-03-14T16:10:00+07:00",
            "gap_pct": None,
            "relative_volume": None,
            "vwap_distance_pct": None,
            "prev_high": 130.0,
            "prev_low": 128.0,
            "prev_close": 129.0,
            "atr_pct": pytest.approx(200 / 129),
            "rsi": pytest.approx(100.0),
        }
    ]


async def test_scan_in_session_excludes_the_forming_bar_from_the_previous_day(server_url):
    # Today's bar (the 15th, close 129) is still forming; the previous day is the 14th, close 128.
    # The 5m bars stop on the 14th, so the daily bar is newer and gives the Price; no Gap without today's open.
    intraday = five_minute_bars({"2024-03-14 16:10": (128.0, 1000)}, [IDX_DAY], tz="Asia/Jakarta")
    fetcher = ScanFakeFetcher({"BBCA.JK": {"1d": daily_bars(rising(), last="2024-03-15"), "5m": intraday}})
    result = await call_tool(server_url(fetcher, now=lambda: IN_SESSION), "scan", {"tickers": ["BBCA.JK"]})

    assert entries(result) == [
        {
            "ticker": "BBCA.JK",
            "price": 129.0,
            "data_date": "2024-03-15T00:00:00+07:00",
            "gap_pct": None,
            "relative_volume": None,
            "vwap_distance_pct": None,
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
            "AAPL": {
                "1d": daily_bars(rising()[::-1], last="2024-03-14", tz="America/New_York"),
                "5m": five_minute_bars({"2024-03-14 15:55": (100.0, 1000)}, US_DAYS),
            },
            "TLKM.JK": {"1d": daily_bars(rising()), "5m": LookupError("5m fetch failed")},
            "BBCA.JK": {"1d": daily_bars(rising(), last="2024-03-14"), "5m": five_minute_bars({}, [])},
        }
    )
    result = await call_tool(
        server_url(fetcher, now=lambda: IN_SESSION),
        "scan",
        {"tickers": ["AAPL", "NOPE", "TLKM.JK", "BBCA.JK"]},
    )

    aapl, nope, tlkm, bbca = entries(result)
    assert aapl["ticker"] == "AAPL"
    assert aapl["rsi"] == pytest.approx(0.0)
    assert nope == {"ticker": "NOPE", "error": "no history data for ticker: NOPE"}
    assert tlkm == {"ticker": "TLKM.JK", "error": "5m fetch failed"}
    assert bbca["ticker"] == "BBCA.JK"
    assert bbca["prev_close"] == 129.0


async def test_scan_makes_no_classifier_call_and_needs_no_openrouter_key(server_url, monkeypatch):
    monkeypatch.delenv("SMM_OPENROUTER_API_KEY", raising=False)
    classifier = FakeClassifier()
    fetcher = ScanFakeFetcher({"BBCA.JK": {"1d": daily_bars(rising()), "5m": five_minute_bars({}, [])}})
    result = await call_tool(
        server_url(fetcher, classifier=classifier, now=lambda: IN_SESSION), "scan", {"tickers": ["BBCA.JK"]}
    )

    assert "error" not in entries(result)[0]
    assert classifier.prompts == []


async def test_scan_us_gap_before_the_open_uses_the_latest_pre_market_price(server_url):
    # Previous close 129; today's pre-market last trades at 132, newer than any daily bar.
    intraday = five_minute_bars(
        {"2024-03-14 15:55": (129.0, 1000), "2024-03-15 07:00": (135.0, 50), "2024-03-15 07:55": (132.0, 50)},
        US_DAYS[:-1],
    )
    daily = daily_bars(rising(), last="2024-03-14", tz="America/New_York")
    fetcher = ScanFakeFetcher({"AAPL": {"1d": daily, "5m": intraday}})
    result = await call_tool(server_url(fetcher, now=lambda: US_PRE_MARKET), "scan", {"tickers": ["AAPL"]})

    [entry] = entries(result)
    assert entry["price"] == 132.0
    assert entry["data_date"] == "2024-03-15T07:55:00-04:00"
    assert entry["prev_close"] == 129.0
    assert entry["gap_pct"] == pytest.approx(300 / 129)
    assert entry["relative_volume"] is None
    assert entry["vwap_distance_pct"] is None


async def test_scan_in_session_gap_and_vwap_use_regular_session_bars_only(server_url):
    # Previous close 128 (today's daily bar is forming). A pre-market bar at 120 with heavy volume
    # must not count: the Gap uses the 09:30 open (130), VWAP is (130*100 + 140*300) / 400 = 137.5.
    intraday = five_minute_bars(
        {"2024-03-15 08:00": (120.0, 5000), "2024-03-15 09:30": (130.0, 100), "2024-03-15 09:35": (140.0, 300)},
        US_DAYS,
    )
    daily = daily_bars(rising(), last="2024-03-15", tz="America/New_York")
    fetcher = ScanFakeFetcher({"AAPL": {"1d": daily, "5m": intraday}})
    result = await call_tool(server_url(fetcher, now=lambda: US_IN_SESSION), "scan", {"tickers": ["AAPL"]})

    [entry] = entries(result)
    assert entry["price"] == 140.0
    assert entry["data_date"] == "2024-03-15T09:35:00-04:00"
    assert entry["gap_pct"] == pytest.approx(200 / 128)
    assert entry["vwap_distance_pct"] == pytest.approx(250 / 137.5)
    assert fetcher.history_calls == [("AAPL", "3mo", "1d", False), ("AAPL", "1mo", "5m", True)]


async def test_scan_relative_volume_compares_10_prior_sessions_at_the_same_time_of_day(server_url):
    # 11 prior sessions trade 100 at 09:30 and 09:35, then 10000 at 09:40 (after today's latest bar).
    # The oldest also has 10000 at 09:30, so counting it would change the average.
    bars: dict[str, tuple[float, float]] = {}
    for day in range(4, 15):
        bars[f"2024-03-{day:02} 09:30"] = (100.0, 10000 if day == 4 else 100)
        bars[f"2024-03-{day:02} 09:35"] = (100.0, 100)
        bars[f"2024-03-{day:02} 09:40"] = (100.0, 10000)
    bars["2024-03-15 08:00"] = (100.0, 9999)  # pre-market, not counted
    bars["2024-03-15 09:30"] = (100.0, 150)
    bars["2024-03-15 09:35"] = (100.0, 150)
    daily = daily_bars(rising(), last="2024-03-15", tz="America/New_York")
    fetcher = ScanFakeFetcher({"AAPL": {"1d": daily, "5m": five_minute_bars(bars, US_DAYS)}})
    result = await call_tool(server_url(fetcher, now=lambda: US_IN_SESSION), "scan", {"tickers": ["AAPL"]})

    [entry] = entries(result)
    assert entry["relative_volume"] == pytest.approx(300 / 200)
