"""Tests for scan on ad-hoc tickers: daily numbers through the MCP surface with fake daily frames and a fixed clock."""

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pandas as pd
import pytest
from fakes import FakeClassifier, FakeFetcher, call_tool, daily_bars, five_minute_bars, text_of
from mcp.types import CallToolResult

from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.watchlists import WatchlistStore

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
    result = await call_tool(server_url(FakeFetcher(history={})), "scan", args)

    assert result.is_error
    assert "exactly one" in text_of(result)


async def test_scan_by_watchlist_scans_its_tickers_like_an_ad_hoc_list(server_url, tmp_path):
    path = tmp_path / "smm.db"
    # Seeded through the store: the fake has no quotes, so set_watchlist can't save these.
    WatchlistStore(path).set("idx", ["BBCA.JK", "NOPE"])
    fetcher = FakeFetcher(history=
        {"BBCA.JK": {"1d": daily_bars(rising(), last="2024-03-14"), "5m": five_minute_bars({}, [])}}
    )
    url = server_url(fetcher, db_path=path, now=lambda: IN_SESSION)

    by_name = entries(await call_tool(url, "scan", {"watchlist": "idx"}))
    ad_hoc = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK", "NOPE"]}))

    assert [entry["ticker"] for entry in by_name] == ["BBCA.JK", "NOPE"]
    assert "error" not in by_name[0]
    assert by_name == ad_hoc


async def test_scan_of_an_unknown_watchlist_names_it(server_url, tmp_path):
    path = tmp_path / "smm.db"
    WatchlistStore(path).set("idx", ["BBCA.JK"])
    fetcher = FakeFetcher(history={})
    result = await call_tool(server_url(fetcher, db_path=path), "scan", {"watchlist": "swing"})

    assert result.is_error
    assert "swing" in text_of(result)
    assert fetcher.history_calls == []


async def test_scan_of_a_broken_database_file_errors_like_get_watchlists(server_url, tmp_path):
    path = tmp_path / "smm.db"
    path.write_text("{not a database")
    url = server_url(FakeFetcher(history={}), db_path=path)

    scanned = await call_tool(url, "scan", {"watchlist": "idx"})
    listed = await call_tool(url, "get_watchlists", {})

    assert scanned.is_error
    # Same message after the "Error executing tool <name>:" prefix.
    assert text_of(scanned).split(": ", 1)[1] == text_of(listed).split(": ", 1)[1]
    assert path.read_text() == "{not a database"


async def test_scan_caps_tickers_at_15(server_url):
    tickers = [f"T{i}" for i in range(16)]
    fetcher = FakeFetcher(history={})
    result = await call_tool(server_url(fetcher), "scan", {"tickers": tickers})

    assert result.is_error
    assert "15" in text_of(result)
    assert fetcher.history_calls == []


async def test_scan_upper_cases_and_dedupes_tickers_keeping_order(server_url):
    # 16 raw tickers, 15 after de-duplication, so the cap counts unique tickers.
    tickers = ["aapl", "bbca.jk", "AAPL"] + [f"t{i}" for i in range(13)]
    fetcher = FakeFetcher(history={})
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
    fetcher = FakeFetcher(history={"BBCA.JK": {"1d": daily_bars(rising(), last="2024-03-14"), "5m": intraday}})
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
            "re_entry_locks": [],
        }
    ]


async def test_scan_in_session_excludes_the_forming_bar_from_the_previous_day(server_url):
    # Today's bar (the 15th, close 129) is still forming; the previous day is the 14th, close 128.
    # The 5m bars stop on the 14th, so the daily bar is newer and gives the Price; no Gap without today's open.
    intraday = five_minute_bars({"2024-03-14 16:10": (128.0, 1000)}, [IDX_DAY], tz="Asia/Jakarta")
    fetcher = FakeFetcher(history={"BBCA.JK": {"1d": daily_bars(rising(), last="2024-03-15"), "5m": intraday}})
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
            "re_entry_locks": [],
        }
    ]


async def test_scan_gives_a_failing_ticker_its_own_error_entry(server_url):
    # Falling closes: no gains, so RSI is 0.
    fetcher = FakeFetcher(history=
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
    assert nope == {"ticker": "NOPE", "error": "no history data for ticker: NOPE", "re_entry_locks": []}
    assert tlkm == {"ticker": "TLKM.JK", "error": "5m fetch failed", "re_entry_locks": []}
    assert bbca["ticker"] == "BBCA.JK"
    assert bbca["prev_close"] == 129.0


async def test_scan_makes_no_classifier_call_and_needs_no_openrouter_key(server_url, monkeypatch):
    monkeypatch.delenv("SMM_OPENROUTER_API_KEY", raising=False)
    classifier = FakeClassifier()
    fetcher = FakeFetcher(history={"BBCA.JK": {"1d": daily_bars(rising()), "5m": five_minute_bars({}, [])}})
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
    fetcher = FakeFetcher(history={"AAPL": {"1d": daily, "5m": intraday}})
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
    fetcher = FakeFetcher(history={"AAPL": {"1d": daily, "5m": intraday}})
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
    fetcher = FakeFetcher(history={"AAPL": {"1d": daily, "5m": five_minute_bars(bars, US_DAYS)}})
    result = await call_tool(server_url(fetcher, now=lambda: US_IN_SESSION), "scan", {"tickers": ["AAPL"]})

    [entry] = entries(result)
    assert entry["relative_volume"] == pytest.approx(300 / 200)


async def test_scan_atr_and_rsi_ignore_the_forming_bar(server_url):
    # Today's forming bar crashes to 50; on the completed bars ATR is still 2 (2/128 = 1.5625%) and RSI 100.
    daily = daily_bars(rising()[:-1] + [50.0], last="2024-03-15")
    fetcher = FakeFetcher(history={"BBCA.JK": {"1d": daily, "5m": five_minute_bars({}, [])}})
    result = await call_tool(server_url(fetcher, now=lambda: IN_SESSION), "scan", {"tickers": ["BBCA.JK"]})

    [entry] = entries(result)
    assert entry["price"] == 50.0
    assert entry["atr_pct"] == pytest.approx(1.5625)
    assert entry["rsi"] == pytest.approx(100.0)


async def test_scan_gap_and_vwap_distance_are_null_when_not_finite(server_url):
    # The 09:30 bar has no open or high, so neither the Gap nor the VWAP has a value.
    intraday = five_minute_bars({"2024-03-15 09:30": (130.0, 100), "2024-03-15 09:35": (140.0, 300)}, US_DAYS)
    intraday.loc[intraday.index[0], ["Open", "High"]] = float("nan")
    daily = daily_bars(rising(), last="2024-03-15", tz="America/New_York")
    fetcher = FakeFetcher(history={"AAPL": {"1d": daily, "5m": intraday}})
    result = await call_tool(server_url(fetcher, now=lambda: US_IN_SESSION), "scan", {"tickers": ["AAPL"]})

    [entry] = entries(result)
    assert entry["gap_pct"] is None
    assert entry["vwap_distance_pct"] is None
    # The text content must be valid JSON too, which has no NaN.
    assert "NaN" not in text_of(result)


def lock_bars(closes: list[float]) -> pd.DataFrame:
    """Bars from 2024-01-01, a day apart, with High/Low one unit from Close."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", name="Date")
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000})


# A cut on these is at Price 100 on 2024-02-29 with ATR 2, so the Buyback Price is 100 - 2 * 2 * 2 = 92.
FLAT = [100.0] * 60
CUT = {"verdict": "cut", "confidence": 0.7, "rationale": "test"}
HOLD = {"verdict": "hold", "confidence": 0.7, "rationale": "test"}


def lock_server(server_url, tmp_path, history: dict, replies: list[dict]):
    """A server over BBCA.JK's Scan bars plus the given {Interval: bars}, which tests may swap between calls; the
    classifier gives replies one per get_signal."""
    frames = {"1d": daily_bars(rising(), last="2024-03-14"), "5m": five_minute_bars({}, []), **history}
    fetcher = FakeFetcher(history={"BBCA.JK": frames})
    url = server_url(
        fetcher,
        classifier=FakeClassifier(replies=replies),
        db_path=tmp_path / "smm.db",
        history_ttl=0,
        signal_ttl=0,
        now=lambda: IN_SESSION,
    )
    return url, fetcher, frames


async def lock_of(url, interval) -> dict:
    result = await call_tool(url, "get_signal", {"ticker": "BBCA.JK", "interval": interval})
    assert not result.is_error, text_of(result)
    return json.loads(text_of(result))["re_entry_lock"]


async def test_scan_lists_each_intervals_re_entry_lock_as_get_signal_shows_it(server_url, tmp_path):
    url, _, _ = lock_server(
        server_url,
        tmp_path,
        {"1h": lock_bars(FLAT), "15m": lock_bars([120.0] * 59 + [100.0])},
        [CUT, CUT, HOLD, HOLD],
    )
    PortfolioStore(tmp_path / "smm.db").set("BBCA.JK", 10, 110.0)
    await lock_of(url, "1h")
    await lock_of(url, "15m")
    # Cleared after the cuts, so the locks give a Buyback Quantity: 10 * 100 / 92 = 10.87, rounded down.
    PortfolioStore(tmp_path / "smm.db").delete("BBCA.JK")

    bbca, other = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK", "TLKM.JK"]}))

    assert bbca["re_entry_locks"] == [
        {"interval": "15m", **await lock_of(url, "15m")},
        {"interval": "1h", **await lock_of(url, "1h")},
    ]
    assert bbca["re_entry_locks"][1] == {
        "interval": "1h",
        "state": "locked",
        "cut_price": 100.0,
        "cut_date": "2024-02-29T00:00:00",
        "buyback_price": 92.0,
        "ema": 100.0,
        "re_entry_price": 100.0,
        "buyback_quantity": 10,
    }
    assert other == {"ticker": "TLKM.JK", "error": "no history data for ticker: TLKM.JK", "re_entry_locks": []}


async def test_scan_records_a_lift_it_finds_and_lists_lifted_locks_without_a_fetch(server_url, tmp_path):
    url, fetcher, frames = lock_server(server_url, tmp_path, {"1h": lock_bars(FLAT)}, [CUT, HOLD])
    PortfolioStore(tmp_path / "smm.db").set("BBCA.JK", 10, 110.0)
    await lock_of(url, "1h")
    # The 2024-03-01 bar's low (90) touches the Buyback Price (92).
    frames["1h"] = lock_bars(FLAT + [91.0])

    [first] = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK"]}))
    frames["1h"] = LookupError("1h fetch failed")
    fetcher.history_calls.clear()
    [second] = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK"]}))
    # Back to bars without the lifting one, so only the recorded lift can show it lifted.
    frames["1h"] = lock_bars(FLAT)

    lifted = {
        "state": "lifted",
        "cut_price": 100.0,
        "cut_date": "2024-02-29T00:00:00",
        "buyback_price": 92.0,
        "lifted_date": "2024-03-01T00:00:00",
        "lifted_by": "buyback",
    }
    assert first["re_entry_locks"] == second["re_entry_locks"] == [{"interval": "1h", **lifted}]
    assert all(interval != "1h" for _, _, interval, _ in fetcher.history_calls)
    assert await lock_of(url, "1h") == lifted


async def test_scan_shows_a_lock_whose_fetch_fails_as_stale_and_keeps_the_rest(server_url, tmp_path):
    url, _, frames = lock_server(server_url, tmp_path, {"1h": lock_bars(FLAT), "15m": lock_bars(FLAT)}, [CUT, CUT])
    PortfolioStore(tmp_path / "smm.db").set("BBCA.JK", 10, 110.0)
    await lock_of(url, "1h")
    fifteen = await lock_of(url, "15m")
    PortfolioStore(tmp_path / "smm.db").delete("BBCA.JK")
    frames["1h"] = LookupError("1h fetch failed")

    [entry] = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK"]}))

    assert entry["prev_close"] == 129.0
    assert entry["re_entry_locks"] == [
        {"interval": "15m", **fifteen, "buyback_quantity": 10},
        {
            "interval": "1h",
            "state": "locked",
            "cut_price": 100.0,
            "cut_date": "2024-02-29T00:00:00",
            "buyback_price": 92.0,
            "buyback_quantity": 10,
            "stale": True,
            "error": "1h fetch failed",
        },
    ]


async def test_scan_keeps_a_tickers_locks_when_its_numbers_fail(server_url, tmp_path):
    url, _, frames = lock_server(server_url, tmp_path, {"1h": lock_bars(FLAT)}, [CUT])
    PortfolioStore(tmp_path / "smm.db").set("BBCA.JK", 10, 110.0)
    lock = await lock_of(url, "1h")
    frames["1d"] = LookupError("1d fetch failed")

    [entry] = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK"]}))

    assert entry == {"ticker": "BBCA.JK", "error": "1d fetch failed", "re_entry_locks": [{"interval": "1h", **lock}]}


async def test_scan_keeps_a_tickers_numbers_when_its_locks_fail_on_a_short_position(server_url, tmp_path):
    url, _, _ = lock_server(server_url, tmp_path, {}, [])
    with closing(sqlite3.connect(tmp_path / "smm.db")) as conn, conn:
        PortfolioStore(tmp_path / "smm.db").set("AAPL", 10, 180.0)
        conn.execute("INSERT INTO positions VALUES ('MSFT', 'short', 1, 100.0)")

    [entry] = entries(await call_tool(url, "scan", {"tickers": ["BBCA.JK"]}))

    assert entry["prev_close"] == 129.0
    assert "re_entry_locks" not in entry
    assert entry["re_entry_locks_error"] == "short Positions MSFT; clear each with set_position quantity 0"
