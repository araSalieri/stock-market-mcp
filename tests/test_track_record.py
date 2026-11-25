"""Tests for get_track_record over MCP: Outcome counts and Hit Rates by Confidence band and Verdict."""

import json
import sqlite3
from contextlib import closing

import pytest

from stock_market_mcp.prompt import PROMPT_VERSION

from fakes import FakeClassifier, FakeFetcher, call_tool, flat_history, seed_signals, text_of

from stock_market_mcp.portfolio import PortfolioStore


async def _track_record(server_url, tmp_path, seeds, args=None) -> dict:
    """get_track_record reply after seeding seed_signals kwargs; the fakes fail loudly on any Yahoo call."""
    db = tmp_path / "s.db"
    for seed in seeds:
        seed_signals(db, **seed)
    classifier = FakeClassifier()
    result = await call_tool(server_url(FakeFetcher(), classifier=classifier, db_path=db), "get_track_record", args or {})
    assert not result.is_error, text_of(result)
    assert classifier.prompts == []
    return json.loads(text_of(result))


def _seed(outcomes: dict, **kwargs) -> dict:
    return {"outcomes": [outcome for outcome, n in outcomes.items() for _ in range(n)], **kwargs}


async def test_counts_signals_by_band_and_verdict_leaving_out_empty_verdicts(server_url, tmp_path):
    seeds = [
        _seed({"target": 2, "stop": 1, "expired": 1, None: 3}, confidence=0.8),
        _seed({"stop": 2}, verdict="cut", confidence=0.6, ticker="MSFT", interval="1h"),
        _seed({"quiet": 3, "missed": 1, None: 1}, verdict="hold", confidence=0.69),
        _seed({"target": 1, None: 1}, verdict="dip", confidence=0.85),
    ]
    reply = await _track_record(server_url, tmp_path, seeds)

    empty_plan_counts = {"target": 0, "stop": 0, "expired": 0, "pending": 0, "hit_rate": None}
    assert reply == {
        "prompt_version": PROMPT_VERSION,
        "ticker": None,
        "interval": None,
        "bands": [
            {
                "band": 0.6,
                "cut": {**empty_plan_counts, "stop": 2, "break_even": pytest.approx(1 / 3)},
                "hold": {"quiet": 3, "missed": 1, "pending": 1, "hit_rate": None},
                "buy_hit_rate": None,
            },
            {
                "band": 0.8,
                "buy": {"target": 2, "stop": 1, "expired": 1, "pending": 3, "hit_rate": None},
                "dip": {**empty_plan_counts, "target": 1, "pending": 1, "break_even": pytest.approx(1 / 3)},
                "buy_hit_rate": None,
            },
        ],
        "min_confidence": 0.6,
        "cut_loss_hit_rate": None,
        "cut_min_confidence": None,
        "dip_min_confidence": 0.6,
    }


@pytest.mark.parametrize(
    ("verdict", "hit", "miss"), [("buy", "target", "stop"), ("cut", "target", "stop"), ("hold", "quiet", "missed")]
)
async def test_hit_rate_is_null_at_9_scored_and_set_at_10_with_expired_and_pending_left_out(
    server_url, tmp_path, verdict, hit, miss
):
    seeds = [
        _seed({hit: 3, miss: 6, "expired": 5, None: 5}, verdict=verdict, confidence=0.5),
        _seed({hit: 4, miss: 6, "expired": 5, None: 5}, verdict=verdict, confidence=0.7),
    ]
    low, high = (await _track_record(server_url, tmp_path, seeds))["bands"]

    assert low[verdict]["hit_rate"] is None
    assert high[verdict]["hit_rate"] == pytest.approx(0.4)


async def test_other_prompt_versions_and_signals_that_can_never_be_scored_are_left_out(server_url, tmp_path):
    seeds = [
        _seed({"target": 1}, confidence=0.8),
        _seed({"target": 5}, confidence=0.8, prompt_version=PROMPT_VERSION - 1),
        _seed({"target": 5}, confidence=0.8, long_plan=False),
        _seed({"target": 5}, verdict="cut", confidence=0.8, short_plan=False),
        _seed({"quiet": 5}, verdict="hold", confidence=0.8, short_plan=False),
        _seed({"quiet": 5}, verdict="hold", confidence=0.8, long_plan=False),
        _seed({"target": 5}, confidence=0.3, prompt_version=PROMPT_VERSION - 1),
        _seed({"target": 5}, verdict="dip", confidence=0.8, long=(None, 108.0)),
        _seed({"target": 5}, verdict="dip", confidence=0.8, long=(96.0, None)),
        # Still scored: a buy only needs the long plan, a cut the mirrored one.
        _seed({"target": 1}, verdict="cut", confidence=0.8, long_plan=False),
        _seed({"stop": 1}, confidence=0.8, short_plan=False),
        _seed({"stop": 1}, verdict="dip", confidence=0.8, short_plan=False),
    ]
    (band,) = (await _track_record(server_url, tmp_path, seeds))["bands"]

    assert band["band"] == 0.8
    assert (band["buy"]["target"], band["buy"]["stop"]) == (1, 1)
    assert band["cut"]["target"] == 1
    assert (band["dip"]["target"], band["dip"]["stop"]) == (0, 1)
    assert "hold" not in band


@pytest.mark.parametrize(
    ("args", "echo", "targets"),
    [
        ({"ticker": "aapl"}, {"ticker": "AAPL", "interval": None}, 1 + 2),
        ({"interval": "1h"}, {"ticker": None, "interval": "1h"}, 2 + 4),
        ({"ticker": "aapl", "interval": "1h"}, {"ticker": "AAPL", "interval": "1h"}, 2),
        ({"ticker": "nope"}, {"ticker": "NOPE", "interval": None}, None),
    ],
)
async def test_ticker_and_interval_narrow_the_bands_and_are_echoed(server_url, tmp_path, args, echo, targets):
    seeds = [
        _seed({"target": 1}, ticker="AAPL", interval="1d"),
        _seed({"target": 2}, ticker="AAPL", interval="1h"),
        _seed({"target": 4}, ticker="MSFT", interval="1h"),
        _seed({"target": 8}, ticker="MSFT", interval="1d"),
    ]
    reply = await _track_record(server_url, tmp_path, seeds, args)

    assert {"ticker": reply["ticker"], "interval": reply["interval"]} == echo
    assert [band["buy"]["target"] for band in reply["bands"]] == ([targets] if targets else [])


async def test_a_bad_interval_is_rejected(server_url):
    result = await call_tool(server_url(FakeFetcher()), "get_track_record", {"interval": "2d"})

    assert result.is_error


async def test_buy_hit_rate_leaves_out_cuts_and_dips_and_is_null_at_29_scored_and_set_at_30(server_url, tmp_path):
    seeds = [
        _seed({"target": 10, "stop": 19, "expired": 3, None: 3}, confidence=0.7),
        _seed({"target": 5, "stop": 9}, verdict="cut", confidence=0.7),
        _seed({"target": 1}, verdict="dip", confidence=0.7),
        _seed({"stop": 30}, verdict="dip", confidence=0.8),
        _seed({"target": 15, "stop": 15}, confidence=0.8),
        _seed({"target": 30}, verdict="cut", confidence=0.8),
        _seed({"quiet": 30}, verdict="hold", confidence=0.8),
    ]
    low, high = (await _track_record(server_url, tmp_path, seeds))["bands"]

    assert low["buy_hit_rate"] is None
    assert high["buy_hit_rate"] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("args", "buy_hit_rates"),
    [
        ({}, [pytest.approx(0.2), pytest.approx(0.7)]),
        ({"ticker": "msft"}, [pytest.approx(0.2)]),
        ({"interval": "1d"}, [pytest.approx(0.7)]),
        ({"ticker": "nope"}, []),
    ],
)
async def test_min_confidence_and_cut_loss_hit_rate_are_server_wide_while_buy_hit_rate_follows_filters(
    server_url, tmp_path, args, buy_hit_rates
):
    seeds = [
        # Band 0.6 is under break-even, so the Minimum Confidence is band 0.7.
        _seed({"target": 6, "stop": 24}, confidence=0.6, ticker="MSFT", interval="1h"),
        _seed({"target": 15, "stop": 15}, verdict="cut", confidence=0.7, cut_loss=1),
        _seed({"target": 21, "stop": 9}, confidence=0.7, cut_loss=1),
    ]
    reply = await _track_record(server_url, tmp_path, seeds, args)

    assert reply["min_confidence"] == 0.7
    assert reply["cut_loss_hit_rate"] == pytest.approx(0.6)
    assert [band["buy_hit_rate"] for band in reply["bands"]] == buy_hit_rates


async def test_min_confidence_and_cut_loss_hit_rate_match_get_signal(server_url, tmp_path):
    db = tmp_path / "s.db"
    seed_signals(db, ["target"] * 6 + ["stop"] * 24, confidence=0.6)
    seed_signals(db, ["target"] * 18 + ["stop"] * 12, confidence=0.7, cut_loss=1)
    PortfolioStore(db).set("TSLA", 10, 90.0)
    url = server_url(FakeFetcher(history=flat_history()), classifier=FakeClassifier(), db_path=db)

    signal = json.loads(text_of(await call_tool(url, "get_signal", {"ticker": "TSLA", "interval": "1d"})))
    track_record = json.loads(text_of(await call_tool(url, "get_track_record", {})))

    assert track_record["min_confidence"] == signal["min_confidence"] == 0.7
    assert track_record["cut_loss_hit_rate"] == signal["position"]["cut_loss_hit_rate"] == pytest.approx(0.6)


async def test_opening_a_database_with_re_entry_lock_data_drops_it_and_keeps_old_signals(server_url, tmp_path):
    # A file from before ADR-0009; built by hand since no module writes the lock schema any more.
    db = tmp_path / "s.db"
    seed_signals(db, ["target", "stop"])
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute("ALTER TABLE signals ADD COLUMN re_entry TEXT")
        conn.execute("UPDATE signals SET re_entry = 'buyback'")
        conn.execute("CREATE TABLE reentry_locks (ticker TEXT, interval TEXT, cut_price REAL, cut_date TEXT, lifted INTEGER)")
        conn.execute("INSERT INTO reentry_locks VALUES ('AAPL', '1d', 100.0, '2023-01-01T00:00:00', 0)")

    reply = await _track_record(server_url, tmp_path, [])

    assert "re_entry_hit_rate" not in reply
    assert reply["bands"][0]["buy"]["target"] == reply["bands"][0]["buy"]["stop"] == 1
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'reentry_locks'").fetchone() is None
        assert "re_entry" not in {column[1] for column in conn.execute("PRAGMA table_info(signals)")}


async def test_cut_break_even_per_band_follows_filters_and_is_null_without_scored_cuts(server_url, tmp_path):
    seeds = [
        # Default cut levels: Cut Stop 104 and Buyback Price 92 at a Price of 100, break-even 4 / 12 = 1/3.
        _seed({"target": 1, "stop": 1}, verdict="cut", confidence=0.6),
        # Cut Stop 108 and Buyback Price 98: break-even 0.8; expired and pending cuts don't move it.
        _seed({"stop": 2}, verdict="cut", confidence=0.6, short=(108.0, 98.0)),
        _seed({"expired": 1, None: 1}, verdict="cut", confidence=0.6, short=(101.0, 50.0)),
        _seed({"target": 1}, verdict="cut", confidence=0.6, ticker="MSFT", short=(108.0, 98.0)),
        _seed({"expired": 1, None: 1}, verdict="cut", confidence=0.8),
        # A scored dip in the band does not move the Cut Break-even.
        _seed({"stop": 1}, verdict="dip", confidence=0.6, short=(101.0, 50.0)),
    ]
    all_tickers = await _track_record(server_url, tmp_path, seeds)
    aapl = await _track_record(server_url, tmp_path, [], {"ticker": "aapl"})

    # (1/3 * 2 + 0.8 * 3) / 5 over every ticker; (1/3 * 2 + 0.8 * 2) / 4 over AAPL.
    assert [band["cut"]["break_even"] for band in all_tickers["bands"]] == [pytest.approx(46 / 75), None]
    assert [band["cut"]["break_even"] for band in aapl["bands"]] == [pytest.approx(17 / 30), None]


@pytest.mark.parametrize(("args", "expected"), [({"interval": "1d"}, 0.7), ({"interval": "1h"}, 0.6), ({}, None)])
async def test_cut_min_confidence_matches_get_signal_for_the_interval_and_is_null_without_one(
    signal_maker, server_url, tmp_path, args, expected
):
    db = tmp_path / "s.db"
    seed_signals(db, ["target"] * 20 + ["stop"] * 10, verdict="cut", confidence=0.7)
    PortfolioStore(db).set("TSLA", 10, 90.0)
    reply = await _track_record(server_url, tmp_path, [], args)

    assert reply["cut_min_confidence"] == expected
    if expected is not None:
        signal = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(), db_path=db).make("tsla", args["interval"])
        assert signal["position"]["cut_min_confidence"] == expected


async def test_dip_break_even_per_band_follows_filters_and_is_null_without_scored_dips(server_url, tmp_path):
    seeds = [
        # Default dip levels: Stop 96 and Target 108 at a Price of 100, break-even 4 / 12 = 1/3.
        _seed({"target": 1, "stop": 1}, verdict="dip", confidence=0.6),
        # Stop 90 and Target 105: break-even 10 / 15 = 2/3; expired and pending dips don't move it.
        _seed({"stop": 2}, verdict="dip", confidence=0.6, long=(90.0, 105.0)),
        _seed({"expired": 1, None: 1}, verdict="dip", confidence=0.6, long=(99.0, 150.0)),
        _seed({"target": 1}, verdict="dip", confidence=0.6, interval="1h", long=(90.0, 105.0)),
        _seed({"expired": 1, None: 1}, verdict="dip", confidence=0.8),
    ]
    every_interval = await _track_record(server_url, tmp_path, seeds)
    one_day = await _track_record(server_url, tmp_path, [], {"interval": "1d"})

    # (1/3 * 2 + 2/3 * 3) / 5 over every Interval; (1/3 * 2 + 2/3 * 2) / 4 over 1d.
    assert [band["dip"] for band in every_interval["bands"]] == [
        {"target": 2, "stop": 3, "expired": 1, "pending": 1, "hit_rate": None, "break_even": pytest.approx(8 / 15)},
        {"target": 0, "stop": 0, "expired": 1, "pending": 1, "hit_rate": None, "break_even": None},
    ]
    assert [band["dip"]["break_even"] for band in one_day["bands"]] == [pytest.approx(1 / 2), None]


async def test_ticker_filter_applies_to_dip_rows_and_their_dip_break_even(server_url, tmp_path):
    seeds = [
        # Default dip levels: Stop 96 and Target 108 at a Price of 100, break-even 1/3.
        _seed({"target": 1, "stop": 1}, verdict="dip", confidence=0.6),
        # Stop 90 and Target 105: break-even 2/3.
        _seed({"stop": 2, None: 1}, verdict="dip", confidence=0.6, ticker="MSFT", long=(90.0, 105.0)),
        _seed({"expired": 1}, verdict="dip", confidence=0.8, ticker="MSFT"),
    ]
    await _track_record(server_url, tmp_path, seeds)
    msft = await _track_record(server_url, tmp_path, [], {"ticker": "msft"})

    assert [band["dip"] for band in msft["bands"]] == [
        {"target": 0, "stop": 2, "expired": 0, "pending": 1, "hit_rate": None, "break_even": pytest.approx(2 / 3)},
        {"target": 0, "stop": 0, "expired": 1, "pending": 0, "hit_rate": None, "break_even": None},
    ]


@pytest.mark.parametrize("args", [{}, {"ticker": "msft", "interval": "1h"}])
async def test_dip_min_confidence_matches_get_signal_whatever_the_filter(signal_maker, server_url, tmp_path, args):
    db = tmp_path / "s.db"
    # 9 of 30 under the 1/3 Dip Break-even, then 12 of 30 above it: the Dip Minimum Confidence is band 0.5.
    seed_signals(db, ["target"] * 9 + ["stop"] * 21, verdict="dip", confidence=0.4)
    seed_signals(db, ["target"] * 12 + ["stop"] * 18, verdict="dip", confidence=0.5)
    reply = await _track_record(server_url, tmp_path, [], args)
    signal = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(), db_path=db).make("tsla", "1d")

    assert reply["dip_min_confidence"] == signal["dip_min_confidence"] == 0.5
