"""Tests for the automatic Minimum Confidence (#55): set from Hit Rates per Confidence band."""

import json
import sqlite3
from contextlib import closing

import pytest

from fakes import FakeClassifier, FlatFakeFetcher, call_tool, text_of


def _signal(verdict="buy", confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


async def test_no_scored_signals_applies_0_6(server_url):
    result = await call_tool(
        server_url(FlatFakeFetcher(), classifier=FakeClassifier(_signal("buy", 0.6))), "get_signal", {"ticker": "AAPL"}
    )

    assert not result.is_error, text_of(result)
    payload = json.loads(text_of(result))
    assert payload["min_confidence"] == 0.6
    assert payload["trade_plan"]["direction"] == "long"


def _seed(db, confidence, targets, stops, expired=0, prompt_version=1, ticker="AAPL", interval="1d"):
    # 30+ scored Signals per band would take dozens of MCP calls, so these rows go in directly.
    outcomes = ["target"] * targets + ["stop"] * stops + ["expired"] * expired
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.executemany(
            "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, outcome) "
            "VALUES (?, ?, ?, '2023-01-01T00:00:00', 'buy', ?, 100.0, ?)",
            [(ticker, interval, prompt_version, confidence, o) for o in outcomes],
        )


async def _reply(server_url, tmp_path, *seeds, signal=None):
    """get_signal reply after seeding _seed kwargs; a first call creates the table."""
    db = tmp_path / "s.db"
    url = server_url(FlatFakeFetcher(), classifier=FakeClassifier(signal or _signal()), db_path=db, signal_ttl=0)
    await call_tool(url, "get_signal", {"ticker": "AAPL"})
    for seed in seeds:
        _seed(db, **seed)
    result = await call_tool(url, "get_signal", {"ticker": "AAPL"})
    assert not result.is_error, text_of(result)
    return json.loads(text_of(result))


async def test_lowest_band_above_break_even_is_applied(server_url, tmp_path):
    seeds = [
        {"confidence": 0.45, "targets": 10, "stops": 20},  # 0.4 band at exactly 1/3, not above
        {"confidence": 0.5, "targets": 6, "stops": 9, "ticker": "MSFT"},  # 0.5 band across tickers
        {"confidence": 0.59, "targets": 6, "stops": 9, "interval": "1h"},  # and Intervals: 12/30
        {"confidence": 0.8, "targets": 20, "stops": 10},
    ]
    payload = await _reply(server_url, tmp_path, *seeds, signal=_signal("buy", 0.5))

    assert payload["min_confidence"] == 0.5
    assert payload["trade_plan"]["direction"] == "long"


async def test_bands_under_30_targets_and_stops_are_ignored_and_expired_dont_count(server_url, tmp_path):
    seeds = [
        {"confidence": 0.3, "targets": 29, "stops": 0, "expired": 10},
        {"confidence": 0.7, "targets": 20, "stops": 10},
    ]
    assert (await _reply(server_url, tmp_path, *seeds))["min_confidence"] == 0.7


async def test_older_prompt_versions_are_ignored(server_url, tmp_path):
    seeds = [
        {"confidence": 0.3, "targets": 30, "stops": 0, "prompt_version": 0},
        {"confidence": 0.7, "targets": 20, "stops": 10},
    ]
    assert (await _reply(server_url, tmp_path, *seeds))["min_confidence"] == 0.7


@pytest.mark.parametrize("verdict", ["buy", "sell"])
async def test_enough_data_with_no_band_above_break_even_gives_no_trade_plan(server_url, tmp_path, verdict):
    seeds = [{"confidence": 0.9, "targets": 10, "stops": 20}, {"confidence": 0.3, "targets": 20, "stops": 0}]
    payload = await _reply(server_url, tmp_path, *seeds, signal=_signal(verdict, 1.0))

    assert payload["min_confidence"] is None
    assert payload["trade_plan"] is None

