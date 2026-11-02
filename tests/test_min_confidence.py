"""Tests for the automatic Minimum Confidence (#55): set from Hit Rates per Confidence band."""

import json

import pytest

from fakes import FakeClassifier, FakeFetcher, call_tool, flat_history, seed_signals, text_of


def _signal(verdict="buy", confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


async def test_no_scored_signals_applies_0_6(server_url):
    result = await call_tool(
        server_url(FakeFetcher(history=flat_history()), classifier=FakeClassifier(_signal("buy", 0.6))), "get_signal", {"ticker": "AAPL"}
    )

    assert not result.is_error, text_of(result)
    payload = json.loads(text_of(result))
    assert payload["min_confidence"] == 0.6
    assert payload["trade_plan"]["direction"] == "long"


def _seed(db, targets, stops, expired=0, **kwargs):
    seed_signals(db, ["target"] * targets + ["stop"] * stops + ["expired"] * expired, **kwargs)


async def _reply(server_url, tmp_path, *seeds, signal=None):
    """get_signal reply after seeding _seed kwargs."""
    db = tmp_path / "s.db"
    url = server_url(FakeFetcher(history=flat_history()), classifier=FakeClassifier(signal or _signal()), db_path=db)
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

