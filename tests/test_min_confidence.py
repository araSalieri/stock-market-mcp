"""Tests for the automatic Minimum Confidence: set from Hit Rates per Confidence band, applied by SignalMaker."""

import pytest

from fakes import FakeClassifier, FakeFetcher, flat_history, seed_signals


def _signal(verdict="buy", confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test"}


def test_no_scored_signals_applies_0_6(signal_maker):
    payload = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(_signal("buy", 0.6))).make("AAPL")

    assert payload["min_confidence"] == 0.6
    assert payload["trade_plan"] is not None


def _seed(db, targets, stops, expired=0, **kwargs):
    seed_signals(db, ["target"] * targets + ["stop"] * stops + ["expired"] * expired, **kwargs)


def _reply(signal_maker, tmp_path, *seeds, signal=None):
    """SignalMaker reply after seeding _seed kwargs."""
    db = tmp_path / "s.db"
    maker = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(signal or _signal()), db_path=db)
    for seed in seeds:
        _seed(db, **seed)
    return maker.make("AAPL")


def test_lowest_band_above_break_even_is_applied(signal_maker, tmp_path):
    seeds = [
        {"confidence": 0.45, "targets": 10, "stops": 20},  # 0.4 band at exactly 1/3, not above
        {"confidence": 0.5, "targets": 6, "stops": 9, "ticker": "MSFT"},  # 0.5 band across tickers
        {"confidence": 0.59, "targets": 6, "stops": 9, "interval": "1h"},  # and Intervals: 12/30
        {"confidence": 0.8, "targets": 20, "stops": 10},
    ]
    payload = _reply(signal_maker, tmp_path, *seeds, signal=_signal("buy", 0.5))

    assert payload["min_confidence"] == 0.5
    assert payload["trade_plan"] is not None


def test_bands_under_30_targets_and_stops_are_ignored_and_expired_dont_count(signal_maker, tmp_path):
    seeds = [
        {"confidence": 0.3, "targets": 29, "stops": 0, "expired": 10},
        {"confidence": 0.7, "targets": 20, "stops": 10},
    ]
    assert (_reply(signal_maker, tmp_path, *seeds))["min_confidence"] == 0.7


def test_older_prompt_versions_are_ignored(signal_maker, tmp_path):
    seeds = [
        {"confidence": 0.3, "targets": 30, "stops": 0, "prompt_version": 0},
        {"confidence": 0.7, "targets": 20, "stops": 10},
    ]
    assert (_reply(signal_maker, tmp_path, *seeds))["min_confidence"] == 0.7


def test_enough_data_with_no_band_above_break_even_gives_no_trade_plan(signal_maker, tmp_path):
    seeds = [{"confidence": 0.9, "targets": 10, "stops": 20}, {"confidence": 0.3, "targets": 20, "stops": 0}]
    payload = _reply(signal_maker, tmp_path, *seeds, signal=_signal("buy", 1.0))

    assert payload["min_confidence"] is None
    assert payload["trade_plan"] is None


def test_cut_outcomes_do_not_move_it(signal_maker, tmp_path):
    seeds = [
        {"confidence": 0.5, "targets": 30, "stops": 0, "verdict": "cut"},
        {"confidence": 0.7, "targets": 20, "stops": 10},
    ]
    assert (_reply(signal_maker, tmp_path, *seeds))["min_confidence"] == 0.7
