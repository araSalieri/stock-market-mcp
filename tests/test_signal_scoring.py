"""Tests for scoring Signals: pending Signals get an Outcome from later bars."""

import json

import pandas as pd
import pytest
from fakes import FakeClassifier, FakeFetcher, flat_history, seed_signals

from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.prompt import PROMPT_VERSION
from stock_market_mcp.signals import SignalLog


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test"}


def _later(*bars):
    """The 60 flat bars the first Signal is made on, then later bars given as (high, low)."""
    df = flat_history(rows=60 + len(bars))
    for i, (high, low) in enumerate(bars):
        df.loc[df.index[60 + i], ["High", "Low"]] = [high, low]
    return df


FLAT = (101.0, 99.0)


def _outcome(signal_maker, verdict, *bars, confidence=0.7, first=None, scored_on=None):
    """Outcome the classifier sees for one Signal made on first (default: 60 flat bars), then scored on scored_on (default: _later(*bars))."""
    # Entry 100, ATR 2, default plans: long Stop 96 / Target 108, mirrored: Stop 104 / Target 92.
    first = flat_history() if first is None else first
    scored_on = _later(*bars) if scored_on is None else scored_on
    classifier = FakeClassifier(replies=[_signal(verdict, confidence), _signal("hold")])
    maker = signal_maker(FakeFetcher(history=[first, scored_on]), classifier, ttl=0)
    maker.make("AAPL", "1d")
    maker.make("AAPL", "1d")
    return json.loads(classifier.prompts[1])["previous_signal"]["outcome"]


def test_buy_reaching_its_target_first_is_a_target(signal_maker):
    assert _outcome(signal_maker, "buy", FLAT, (109.0, 99.0), (101.0, 90.0)) == "target"


def _cut_outcome(signal_maker, tmp_path, *bars, buyback="low_20", cut_stop="high_20"):
    """Outcome of a cut on a held AAPL, made on 60 flat bars, then scored on _later(*bars).

    On flat bars the picks are the 20-bar low 99 (Buyback Price) and high 101 (Cut Stop).
    """
    PortfolioStore(tmp_path / "server.db").set("AAPL", 50, 90.0)
    picks = {"buyback": {"choice": buyback, "confidence": 0.8}, "cut_stop": {"choice": cut_stop, "confidence": 0.6}}
    classifier = FakeClassifier(replies=[{**_signal("cut", 0.3), "levels": picks}, _signal("hold")])
    maker = signal_maker(FakeFetcher(history=[flat_history(), _later(*bars)]), classifier, ttl=0)
    maker.make("AAPL", "1d")
    maker.make("AAPL", "1d")
    return json.loads(classifier.prompts[1])["previous_signal"]["outcome"]


QUIET = (100.5, 99.5)


def test_cut_falling_to_its_buyback_price_first_is_a_target(signal_maker, tmp_path):
    assert _cut_outcome(signal_maker, tmp_path, QUIET, (100.5, 98.5), (101.5, 99.5)) == "target"


def test_cut_rising_to_its_cut_stop_first_is_a_stop(signal_maker, tmp_path):
    assert _cut_outcome(signal_maker, tmp_path, QUIET, (101.5, 99.5), (100.5, 98.5)) == "stop"


def test_cut_reaching_both_levels_in_one_bar_is_a_stop(signal_maker, tmp_path):
    assert _cut_outcome(signal_maker, tmp_path, QUIET, (101.5, 98.5)) == "stop"


def test_cut_reaching_neither_level_within_20_later_bars_is_expired(signal_maker, tmp_path):
    assert _cut_outcome(signal_maker, tmp_path, *[QUIET] * 20, (100.5, 98.5)) == "expired"


@pytest.mark.parametrize("picks", [{"buyback": "none"}, {"cut_stop": "none"}])
def test_cut_with_a_none_pick_is_never_scored(signal_maker, tmp_path, picks):
    assert _cut_outcome(signal_maker, tmp_path, (101.5, 98.5), *[QUIET] * 20, **picks) == "pending"
    # Only the later hold's 0.7 band: the 0.3 cut is left out.
    assert [band["band"] for band in SignalLog(tmp_path / "server.db").track_record(None, None)["bands"]] == [0.7]


def test_pending_cut_from_prompt_version_3_is_scored_at_its_stored_distances(signal_maker, tmp_path):
    # seed_signals stores the mirrored default distances: Stop 104, Target 92.
    seed_signals(tmp_path / "server.db", [None], verdict="cut", prompt_version=3)
    history = _later((103.0, 91.0))
    history.index = pd.date_range(end="2023-01-02", periods=len(history), freq="D", name="Date")
    classifier = FakeClassifier(replies=[_signal("hold")])
    signal_maker(FakeFetcher(history=history), classifier, ttl=0).make("AAPL", "1d")
    assert json.loads(classifier.prompts[0])["previous_signal"]["outcome"] == "target"


def test_pending_sell_from_an_older_prompt_version_is_scored_as_a_cut(signal_maker, tmp_path):
    seed_signals(tmp_path / "server.db", [None], verdict="sell", prompt_version=PROMPT_VERSION - 1)
    history = _later((101.0, 91.0))
    # seed_signals stores its Signals on 2023-01-01: make that the bar before the drop.
    history.index = pd.date_range(end="2023-01-02", periods=len(history), freq="D", name="Date")
    classifier = FakeClassifier(replies=[_signal("hold")])
    signal_maker(FakeFetcher(history=history), classifier, ttl=0).make("AAPL", "1d")
    assert json.loads(classifier.prompts[0])["previous_signal"]["outcome"] == "target"


def test_buy_reaching_its_stop_first_is_a_stop(signal_maker):
    assert _outcome(signal_maker, "buy", FLAT, (101.0, 95.0), (109.0, 99.0)) == "stop"


def test_stop_and_target_in_one_bar_is_a_stop(signal_maker):
    assert _outcome(signal_maker, "buy", FLAT, (109.0, 95.0)) == "stop"


def test_neither_within_20_later_bars_is_expired(signal_maker):
    assert _outcome(signal_maker, "buy", *[FLAT] * 20, (109.0, 99.0)) == "expired"


def test_fewer_than_20_later_bars_and_no_hit_stays_pending(signal_maker):
    assert _outcome(signal_maker, "buy", *[FLAT] * 19) == "pending"


def test_hold_reaching_neither_target_within_20_later_bars_is_quiet(signal_maker):
    assert _outcome(signal_maker, "hold", *[FLAT] * 20, (109.0, 91.0)) == "quiet"


@pytest.mark.parametrize("bar", [(109.0, 99.0), (101.0, 91.0)], ids=["up", "down"])
def test_hold_reaching_either_target_is_missed(signal_maker, bar):
    assert _outcome(signal_maker, "hold", FLAT, bar) == "missed"


def test_hold_with_fewer_than_20_later_bars_and_no_target_stays_pending(signal_maker):
    assert _outcome(signal_maker, "hold", *[FLAT] * 19) == "pending"


def test_buy_below_the_minimum_confidence_is_scored_too(signal_maker):
    assert _outcome(signal_maker, "buy", (109.0, 99.0), confidence=0.3) == "target"


def test_signal_whose_bar_is_outside_the_history_stays_pending(signal_maker):
    # The history starts on the bar after the Signal's, which reaches the Target.
    assert _outcome(signal_maker, "buy", scored_on=_later((109.0, 99.0), *[FLAT] * 59).iloc[60:]) == "pending"


def test_signal_without_a_default_plan_stays_pending(signal_maker):
    # Close 1 with ATR 2 puts the long default Stop below zero, so it is stored null.
    # Below the Minimum Confidence, so no Trade Plan is asked for and the Signal is still stored.
    first = flat_history(close=1.0)
    outcome = _outcome(signal_maker, "buy", confidence=0.5, first=first, scored_on=flat_history(close=1.0, rows=90))
    assert outcome == "pending"


def test_second_call_scores_the_first_before_the_classifier_sees_it(signal_maker):
    classifier = FakeClassifier(replies=[_signal("buy"), _signal("hold")])
    maker = signal_maker(FakeFetcher(history=[flat_history(), _later(FLAT, (109.0, 99.0))]), classifier, ttl=0)

    maker.make("AAPL", "1d")
    maker.make("AAPL", "1d")

    assert json.loads(classifier.prompts[1])["previous_signal"]["outcome"] == "target"


def _hit_rate(signal_maker, tmp_path, *seeds, reply="buy"):
    """hit_rate in the SignalMaker reply after seeding (outcomes, seed_signals kwargs) pairs."""
    db = tmp_path / "s.db"
    maker = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(_signal(reply)), db_path=db)
    for outcomes, kwargs in seeds:
        seed_signals(db, outcomes, **kwargs)
    return maker.make("AAPL", "1d")["hit_rate"]


def test_hit_rate_is_targets_over_targets_and_stops_ignoring_expired(signal_maker, tmp_path):
    seeds = [(["target"] * 3 + ["stop"] * 7 + ["expired"] * 5, {})]
    assert _hit_rate(signal_maker, tmp_path, *seeds) == pytest.approx(3 / 10)


def test_buy_reply_ignores_scored_cuts(signal_maker, tmp_path):
    seeds = [(["target"] * 9, {}), (["stop"] * 10, {"verdict": "cut"})]
    assert _hit_rate(signal_maker, tmp_path, *seeds) is None


def test_hit_rate_is_null_under_10_scored(signal_maker, tmp_path):
    assert _hit_rate(signal_maker, tmp_path, (["target"] * 9 + ["expired"] * 5, {})) is None


def test_hit_rate_counts_only_the_current_prompt_version(signal_maker, tmp_path):
    seeds = [(["target"] * 9, {}), (["stop"] * 10, {"prompt_version": 0})]
    assert _hit_rate(signal_maker, tmp_path, *seeds) is None


def test_hold_reply_has_the_hold_hit_rate(signal_maker, tmp_path):
    seeds = [(["quiet"] * 3 + ["missed"] * 7, {"verdict": "hold"}), (["target"] * 10, {})]
    assert _hit_rate(signal_maker, tmp_path, *seeds, reply="hold") == pytest.approx(3 / 10)


def test_buy_reply_ignores_scored_holds(signal_maker, tmp_path):
    seeds = [(["target"] * 9, {}), (["quiet"] * 10, {"verdict": "hold"})]
    assert _hit_rate(signal_maker, tmp_path, *seeds) is None


def test_hold_hit_rate_is_null_under_10_scored_holds(signal_maker, tmp_path):
    assert _hit_rate(signal_maker, tmp_path, (["quiet"] * 9, {"verdict": "hold"}), reply="hold") is None
