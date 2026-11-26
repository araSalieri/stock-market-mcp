"""The dip Verdict on a ticker not held: its question, picked levels, Trade Plan and Dip Minimum Confidence, through SignalMaker with fakes."""

import json

import pytest
from fakes import FakeClassifier, FakeFetcher, flat_history, seed_signals

from stock_market_mcp.portfolio import PortfolioStore

DIP = (
    "Price stretched below its mean (near or under the lower Bollinger Band, RSI oversold) while the longer trend "
    "still points up: a pullback likely to snap back."
)


def _signal(verdict="dip", confidence=0.7, target="high_20", stop="low_20"):
    # Flat close 100 with High 101 and Low 99: the 20-bar high (101) and low (99) are the only levels off the Price.
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": "test",
        "picks": {
            "dip_target": {"choice": target, "confidence": 0.8},
            "dip_stop": {"choice": stop, "confidence": 0.6},
        },
    }


def _make(signal_maker, tmp_path, classifier, held=None, history=None, **kwargs):
    """The 1d reply for AAPL at a flat close of 100 (or history), with a Position of 50 @ 90 in held if given."""
    db = tmp_path / "smm.db"
    if held:
        PortfolioStore(db).set(held, 50, 90.0)
    maker = signal_maker(FakeFetcher(history=flat_history() if history is None else history), classifier, db_path=db)
    return maker.make("aapl", "1d", **kwargs)


def test_not_held_is_asked_buy_dip_or_hold_and_the_dip_levels(signal_maker, tmp_path):
    classifier = FakeClassifier({"verdict": "hold", "confidence": 0.7, "rationale": "test"})
    _make(signal_maker, tmp_path, classifier)

    [questions] = classifier.questions
    assert list(questions) == ["verdict", "dip_target", "dip_stop"]
    criteria = questions["verdict"]["criteria"]
    assert list(criteria) == ["buy", "dip", "hold"]
    assert criteria["dip"] == DIP
    assert criteria["buy"] == "Bullish momentum or trend evidence in the supplied data."
    assert questions["dip_target"]["criteria"] == {"high_20": "20-bar high at 101", "none": "No level above the Price fits."}
    assert questions["dip_stop"]["criteria"] == {"low_20": "20-bar low at 99", "none": "No level below the Price fits."}


def test_held_is_never_offered_dip(signal_maker, tmp_path):
    classifier = FakeClassifier({"verdict": "hold", "confidence": 0.7, "rationale": "test"})
    _make(signal_maker, tmp_path, classifier, held="AAPL")

    [questions] = classifier.questions
    assert list(questions) == ["verdict", "buyback", "cut_stop"]
    assert list(questions["verdict"]["criteria"]) == ["buy", "hold", "cut"]


def test_confident_dip_gets_a_trade_plan_from_the_picked_levels(signal_maker, tmp_path):
    reply = _make(signal_maker, tmp_path, FakeClassifier(_signal()))

    # Stop 1 below the Price of 100 and Target 1 above: a Reward Ratio of 1.
    assert reply["trade_plan"] == {"entry": 100.0, "stop": 99.0, "target": 101.0, "reward_ratio": 1.0}
    assert reply["dip_min_confidence"] == 0.4
    assert not {"picks", "dip_levels"} & reply.keys()


def test_dip_below_the_dip_minimum_confidence_has_no_trade_plan(signal_maker, tmp_path):
    reply = _make(signal_maker, tmp_path, FakeClassifier(_signal(confidence=0.39)))

    assert reply["verdict"] == "dip"
    assert reply["trade_plan"] is None


@pytest.mark.parametrize("pick", [{"target": "none"}, {"stop": "none"}])
def test_dip_missing_a_level_has_no_trade_plan(signal_maker, tmp_path, pick):
    reply = _make(signal_maker, tmp_path, FakeClassifier(_signal(confidence=0.9, **pick)))

    assert reply["trade_plan"] is None


def test_dip_ignores_atr_multiple_and_reward_ratio(signal_maker, tmp_path):
    reply = _make(signal_maker, tmp_path, FakeClassifier(_signal()), atr_multiple=1, reward_ratio=3)

    assert reply["trade_plan"] == {"entry": 100.0, "stop": 99.0, "target": 101.0, "reward_ratio": 1.0}


def test_held_reply_has_no_dip_minimum_confidence(signal_maker, tmp_path):
    reply = _make(signal_maker, tmp_path, FakeClassifier({"verdict": "hold", "confidence": 0.7, "rationale": "test"}), held="AAPL")

    assert "dip_min_confidence" not in reply


def _later(*bars):
    """The 60 flat bars the dip is made on, then later bars given as (high, low)."""
    df = flat_history(rows=60 + len(bars))
    for i, (high, low) in enumerate(bars):
        df.loc[df.index[60 + i], ["High", "Low"]] = [high, low]
    return df


QUIET = (100.5, 99.5)


def _outcome(signal_maker, *bars, target="high_20", stop="low_20"):
    """Outcome the classifier sees for a dip made on 60 flat bars (Target 101, Stop 99), then scored on _later(*bars)."""
    classifier = FakeClassifier(replies=[_signal(confidence=0.3, target=target, stop=stop), {"verdict": "hold", "confidence": 0.7, "rationale": "test"}])
    maker = signal_maker(FakeFetcher(history=[flat_history(), _later(*bars)]), classifier, ttl=0)
    maker.make("AAPL", "1d")
    maker.make("AAPL", "1d")
    return json.loads(classifier.prompts[1])["previous_signal"]["outcome"]


@pytest.mark.parametrize(
    ("bars", "outcome"),
    [
        ([QUIET, (101.5, 99.5), (100.5, 98.5)], "target"),
        ([QUIET, (100.5, 98.5), (101.5, 99.5)], "stop"),
        ([QUIET, (101.5, 98.5)], "stop"),
        ([*[QUIET] * 20, (101.5, 99.5)], "expired"),
    ],
)
def test_dip_is_scored_against_its_picked_levels(signal_maker, bars, outcome):
    assert _outcome(signal_maker, *bars) == outcome


@pytest.mark.parametrize("pick", [{"target": "none"}, {"stop": "none"}])
def test_dip_missing_a_level_is_never_scored(signal_maker, pick):
    assert _outcome(signal_maker, (101.5, 98.5), *[QUIET] * 20, **pick) == "pending"


def _seeded(signal_maker, tmp_path, *seeds, signal=None):
    """The reply for AAPL not held after seeding scored dips at a Price of 100.

    Each seed is seed_signals kwargs plus targets, stops and expired counts; seeded dips default to a Stop of 96 and
    a Target of 108, a Dip Break-even of 4 / (4 + 8) = 1/3.
    """
    db = tmp_path / "smm.db"
    for seed in seeds:
        seed = {"verdict": "dip", **seed}
        outcomes = ["target"] * seed.pop("targets", 0) + ["stop"] * seed.pop("stops", 0)
        seed_signals(db, outcomes + ["expired"] * seed.pop("expired", 0), **seed)
    maker = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(signal or _signal()), db_path=db)
    return maker.make("aapl", "1d")


def test_dip_minimum_confidence_is_0_4_until_a_band_has_30_scored_dips(signal_maker, tmp_path):
    reply = _seeded(signal_maker, tmp_path, {"confidence": 0.9, "stops": 29, "expired": 10}, {"confidence": 0.5, "targets": 29})

    assert reply["dip_min_confidence"] == 0.4


def test_dip_minimum_confidence_is_the_lowest_band_beating_its_dip_break_even_across_intervals(signal_maker, tmp_path):
    seeds = [
        # 0.3 band: break-even averages 1/3 and 0.8 over its 30 scored dips (0.567); 16 of 30 is under it.
        {"confidence": 0.3, "targets": 15},
        {"confidence": 0.3, "targets": 1, "stops": 14, "long": (92.0, 102.0)},
        {"confidence": 0.3, "expired": 30, "long": (99.0, 150.0)},  # Expired dips don't move break-even
        {"confidence": 0.4, "targets": 9, "stops": 21},  # under 1/3
        {"confidence": 0.5, "targets": 12, "stops": 18, "interval": "1h", "ticker": "MSFT"},  # 40%: above 1/3
        {"confidence": 0.8, "targets": 30},
    ]
    reply = _seeded(signal_maker, tmp_path, *seeds)

    assert reply["dip_min_confidence"] == 0.5


def test_no_trade_plan_when_no_band_beats_its_dip_break_even(signal_maker, tmp_path):
    # Stop 8 below and Target 2 above: break-even 0.8, so 20 of 30 isn't enough.
    reply = _seeded(signal_maker, tmp_path, {"confidence": 0.9, "targets": 20, "stops": 10, "long": (92.0, 102.0)}, signal=_signal(confidence=1.0))

    assert reply["dip_min_confidence"] is None
    assert reply["trade_plan"] is None


def test_dip_reply_has_the_dip_hit_rate_for_its_ticker_and_interval(signal_maker, tmp_path):
    seeds = [
        {"targets": 6, "stops": 4, "expired": 5},
        {"targets": 10, "interval": "1h"},
        {"targets": 10, "ticker": "MSFT"},
        {"verdict": "buy", "stops": 10},
    ]
    reply = _seeded(signal_maker, tmp_path, *seeds)

    assert reply["hit_rate"] == 0.6


def test_buy_minimum_confidence_and_trade_plan_ignore_dips(signal_maker, tmp_path):
    # The dips give a null Dip Minimum Confidence; the buy keeps the 0.4 default and its ATR Trade Plan.
    reply = _seeded(signal_maker, tmp_path, {"confidence": 0.6, "stops": 30}, signal=_signal("buy", 0.6))

    assert reply["dip_min_confidence"] is None
    assert reply["min_confidence"] == 0.4
    assert reply["trade_plan"]["atr_multiple"] == 2.0
