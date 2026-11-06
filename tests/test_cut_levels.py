"""The Buyback Price and Cut Stop on a confident cut (ADR-0010), through SignalMaker with fakes."""

import pytest
from fakes import FakeClassifier, FakeFetcher, flat_history

from stock_market_mcp.portfolio import PortfolioStore

LEVEL_KEYS = {"buyback_price", "buyback_quantity", "buyback_confidence", "cut_stop", "cut_stop_confidence"}


def _signal(verdict="cut", confidence=0.7, buyback="low_20", cut_stop="high_20"):
    # Flat close 100 with High 101 and Low 99: the 20-bar low (99) and high (101) are the only levels off the Price.
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": "test",
        "levels": {
            "buyback": {"choice": buyback, "confidence": 0.8},
            "cut_stop": {"choice": cut_stop, "confidence": 0.6},
        },
    }


def _make(signal_maker, tmp_path, classifier, held="AAPL", calls=1):
    """Replies for AAPL at a flat close of 100, a Position of 50 @ 90 in held."""
    db = tmp_path / "smm.db"
    PortfolioStore(db).set(held, 50, 90.0)
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, db_path=db)
    return [maker.make("aapl", "1d") for _ in range(calls)]


def test_confident_cut_gives_the_picked_levels_and_the_buyback_quantity(signal_maker, tmp_path):
    [reply] = _make(signal_maker, tmp_path, FakeClassifier(_signal()))

    # 50 shares at 100 buy back 5000 / 99 = 50.5 shares at 99, rounded down.
    assert {key: reply[key] for key in LEVEL_KEYS} == {
        "buyback_price": 99.0,
        "buyback_quantity": 50,
        "buyback_confidence": 0.8,
        "cut_stop": 101.0,
        "cut_stop_confidence": 0.6,
    }
    assert "levels" not in reply


def test_a_none_pick_is_null_with_its_confidence_and_quantity(signal_maker, tmp_path):
    [reply] = _make(signal_maker, tmp_path, FakeClassifier(_signal(buyback="none", cut_stop="none")))

    assert {key: reply[key] for key in LEVEL_KEYS} == dict.fromkeys(LEVEL_KEYS)


@pytest.mark.parametrize(("verdict", "confidence"), [("cut", 0.59), ("buy", 0.9), ("hold", 0.9)])
def test_only_a_confident_cut_has_levels(signal_maker, tmp_path, verdict, confidence):
    [reply] = _make(signal_maker, tmp_path, FakeClassifier(_signal(verdict, confidence)))

    assert not (LEVEL_KEYS | {"levels"}) & reply.keys()


def test_a_cached_signal_reuses_its_picks(signal_maker, tmp_path):
    classifier = FakeClassifier(_signal())
    first, second = _make(signal_maker, tmp_path, classifier, calls=2)

    assert len(classifier.prompts) == 1
    assert second["buyback_price"] == first["buyback_price"] == 99.0
    assert second["cut_stop"] == first["cut_stop"] == 101.0


@pytest.mark.parametrize(("held", "asked"), [("AAPL", ["verdict", "buyback", "cut_stop"]), ("MSFT", ["verdict"])])
def test_only_a_held_ticker_is_asked_for_levels(signal_maker, tmp_path, held, asked):
    classifier = FakeClassifier({"verdict": "hold", "confidence": 0.7, "rationale": "test"})
    _make(signal_maker, tmp_path, classifier, held=held)

    assert list(classifier.questions[0]) == asked
