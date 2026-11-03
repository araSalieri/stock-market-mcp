"""Tests for SignalMaker: the Signal cache and the Trade Plan, called directly with fakes and a SQLite file."""

import json

import pytest
from fakes import FakeClassifier, FakeFetcher, flat_history, intraday_history, seed_signals

from stock_market_mcp.errors import InputError


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test"}


def test_second_signal_within_the_ttl_is_served_from_the_cache(signal_maker):
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier)

    first = maker.make("AAPL")
    second = maker.make("AAPL")

    assert second == first
    assert len(classifier.prompts) == 1  # the stored Signal doesn't change the cache key


def test_signal_after_the_ttl_reinvokes_the_classifier(signal_maker):
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, ttl=0)

    first = maker.make("AAPL")
    second = maker.make("AAPL")

    assert second == first
    assert len(classifier.prompts) == 2


def test_failed_signal_is_never_cached(signal_maker):
    classifier = FakeClassifier(error=RuntimeError("openrouter 500"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="openrouter 500"):
            maker.make("AAPL")

    assert len(classifier.prompts) == 2


def test_cached_signal_gets_the_current_minimum_confidence(signal_maker, tmp_path):
    db = tmp_path / "s.db"
    classifier = FakeClassifier(_signal("buy", 0.7))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, db_path=db)

    first = maker.make("AAPL")
    seed_signals(db, ["target"] * 20 + ["stop"] * 10, confidence=0.8)  # the 0.8 band beats break-even
    second = maker.make("AAPL")

    assert first["min_confidence"] == 0.6
    assert second["min_confidence"] == 0.8
    assert second["trade_plan"] is None  # 0.7 is now below it
    assert len(classifier.prompts) == 1


def test_buy_carries_a_long_trade_plan(signal_maker):
    # Entry 100, ATR 2: Stop = 100 - 2*2 = 96, Target = 100 + 2*(100 - 96) = 108.
    reply = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(_signal("buy"))).make("AAPL")

    assert reply == {
        **_signal("buy"), "data_date": "2024-02-29T00:00:00",
        "price": 100.0,
        "hit_rate": None,
        "min_confidence": 0.6,
        "trade_plan": {
            "entry": 100.0,
            "stop": pytest.approx(96.0),
            "target": pytest.approx(108.0),
            "atr": pytest.approx(2.0),
            "atr_multiple": 2.0,
            "reward_ratio": 2.0,
        },
    }


def test_hold_has_no_trade_plan_but_carries_the_price(signal_maker):
    fetcher = FakeFetcher(history=flat_history(last_close=123.5))
    reply = signal_maker(fetcher, FakeClassifier(_signal("hold"))).make("AAPL")

    assert reply == {**_signal("hold"), "data_date": "2024-02-29T00:00:00", "price": 123.5, "hit_rate": None, "min_confidence": 0.6, "trade_plan": None}


def test_buy_below_the_default_minimum_confidence_has_no_trade_plan(signal_maker):
    reply = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(_signal("buy", 0.59))).make("AAPL")

    assert reply == {**_signal("buy", 0.59), "data_date": "2024-02-29T00:00:00", "price": 100.0, "hit_rate": None, "min_confidence": 0.6, "trade_plan": None}


# Entry 100, ATR 2, atr_multiple 1.5, reward_ratio 3: Stop = 100 - 1.5*2 = 97, Target = 100 + 3*(100-97) = 109.
def test_custom_atr_multiple_and_reward_ratio_change_the_plan_and_are_echoed(signal_maker):
    maker = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(_signal("buy")))
    plan = maker.make("AAPL", atr_multiple=1.5, reward_ratio=3)["trade_plan"]

    assert plan["stop"] == pytest.approx(97.0)
    assert plan["target"] == pytest.approx(109.0)
    assert plan["atr_multiple"] == 1.5
    assert plan["reward_ratio"] == 3


# Entry 1, ATR 2: Stop = 1 - 2*2 = -3.
def test_non_positive_plan_price_is_an_input_error(signal_maker):
    maker = signal_maker(FakeFetcher(history=flat_history(close=1.0)), FakeClassifier(_signal("buy")))

    with pytest.raises(InputError, match="non-positive"):
        maker.make("AAPL")


@pytest.mark.parametrize("verdict", ["buy", "hold"])
def test_nan_last_close_is_an_input_error_before_the_classifier(signal_maker, verdict):
    classifier = FakeClassifier(_signal(verdict))
    maker = signal_maker(FakeFetcher(history=flat_history(last_close=float("nan"))), classifier)

    with pytest.raises(InputError, match="non-finite close"):
        maker.make("AAPL")

    assert classifier.prompts == []


def test_cached_signal_carries_the_price_and_a_recomputed_plan(signal_maker):
    # atr_multiple 1: Stop = 100 - 1*2 = 98, Target = 100 + 2*(100 - 98) = 104.
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier)

    first = maker.make("AAPL")
    second = maker.make("AAPL", atr_multiple=1)

    assert first["trade_plan"]["stop"] == pytest.approx(96.0)
    assert second["price"] == 100.0
    assert second["trade_plan"]["stop"] == pytest.approx(98.0)
    assert second["trade_plan"]["target"] == pytest.approx(104.0)
    assert len(classifier.prompts) == 1


def test_prompt_carries_the_interval_and_its_default_period(signal_maker):
    classifier = FakeClassifier(_signal("hold"))
    signal_maker(FakeFetcher(history=intraday_history()), classifier).make("AAPL", "5m")

    payload = json.loads(classifier.prompts[0])
    assert (payload["interval"], payload["period"]) == ("5m", "1mo")


def test_intraday_trade_plan_uses_the_intraday_atr(signal_maker):
    # 5m bars with High/Low 0.25 from a flat 100 close: ATR(14) is 0.5.
    # Stop = 100 - 2*0.5 = 99, Target = 100 + 2*(100 - 99) = 102.
    maker = signal_maker(FakeFetcher(history=intraday_history()), FakeClassifier(_signal("buy")))
    plan = maker.make("BBCA.JK", "5m")["trade_plan"]

    assert plan["atr"] == pytest.approx(0.5)
    assert plan["stop"] == pytest.approx(99.0)
    assert plan["target"] == pytest.approx(102.0)


def test_default_call_uses_5m_bars_over_a_month(signal_maker):
    fetcher, classifier = FakeFetcher(history=flat_history()), FakeClassifier(_signal("buy"))
    signal_maker(fetcher, classifier).make("AAPL")

    assert fetcher.history_calls == [("AAPL", "1mo", "5m", False)]
    payload = json.loads(classifier.prompts[0])
    assert (payload["period"], payload["interval"]) == ("1mo", "5m")
    assert payload["rows"][-1]["Date"].startswith("2024-02-29")


def test_data_date_is_the_latest_bars_timestamp_not_the_classifiers(signal_maker):
    # The classifier reply carries no date; SignalMaker reads it from the bars it fetched.
    classifier = FakeClassifier({"verdict": "hold", "confidence": 0.5, "rationale": "test"})
    maker = signal_maker(FakeFetcher(history=intraday_history()), classifier)

    assert maker.make("AAPL")["data_date"] == "2024-03-01T13:55:00+07:00"
