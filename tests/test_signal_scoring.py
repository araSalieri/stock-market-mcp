"""Tests for scoring buy/sell Signals (#53): get_signal scores pending Signals on the history it fetched."""

import json

import pytest
from fakes import FakeClassifier, FlatFakeFetcher, SequenceFakeFetcher, call_tool, flat_history, seed_signals, text_of


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


def _later(*bars):
    """The 60 flat bars the first Signal is made on, then later bars given as (high, low)."""
    df = flat_history(rows=60 + len(bars))
    for i, (high, low) in enumerate(bars):
        df.loc[df.index[60 + i], ["High", "Low"]] = [high, low]
    return df


FLAT = (101.0, 99.0)


async def _previous_outcome(server_url, verdict, *bars, confidence=0.7):
    # Entry 100, ATR 2, default plans: long Stop 96 / Target 108, short Stop 104 / Target 92.
    classifier = FakeClassifier(replies=[_signal(verdict, confidence), _signal("hold")])
    url = server_url(SequenceFakeFetcher([flat_history(), _later(*bars)]), classifier=classifier, signal_ttl=0, history_ttl=0)

    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "1d"})
    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "1d"})

    return json.loads(classifier.prompts[1])["previous_signal"]["outcome"]


async def test_buy_reaching_its_target_first_is_a_target(server_url):
    assert await _previous_outcome(server_url, "buy", FLAT, (109.0, 99.0), (101.0, 90.0)) == "target"


async def test_sell_reaching_its_target_first_is_a_target(server_url):
    assert await _previous_outcome(server_url, "sell", FLAT, (101.0, 91.0), (105.0, 99.0)) == "target"


async def test_buy_reaching_its_stop_first_is_a_stop(server_url):
    assert await _previous_outcome(server_url, "buy", FLAT, (101.0, 95.0), (109.0, 99.0)) == "stop"


async def test_sell_reaching_its_stop_first_is_a_stop(server_url):
    assert await _previous_outcome(server_url, "sell", (105.0, 99.0), (101.0, 91.0)) == "stop"


async def test_stop_and_target_in_one_bar_is_a_stop(server_url):
    assert await _previous_outcome(server_url, "buy", FLAT, (109.0, 95.0)) == "stop"


async def test_neither_within_20_later_bars_is_expired(server_url):
    assert await _previous_outcome(server_url, "buy", *[FLAT] * 20, (109.0, 99.0)) == "expired"


async def test_fewer_than_20_later_bars_and_no_hit_stays_pending(server_url):
    assert await _previous_outcome(server_url, "buy", *[FLAT] * 19) == "pending"



async def test_hold_reaching_neither_target_within_20_later_bars_is_quiet(server_url):
    assert await _previous_outcome(server_url, "hold", *[FLAT] * 20, (109.0, 91.0)) == "quiet"


@pytest.mark.parametrize("bar", [(109.0, 99.0), (101.0, 91.0)], ids=["long", "short"])
async def test_hold_reaching_either_target_is_missed(server_url, bar):
    assert await _previous_outcome(server_url, "hold", FLAT, bar) == "missed"


async def test_hold_with_fewer_than_20_later_bars_and_no_target_stays_pending(server_url):
    assert await _previous_outcome(server_url, "hold", *[FLAT] * 19) == "pending"

async def test_buy_below_the_minimum_confidence_is_scored_too(server_url):
    assert await _previous_outcome(server_url, "buy", (109.0, 99.0), confidence=0.3) == "target"


async def test_signal_whose_bar_is_outside_the_fetched_history_stays_pending(server_url):
    classifier = FakeClassifier(replies=[_signal("buy"), _signal("hold")])
    # The second history starts on the bar after the Signal's, which reaches the Target.
    fetcher = SequenceFakeFetcher([flat_history(), _later((109.0, 99.0), *[FLAT] * 59).iloc[60:]])
    url = server_url(fetcher, classifier=classifier, signal_ttl=0, history_ttl=0)

    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "1d"})
    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "1d"})

    assert json.loads(classifier.prompts[1])["previous_signal"]["outcome"] == "pending"
    assert len(fetcher.history_calls) == 2  # no extra fetch to reach the window


async def test_signal_without_a_default_plan_stays_pending(server_url):
    # Close 1 with ATR 2 puts the long default Stop below zero, so it is stored null.
    # Below the Minimum Confidence, so the call gets no Trade Plan and doesn't fail on it.
    classifier = FakeClassifier(replies=[_signal("buy", 0.3), _signal("hold")])
    fetcher = SequenceFakeFetcher([flat_history(close=1.0), flat_history(close=1.0, rows=90)])
    url = server_url(fetcher, classifier=classifier, signal_ttl=0, history_ttl=0)

    await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "1d"})
    second = await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "1d"})

    assert not second.is_error
    assert json.loads(classifier.prompts[1])["previous_signal"]["outcome"] == "pending"


async def _hit_rate(server_url, tmp_path, *seeds, reply="buy"):
    """hit_rate in the reply after seeding (outcomes, seed_signals kwargs) pairs."""
    db = tmp_path / "s.db"
    url = server_url(FlatFakeFetcher(), classifier=FakeClassifier(_signal(reply)), db_path=db)
    for outcomes, kwargs in seeds:
        seed_signals(db, outcomes, **kwargs)
    result = await call_tool(url, "get_signal", {"ticker": "AAPL", "interval": "1d"})
    return json.loads(text_of(result))["hit_rate"]


async def test_hit_rate_is_targets_over_targets_and_stops_ignoring_expired(server_url, tmp_path):
    seeds = [(["target"] * 3 + ["stop"] * 7 + ["expired"] * 5, {}), (["stop"], {"verdict": "sell"})]
    assert await _hit_rate(server_url, tmp_path, *seeds) == pytest.approx(3 / 11)


async def test_hit_rate_is_null_under_10_scored(server_url, tmp_path):
    assert await _hit_rate(server_url, tmp_path, (["target"] * 9 + ["expired"] * 5, {})) is None


async def test_hit_rate_counts_only_the_current_prompt_version(server_url, tmp_path):
    seeds = [(["target"] * 9, {}), (["stop"] * 10, {"prompt_version": 0})]
    assert await _hit_rate(server_url, tmp_path, *seeds) is None


async def test_hold_reply_has_the_hold_hit_rate(server_url, tmp_path):
    seeds = [(["quiet"] * 3 + ["missed"] * 7, {"verdict": "hold"}), (["target"] * 10, {})]
    assert await _hit_rate(server_url, tmp_path, *seeds, reply="hold") == pytest.approx(3 / 10)


async def test_buy_reply_ignores_scored_holds(server_url, tmp_path):
    seeds = [(["target"] * 9, {}), (["quiet"] * 10, {"verdict": "hold"})]
    assert await _hit_rate(server_url, tmp_path, *seeds) is None


async def test_hold_hit_rate_is_null_under_10_scored_holds(server_url, tmp_path):
    assert await _hit_rate(server_url, tmp_path, (["quiet"] * 9, {"verdict": "hold"}), reply="hold") is None
