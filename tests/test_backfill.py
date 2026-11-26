"""Backfilled Signals: SignalMaker.backfill makes first Signals for a ticker not held and scores them, through fakes."""

import json
from datetime import UTC, datetime

import pandas as pd
import pytest
from fakes import FakeClassifier, FakeFetcher, flat_history, seed_signals

from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PortfolioStore

NOW = datetime(2024, 12, 1, tzinfo=UTC)
# flat_history starts on 2024-01-01, so this is bar 60.
SINCE = "2024-03-01"


def _signal(verdict="buy", confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test"}


def _maker(signal_maker, tmp_path, history, classifier):
    return signal_maker(FakeFetcher(history=history), classifier, db_path=tmp_path / "smm.db", now=lambda: NOW)


def test_backfills_bars_from_since_with_20_later_bars_as_first_signals_not_held(signal_maker, tmp_path):
    db = tmp_path / "smm.db"
    PortfolioStore(db).set("AAPL", 50, 90.0)
    seed_signals(db, [None])
    history = flat_history(rows=100)
    fetcher = FakeFetcher(history=history)
    classifier = FakeClassifier(_signal("hold"))

    reply = signal_maker(fetcher, classifier, db_path=db, now=lambda: NOW).backfill("aapl", "1d", SINCE)

    # Bars 60 to 79: bar 79 is the last with 20 bars after it.
    assert reply == {"logged": 20, "remaining": 0}
    states = [json.loads(prompt) for prompt in classifier.prompts]
    assert [state["rows"][-1]["Date"] for state in states] == [t.isoformat() for t in history.index[60:80]]
    assert all(state["position"] is None and "previous_signal" not in state for state in states)
    assert all(list(questions["verdict"]["criteria"]) == ["buy", "dip", "hold"] for questions in classifier.questions)
    # Fetched from a year before since, so the first backfilled bar has the 1d default period behind it.
    assert fetcher.history_calls == [("AAPL", "2y", "1d", False)]


def test_backfilled_bar_sees_what_a_live_signal_on_that_bar_would(signal_maker, tmp_path):
    # Varying bars, and wild ones after bar 79, so any indicator that looks ahead changes the state.
    history = flat_history(rows=100)
    history["Close"] = [100.0 + (i * 7) % 11 for i in range(100)]
    history.loc[history.index[80:], ["High", "Close"]] = [300.0, 250.0]
    backfilled = FakeClassifier(_signal("hold"))
    live = FakeClassifier(_signal("hold"))

    _maker(signal_maker, tmp_path, history, backfilled).backfill("AAPL", "1d", SINCE)
    signal_maker(FakeFetcher(history=history.iloc[:80]), live, db_path=tmp_path / "live.db").make("AAPL", "1d")

    assert backfilled.prompts[-1] == live.prompts[0]
    assert backfilled.questions[-1] == live.questions[0]


def _rising_later():
    """Flat bars of 100 to bar 60 (SINCE), quieter ones to bar 79, then 20 reaching 110, none under 99.5 after bar 60.

    Each bar backfilled (60 to 79) reaches its buy's default Target (108 at most) or its dip's picked Target (the 20-bar
    high, 101, from bar 60) before either Stop (the dip's is bar 60's low, 99)."""
    history = flat_history(rows=100)
    history.loc[history.index[61:80], ["High", "Low"]] = [100.5, 99.5]
    history.loc[history.index[80:], ["High", "Low"]] = [110.0, 99.5]
    return history


DIP_PICKS = {"dip_target": {"choice": "high_20", "confidence": 0.8}, "dip_stop": {"choice": "low_20", "confidence": 0.6}}


@pytest.mark.parametrize(
    ("signal", "min_confidence", "seed"),
    [
        (_signal("buy", 0.5), "min_confidence", {}),
        ({**_signal("dip", 0.5), "picks": DIP_PICKS}, "dip_min_confidence", {"verdict": "dip", "long": (99.0, 101.0)}),
    ],
)
def test_backfilled_signals_are_scored_in_the_call_and_count_toward_the_minimum_confidence(signal_maker, tmp_path, signal, min_confidence, seed):
    # 10 Targets short of the 30 a band needs.
    seed_signals(tmp_path / "smm.db", ["target"] * 10, confidence=0.5, **seed)

    reply = _maker(signal_maker, tmp_path, _rising_later(), FakeClassifier(signal)).backfill("AAPL", "1d", SINCE)
    # A history without the backfilled bars, so only backfill could have scored them.
    live = _maker(signal_maker, tmp_path, flat_history(), FakeClassifier(signal)).make("AAPL", "1d")

    assert reply == {"logged": 20, "remaining": 0}
    assert live["hit_rate"] == 1.0
    # The 0.5 band now has 30 Targets, above break-even; the default is 0.4.
    assert live[min_confidence] == 0.5


class _FailingOn:
    """FakeClassifier that raises on the call given (1-based)."""

    def __init__(self, call, replies):
        self.fake = FakeClassifier(replies=replies)
        self.call = call

    def classify(self, prompt, questions):
        if len(self.fake.prompts) + 1 == self.call:
            raise RuntimeError("openrouter 500")
        return self.fake.classify(prompt, questions)


def test_failing_classifier_keeps_and_scores_earlier_bars_and_logs_nothing_for_the_failing_one(signal_maker, tmp_path):
    # 8 Targets, so the 2 bars before the failing one make the 10 a Hit Rate needs.
    seed_signals(tmp_path / "smm.db", ["target"] * 8)
    failing = _FailingOn(3, [_signal("buy", 0.51), _signal("buy", 0.52)])

    with pytest.raises(RuntimeError, match="openrouter 500"):
        _maker(signal_maker, tmp_path, _rising_later(), failing).backfill("AAPL", "1d", SINCE)
    later = FakeClassifier(_signal("buy"))
    # A history without the backfilled bars, so only backfill could have scored them.
    live = signal_maker(FakeFetcher(history=flat_history()), later, db_path=tmp_path / "smm.db", ttl=0).make("AAPL", "1d")

    # The last row logged is the second bar's.
    assert json.loads(later.prompts[0])["previous_signal"]["confidence"] == 0.52
    assert live["hit_rate"] == 1.0


def test_bar_with_a_non_finite_close_is_skipped_without_a_classifier_call(signal_maker, tmp_path):
    history = flat_history(rows=100)
    history.loc[history.index[65], "Close"] = float("nan")
    classifier = FakeClassifier(_signal("hold"))

    reply = _maker(signal_maker, tmp_path, history, classifier).backfill("AAPL", "1d", SINCE)

    assert reply == {"logged": 19, "remaining": 0}
    assert _last_dates(classifier) == [t.isoformat() for t in history.index[60:80] if t != history.index[65]]


def test_1m_bar_sees_the_5_sessions_a_live_signal_fetches_not_5_calendar_days(signal_maker, tmp_path):
    # 10 bars a session, Monday 2024-03-04 to Thursday 2024-03-14, a weekend between.
    days = pd.bdate_range("2024-03-04", "2024-03-14")
    idx = pd.DatetimeIndex([t for day in days for t in pd.date_range(day + pd.Timedelta("9h30min"), periods=10, freq="min")], name="Datetime")
    history = flat_history(rows=len(idx)).set_axis(idx)
    classifier = FakeClassifier(_signal("hold"))

    _maker(signal_maker, tmp_path, history, classifier).backfill("AAPL", "1m", "2024-03-12")

    # The first bar of 2024-03-12 sees Wednesday 2024-03-06 on: five sessions, not back to Thursday as five days would.
    assert json.loads(classifier.prompts[0])["rows"][0]["Date"] == "2024-03-06T09:30:00"


@pytest.mark.parametrize("since", ["2024-13-01", "last year", "", "2024-12-02"])
def test_unparsable_or_future_since_raises_before_any_classifier_call(signal_maker, tmp_path, since):
    classifier = FakeClassifier(_signal())

    with pytest.raises(InputError, match="since"):
        _maker(signal_maker, tmp_path, flat_history(rows=100), classifier).backfill("AAPL", "1d", since)

    assert classifier.prompts == []


def _last_dates(classifier):
    return [json.loads(prompt)["rows"][-1]["Date"] for prompt in classifier.prompts]


def test_each_call_logs_the_50_oldest_eligible_bars_and_the_next_call_continues(signal_maker, tmp_path):
    # Bars 60 to 179 are eligible: 120 bars.
    history = flat_history(rows=200)
    replies, dates = [], []
    for _ in range(4):
        classifier = FakeClassifier(_signal("hold"))
        replies.append(_maker(signal_maker, tmp_path, history, classifier).backfill("AAPL", "1d", SINCE))
        dates.append(_last_dates(classifier))

    assert replies == [{"logged": 50, "remaining": 70}, {"logged": 50, "remaining": 20}, {"logged": 20, "remaining": 0}, {"logged": 0, "remaining": 0}]
    assert dates[:3] == [[t.isoformat() for t in history.index[start:end]] for start, end in ((60, 110), (110, 160), (160, 180))]
    assert dates[3] == []


def test_bar_with_a_live_signal_is_skipped_without_a_classifier_call(signal_maker, tmp_path):
    history = flat_history(rows=100)
    signal_maker(FakeFetcher(history=history.iloc[:66]), FakeClassifier(_signal("hold")), db_path=tmp_path / "smm.db").make("AAPL", "1d")
    classifier = FakeClassifier(_signal("hold"))

    reply = _maker(signal_maker, tmp_path, history, classifier).backfill("AAPL", "1d", SINCE)

    assert reply == {"logged": 19, "remaining": 0}
    assert _last_dates(classifier) == [t.isoformat() for t in history.index[60:80] if t != history.index[65]]
