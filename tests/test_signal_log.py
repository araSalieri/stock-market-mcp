"""Tests for the Signal log (#52): SignalMaker stores each fresh Signal and shows the classifier its previous one."""

import json
import sqlite3
from contextlib import closing

import pytest
from fakes import FakeClassifier, FakeFetcher, flat_history

from stock_market_mcp.errors import InputError


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


def test_second_call_shows_the_classifier_the_first_signal_as_pending(signal_maker):
    classifier = FakeClassifier(replies=[_signal("sell", 0.8), _signal("buy")])
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, ttl=0)

    maker.make("AAPL")
    maker.make("aapl")

    first, second = (json.loads(p) for p in classifier.prompts)
    assert "previous_signal" not in first
    assert second["previous_signal"] == {"verdict": "sell", "confidence": 0.8, "outcome": "pending"}


def _stored(db):
    # Row counts and plan prices never reach the reply or the classifier, so these tests read the table.
    if not db.exists():
        return []
    with closing(sqlite3.connect(db)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM signals ORDER BY id")]


def test_cache_hit_within_the_ttl_stores_nothing(signal_maker, tmp_path):
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, db_path=tmp_path / "s.db")

    maker.make("AAPL")
    maker.make("AAPL")

    assert len(classifier.prompts) == 1  # the stored Signal doesn't change the cache key
    assert len(_stored(tmp_path / "s.db")) == 1


def _previous_after(signal_maker, db, failing):
    """The previous Signal a working SignalMaker on db shows the classifier after the failing one raised."""
    with pytest.raises(Exception):  # any failure will do; the point is what is stored
        failing.make("AAPL")
    classifier = FakeClassifier()
    signal_maker(FakeFetcher(history=flat_history()), classifier, db_path=db).make("AAPL")
    return json.loads(classifier.prompts[0]).get("previous_signal")


def test_failed_classifier_call_stores_nothing(signal_maker, tmp_path):
    db = tmp_path / "s.db"
    failing = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(error=RuntimeError("down")), db_path=db)
    assert _previous_after(signal_maker, db, failing) is None


def test_failed_fetch_stores_nothing(signal_maker, tmp_path):
    db = tmp_path / "s.db"
    failing = signal_maker(FakeFetcher(history=flat_history(rows=5)), FakeClassifier(), db_path=db)
    assert _previous_after(signal_maker, db, failing) is None


def test_hold_is_stored_with_both_default_plans_whatever_the_caller_asked(signal_maker, tmp_path):
    # Entry 100, ATR 2, defaults 2 ATR and ratio 2: long 96/108, short 104/92.
    maker = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(_signal("hold", 0.4)), db_path=tmp_path / "s.db")

    maker.make("aapl", "1d", atr_multiple=1, reward_ratio=3)

    [row] = _stored(tmp_path / "s.db")
    assert row == {
        "id": 1,
        "ticker": "AAPL",
        "interval": "1d",
        "prompt_version": 2,
        "data_date": "2024-02-29T00:00:00",
        "verdict": "hold",
        "confidence": 0.4,
        "price": 100.0,
        "long_stop": pytest.approx(96.0),
        "long_target": pytest.approx(108.0),
        "short_stop": pytest.approx(104.0),
        "short_target": pytest.approx(92.0),
        "outcome": None,
        "scored_at": None,
        "cut_loss": None,
    }


def test_previous_signal_survives_row_trimming(signal_maker, monkeypatch):
    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "3000")
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, ttl=0)

    maker.make("AAPL")
    maker.make("AAPL")

    second = json.loads(classifier.prompts[1])
    assert len(second["rows"]) < 60  # rows were dropped to fit
    assert second["previous_signal"]["verdict"] == "buy"


def test_unusable_database_fails_before_the_classifier(signal_maker, tmp_path):
    # A directory can't be opened as a SQLite file; the Portfolio read is the first to try.
    classifier = FakeClassifier()
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, db_path=tmp_path)

    with pytest.raises(ValueError, match=str(tmp_path)):
        maker.make("AAPL")

    assert classifier.prompts == []


def test_invalid_caller_plan_fails_before_the_signal_is_stored(signal_maker):
    # Close 10, ATR 2: a 5 ATR Stop is 0, but the default 2 ATR plan is valid.
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history(close=10.0)), classifier)

    with pytest.raises(InputError):
        maker.make("AAPL", "1d", atr_multiple=5)
    maker.make("AAPL", "1d")

    assert "previous_signal" not in json.loads(classifier.prompts[1])  # not stored, and not cached either


def test_failed_store_is_not_cached_so_a_retry_stores_the_signal(signal_maker, tmp_path):
    db = tmp_path / "s.db"
    classifier = FakeClassifier(_signal("buy"))
    maker = signal_maker(FakeFetcher(history=flat_history()), classifier, db_path=db)
    maker.make("MSFT")  # creates the file
    db.chmod(0o444)  # reads still work, the insert fails

    with pytest.raises(ValueError):
        maker.make("AAPL")
    db.chmod(0o644)
    maker.make("AAPL")

    assert len(classifier.prompts) == 3  # the retry asked the classifier again instead of a cache hit
