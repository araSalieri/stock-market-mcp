"""Tests for the Signal log (#52): get_signal stores each fresh Signal and shows the classifier its previous one."""

import json
import sqlite3
from contextlib import closing

import pytest
from fakes import FakeClassifier, FlatFakeFetcher, call_tool, text_of


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


async def test_second_call_shows_the_classifier_the_first_signal_as_pending(server_url):
    classifier = FakeClassifier(replies=[_signal("sell", 0.8), _signal("buy")])
    url = server_url(FlatFakeFetcher(), classifier=classifier, signal_ttl=0)

    await call_tool(url, "get_signal", {"ticker": "AAPL"})
    await call_tool(url, "get_signal", {"ticker": "aapl"})

    first, second = (json.loads(p) for p in classifier.prompts)
    assert "previous_signal" not in first
    assert second["previous_signal"] == {"verdict": "sell", "confidence": 0.8, "outcome": "pending"}


def _stored(db):
    # The reply doesn't show stored rows until scoring (#53), so these tests read the table.
    if not db.exists():
        return []
    with closing(sqlite3.connect(db)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM signals ORDER BY id")]


async def test_cache_hit_within_the_ttl_stores_nothing(server_url, tmp_path):
    classifier = FakeClassifier(_signal("buy"))
    url = server_url(FlatFakeFetcher(), classifier=classifier, db_path=tmp_path / "s.db")

    await call_tool(url, "get_signal", {"ticker": "AAPL"})
    second = await call_tool(url, "get_signal", {"ticker": "AAPL"})

    assert not second.is_error
    assert len(classifier.prompts) == 1  # the stored Signal doesn't change the cache key
    assert len(_stored(tmp_path / "s.db")) == 1


async def test_failed_classifier_call_stores_nothing(server_url, tmp_path):
    classifier = FakeClassifier(error=RuntimeError("down"))
    url = server_url(FlatFakeFetcher(), classifier=classifier, db_path=tmp_path / "s.db")

    result = await call_tool(url, "get_signal", {"ticker": "AAPL"})

    assert result.is_error
    assert _stored(tmp_path / "s.db") == []


async def test_failed_fetch_stores_nothing(server_url, tmp_path):
    url = server_url(FlatFakeFetcher(rows=5), classifier=FakeClassifier(), db_path=tmp_path / "s.db")

    result = await call_tool(url, "get_signal", {"ticker": "AAPL"})

    assert result.is_error
    assert _stored(tmp_path / "s.db") == []


async def test_hold_is_stored_with_both_default_plans_whatever_the_caller_asked(server_url, tmp_path):
    # Entry 100, ATR 2, defaults 2 ATR and ratio 2: long 96/108, short 104/92.
    classifier = FakeClassifier(_signal("hold", 0.4))
    url = server_url(FlatFakeFetcher(), classifier=classifier, db_path=tmp_path / "s.db")

    await call_tool(url, "get_signal", {"ticker": "aapl", "interval": "1d", "atr_multiple": 1, "reward_ratio": 3})

    [row] = _stored(tmp_path / "s.db")
    assert row == {
        "id": 1,
        "ticker": "AAPL",
        "interval": "1d",
        "prompt_version": 1,
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
    }


async def test_previous_signal_survives_row_trimming(server_url, monkeypatch):
    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "3000")
    classifier = FakeClassifier(_signal("buy"))
    url = server_url(FlatFakeFetcher(), classifier=classifier, signal_ttl=0)

    await call_tool(url, "get_signal", {"ticker": "AAPL"})
    await call_tool(url, "get_signal", {"ticker": "AAPL"})

    second = json.loads(classifier.prompts[1])
    assert len(second["rows"]) < 60  # rows were dropped to fit
    assert second["previous_signal"]["verdict"] == "buy"


async def test_unusable_database_fails_get_signal_loudly(server_url, tmp_path):
    # A directory can't be opened as a SQLite file.
    classifier = FakeClassifier()
    url = server_url(FlatFakeFetcher(), classifier=classifier, db_path=tmp_path)

    result = await call_tool(url, "get_signal", {"ticker": "AAPL"})

    assert result.is_error
    assert str(tmp_path) in text_of(result)
    assert classifier.prompts == []
