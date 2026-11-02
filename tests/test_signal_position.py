"""Tests for the Position on a Signal (#59): SignalMaker adds the held Position and its Unrealised P&L at the Signal's Price."""

import json
import sqlite3
from contextlib import closing

import pytest
from fakes import FakeClassifier, FakeFetcher, flat_history, seed_signals

from stock_market_mcp.portfolio import PortfolioStore


def _signal(verdict="sell", confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


def _maker(signal_maker, tmp_path, classifier):
    """SignalMaker over flat close 100 on tmp_path / "smm.db"; its fetcher has no quotes, so quoting fails loudly."""
    return signal_maker(FakeFetcher(history=flat_history()), classifier, db_path=tmp_path / "smm.db")


def set_position(tmp_path, ticker, direction="long", quantity: float = 10, avg_cost: float = 90.0):
    PortfolioStore(tmp_path / "smm.db").set(ticker, direction, quantity, avg_cost)


def test_held_position_rides_on_the_signal_with_pnl_at_its_price(signal_maker, tmp_path):
    # Flat close 100, Position 10 @ 90: up 100 on a cost of 900.
    maker = _maker(signal_maker, tmp_path, FakeClassifier(_signal()))
    set_position(tmp_path, "AAPL")
    payload = maker.make("aapl")

    assert payload["price"] == 100.0
    assert payload["position"] == {
        "direction": "long",
        "quantity": 10,
        "avg_cost": 90.0,
        "unrealised_pnl": pytest.approx(100.0),
        "unrealised_pnl_pct": pytest.approx(100 / 9),
        "cut_loss": False,
        "cut_loss_hit_rate": None,
    }


def test_ticker_not_held_has_no_position_key(signal_maker, tmp_path):
    maker = _maker(signal_maker, tmp_path, FakeClassifier(_signal()))
    set_position(tmp_path, "MSFT")
    payload = maker.make("aapl")

    assert "position" not in payload
    assert payload["verdict"] == "sell"


def test_cached_signal_shows_the_current_position(signal_maker, tmp_path):
    classifier = FakeClassifier(_signal())
    maker = _maker(signal_maker, tmp_path, classifier)
    set_position(tmp_path, "AAPL")

    first = maker.make("aapl")
    set_position(tmp_path, "AAPL", quantity=4)
    second = maker.make("aapl")

    assert first["position"]["quantity"] == 10
    assert second["position"] == {
        "direction": "long",
        "quantity": 4,
        "avg_cost": 90.0,
        "unrealised_pnl": pytest.approx(40.0),
        "unrealised_pnl_pct": pytest.approx(40 / 360 * 100),
        "cut_loss": False,
        "cut_loss_hit_rate": None,
    }
    assert len(classifier.prompts) == 1


@pytest.mark.parametrize("change", [{"avg_cost": 95.0}, {"direction": "short"}, "deleted"])
def test_changing_the_position_beyond_quantity_reclassifies(signal_maker, tmp_path, change):
    classifier = FakeClassifier(_signal())
    maker = _maker(signal_maker, tmp_path, classifier)
    set_position(tmp_path, "AAPL")

    maker.make("aapl")
    if change == "deleted":
        PortfolioStore(tmp_path / "smm.db").delete("AAPL")
    else:
        set_position(tmp_path, "AAPL", **change)
    second = maker.make("aapl")

    assert len(classifier.prompts) == 2
    if change == "deleted":
        assert "position" not in second


def _prompt(classifier, n=-1) -> dict:
    return json.loads(classifier.prompts[n])


def test_prompt_carries_the_held_position_without_quantity(signal_maker, tmp_path):
    classifier = FakeClassifier(_signal())
    maker = _maker(signal_maker, tmp_path, classifier)
    set_position(tmp_path, "AAPL")
    maker.make("aapl")

    assert _prompt(classifier)["position"] == {
        "direction": "long",
        "avg_cost": 90.0,
        "unrealised_pnl_pct": pytest.approx(100 / 9),
    }


def test_prompt_says_no_position_for_an_unheld_ticker(signal_maker, tmp_path):
    classifier = FakeClassifier(_signal())
    maker = _maker(signal_maker, tmp_path, classifier)
    set_position(tmp_path, "MSFT")
    maker.make("aapl")

    prompt = _prompt(classifier)
    assert "position" in prompt
    assert prompt["position"] is None


def test_position_survives_row_trimming(signal_maker, tmp_path, monkeypatch):
    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "3000")
    classifier = FakeClassifier(_signal())
    maker = _maker(signal_maker, tmp_path, classifier)
    set_position(tmp_path, "AAPL", "short")
    maker.make("aapl")

    prompt = _prompt(classifier)
    assert len(prompt["rows"]) < 60  # rows were dropped to fit
    assert prompt["position"]["direction"] == "short"


def _cut_loss_reply(signal_maker, tmp_path, signal, direction="long", avg_cost=110.0, seed=False) -> dict:
    """SignalMaker reply for a held AAPL at flat close 100; seed=True makes the Minimum Confidence null."""
    db = tmp_path / "smm.db"
    maker = _maker(signal_maker, tmp_path, FakeClassifier(signal))
    if seed:  # 30 scored in the 0.9 band at under break-even, so no band qualifies
        seed_signals(db, ["target"] * 10 + ["stop"] * 20, confidence=0.9)
    set_position(tmp_path, "AAPL", direction, avg_cost=avg_cost)
    return maker.make("aapl")


def test_opposing_verdict_on_a_losing_position_is_a_cut_loss(signal_maker, tmp_path):
    # No scored Signals, so the Minimum Confidence is the 0.6 default; 0.6 itself counts.
    payload = _cut_loss_reply(signal_maker, tmp_path, _signal("sell", 0.6))

    assert payload["min_confidence"] == 0.6
    assert payload["position"]["cut_loss"] is True


def test_below_the_minimum_confidence_is_not_a_cut_loss(signal_maker, tmp_path):
    # Which Verdicts oppose a losing Position is tested on Position; this is the Confidence gate.
    payload = _cut_loss_reply(signal_maker, tmp_path, _signal("sell", 0.59))

    assert payload["position"]["cut_loss"] is False


def test_null_minimum_confidence_is_never_a_cut_loss(signal_maker, tmp_path):
    payload = _cut_loss_reply(signal_maker, tmp_path, _signal("sell", 1.0), seed=True)

    assert payload["min_confidence"] is None
    assert payload["position"]["cut_loss"] is False


def test_cut_loss_sell_keeps_its_short_trade_plan(signal_maker, tmp_path):
    payload = _cut_loss_reply(signal_maker, tmp_path, _signal("sell", 0.7))

    assert payload["position"]["cut_loss"] is True
    assert payload["trade_plan"]["direction"] == "short"
    assert payload["trade_plan"]["stop"] > payload["trade_plan"]["entry"] > payload["trade_plan"]["target"]



def _stored_cut_loss(db) -> list:
    # The stored flag never reaches the reply except through the Cut Loss Hit Rate, so these tests read the table.
    with closing(sqlite3.connect(db)) as conn:
        return [r[0] for r in conn.execute("SELECT cut_loss FROM signals ORDER BY id")]


@pytest.mark.parametrize(
    ("held", "verdict", "confidence", "avg_cost", "stored"),
    [
        ("AAPL", "sell", 0.3, 110.0, 1),  # below the Minimum Confidence still counts
        ("AAPL", "sell", 0.7, 90.0, 0),  # profitable
        ("MSFT", "sell", 0.7, 110.0, None),  # AAPL not held
    ],
)
def test_fresh_signal_stores_cut_loss(signal_maker, tmp_path, held, verdict, confidence, avg_cost, stored):
    db = tmp_path / "smm.db"
    maker = _maker(signal_maker, tmp_path, FakeClassifier(_signal(verdict, confidence)))
    set_position(tmp_path, held, avg_cost=avg_cost)
    maker.make("aapl")

    assert _stored_cut_loss(db) == [stored]


def test_old_signals_schema_is_upgraded_on_open(signal_maker, tmp_path):
    # A file written before cut_loss existed; built by hand since no module writes the old schema.
    db = tmp_path / "smm.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "CREATE TABLE signals (id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, interval TEXT NOT NULL, "
            "prompt_version INTEGER NOT NULL, data_date TEXT NOT NULL, verdict TEXT NOT NULL, "
            "confidence REAL NOT NULL, price REAL NOT NULL, long_stop REAL, long_target REAL, short_stop REAL, "
            "short_target REAL, outcome TEXT, scored_at TEXT)"
        )
        conn.execute(
            "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, outcome) "
            "VALUES ('MSFT', '1d', 1, '2023-01-01T00:00:00', 'buy', 0.8, 50.0, 'target')"
        )
    maker = _maker(signal_maker, tmp_path, FakeClassifier(_signal("sell", 0.7)))
    set_position(tmp_path, "AAPL", avg_cost=110.0)
    payload = maker.make("aapl")

    assert payload["position"]["cut_loss"] is True
    assert _stored_cut_loss(db) == [None, 1]
    with closing(sqlite3.connect(db)) as conn:
        old = conn.execute("SELECT ticker, prompt_version, verdict, confidence, outcome FROM signals WHERE id = 1")
        assert old.fetchone() == ("MSFT", 1, "buy", 0.8, "target")


def _cut_loss_hit_rate(signal_maker, tmp_path, seed) -> float | None:
    db = tmp_path / "smm.db"
    seed(db)
    maker = _maker(signal_maker, tmp_path, FakeClassifier(_signal("buy")))
    set_position(tmp_path, "AAPL")
    return maker.make("aapl")["position"]["cut_loss_hit_rate"]


def test_cut_loss_hit_rate_is_null_under_30_scored(signal_maker, tmp_path):
    def seed(db):
        seed_signals(db, ["target"] * 29 + ["expired"] * 5, cut_loss=1)

    assert _cut_loss_hit_rate(signal_maker, tmp_path, seed) is None


def test_cut_loss_hit_rate_pools_scored_cut_loss_signals_of_this_prompt_version(signal_maker, tmp_path):
    def seed(db):
        seed_signals(db, ["target"] * 20, cut_loss=1, ticker="MSFT", interval="1d")
        seed_signals(db, ["stop"] * 5, cut_loss=1, ticker="NVDA", interval="5m", verdict="sell")
        seed_signals(db, ["stop"] * 5, cut_loss=1, confidence=0.2)  # below the Minimum Confidence
        seed_signals(db, ["expired"] * 10, cut_loss=1)
        seed_signals(db, ["stop"] * 30, cut_loss=0)
        seed_signals(db, ["stop"] * 30)  # not held
        seed_signals(db, ["stop"] * 30, cut_loss=1, prompt_version=1)

    assert _cut_loss_hit_rate(signal_maker, tmp_path, seed) == pytest.approx(20 / 30)
