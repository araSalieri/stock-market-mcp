"""Tests for the Position on get_signal (#59): the held Position and its Unrealised P&L at the Signal's Price, over MCP."""

import json

import pytest
from fakes import FakeClassifier, FlatFakeFetcher, call_tool, text_of

from stock_market_mcp.portfolio import PortfolioStore


def _signal(verdict="sell", confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


async def get_signal(url) -> dict:
    result = await call_tool(url, "get_signal", {"ticker": "aapl"})
    assert not result.is_error, text_of(result)
    return json.loads(text_of(result))


@pytest.mark.parametrize(
    ("direction", "pnl", "pct"),
    [("long", 100.0, 100 / 900 * 100), ("short", -100.0, -100 / 900 * 100)],
)
async def test_held_position_rides_on_the_signal_with_pnl_at_its_price(server_url, tmp_path, direction, pnl, pct):
    # Flat close 100, Position 10 @ 90: a long is up 100, a short down 100, on a cost of 900.
    db = tmp_path / "smm.db"
    PortfolioStore(db).set("AAPL", direction, 10, 90.0)
    payload = await get_signal(server_url(FlatFakeFetcher(), classifier=FakeClassifier(_signal()), db_path=db))

    assert payload["price"] == 100.0
    assert payload["position"] == {
        "direction": direction,
        "quantity": 10,
        "avg_cost": 90.0,
        "unrealised_pnl": pytest.approx(pnl),
        "unrealised_pnl_pct": pytest.approx(pct),
    }


async def test_ticker_not_held_has_no_position_key(server_url, tmp_path):
    db = tmp_path / "smm.db"
    PortfolioStore(db).set("MSFT", "long", 10, 90.0)
    payload = await get_signal(server_url(FlatFakeFetcher(), classifier=FakeClassifier(_signal()), db_path=db))

    assert "position" not in payload
    assert payload["verdict"] == "sell"


async def test_cached_signal_shows_the_current_position(server_url, tmp_path):
    db = tmp_path / "smm.db"
    store = PortfolioStore(db)
    store.set("AAPL", "long", 10, 90.0)
    classifier = FakeClassifier(_signal())
    url = server_url(FlatFakeFetcher(), classifier=classifier, db_path=db)

    first = await get_signal(url)
    store.set("AAPL", "short", 4, 110.0)
    second = await get_signal(url)
    store.delete("AAPL")
    third = await get_signal(url)

    assert first["position"]["direction"] == "long"
    assert second["position"] == {
        "direction": "short",
        "quantity": 4,
        "avg_cost": 110.0,
        "unrealised_pnl": pytest.approx(40.0),
        "unrealised_pnl_pct": pytest.approx(40 / 440 * 100),
    }
    assert "position" not in third
    assert len(classifier.prompts) == 1


async def test_prompt_stored_signal_and_scoring_ignore_the_position(server_url, tmp_path):
    # TTL 0: the second call re-classifies, and its prompt carries the stored previous Signal.
    def run(name, held):
        db = tmp_path / f"{name}.db"
        if held:
            PortfolioStore(db).set("AAPL", "long", 10, 90.0)
        classifier = FakeClassifier(_signal())
        return classifier, server_url(FlatFakeFetcher(), classifier=classifier, db_path=db, signal_ttl=0)

    held_classifier, held_url = run("held", True)
    free_classifier, free_url = run("free", False)
    held = [await get_signal(held_url) for _ in range(2)]
    free = [await get_signal(free_url) for _ in range(2)]

    assert held_classifier.prompts[0] != held_classifier.prompts[1]  # the second carries the stored Signal
    assert held_classifier.prompts == free_classifier.prompts
    assert [{k: v for k, v in reply.items() if k != "position"} for reply in held] == free
