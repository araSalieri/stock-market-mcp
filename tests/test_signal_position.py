"""Tests for the Position on get_signal (#59): the held Position and its Unrealised P&L at the Signal's Price, over MCP."""

import json

import pytest
from fakes import FakeClassifier, FlatFakeFetcher, call_tool, text_of


class QuotedFlatFetcher(FlatFakeFetcher):
    """Flat history plus a quote, so set_position can check the ticker; get_signal must still fetch none."""

    def __init__(self):
        super().__init__()
        self.quote_calls = []

    def fetch_quote(self, ticker):
        self.quote_calls.append(ticker)
        return {"price": 100.0, "currency": "USD"}


def _signal(verdict="sell", confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


async def get_signal(url, fetcher: QuotedFlatFetcher) -> dict:
    quotes_before = len(fetcher.quote_calls)
    result = await call_tool(url, "get_signal", {"ticker": "aapl"})
    assert not result.is_error, text_of(result)
    assert len(fetcher.quote_calls) == quotes_before
    return json.loads(text_of(result))


async def set_position(url, ticker, direction="long", quantity: float = 10, avg_cost: float = 90.0):
    args = {"ticker": ticker, "direction": direction, "quantity": quantity, "avg_cost": avg_cost}
    result = await call_tool(url, "set_position", args)
    assert not result.is_error, text_of(result)


@pytest.mark.parametrize(
    ("direction", "pnl", "pct"),
    [("long", 100.0, 100 / 900 * 100), ("short", -100.0, -100 / 900 * 100)],
)
async def test_held_position_rides_on_the_signal_with_pnl_at_its_price(server_url, tmp_path, direction, pnl, pct):
    # Flat close 100, Position 10 @ 90: a long is up 100, a short down 100, on a cost of 900.
    fetcher = QuotedFlatFetcher()
    url = server_url(fetcher, classifier=FakeClassifier(_signal()), db_path=tmp_path / "smm.db")
    await set_position(url, "AAPL", direction)
    payload = await get_signal(url, fetcher)

    assert payload["price"] == 100.0
    assert payload["position"] == {
        "direction": direction,
        "quantity": 10,
        "avg_cost": 90.0,
        "unrealised_pnl": pytest.approx(pnl),
        "unrealised_pnl_pct": pytest.approx(pct),
    }


async def test_ticker_not_held_has_no_position_key(server_url, tmp_path):
    fetcher = QuotedFlatFetcher()
    url = server_url(fetcher, classifier=FakeClassifier(_signal()), db_path=tmp_path / "smm.db")
    await set_position(url, "MSFT")
    payload = await get_signal(url, fetcher)

    assert "position" not in payload
    assert payload["verdict"] == "sell"


async def test_cached_signal_shows_the_current_position(server_url, tmp_path):
    fetcher = QuotedFlatFetcher()
    classifier = FakeClassifier(_signal())
    url = server_url(fetcher, classifier=classifier, db_path=tmp_path / "smm.db")
    await set_position(url, "AAPL")

    first = await get_signal(url, fetcher)
    await set_position(url, "AAPL", "short", 4, 110.0)
    second = await get_signal(url, fetcher)
    await set_position(url, "AAPL", quantity=0)
    third = await get_signal(url, fetcher)

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
    async def run(name, held):
        fetcher, classifier = QuotedFlatFetcher(), FakeClassifier(_signal())
        url = server_url(fetcher, classifier=classifier, db_path=tmp_path / f"{name}.db", signal_ttl=0)
        if held:
            await set_position(url, "AAPL")
        return classifier, [await get_signal(url, fetcher) for _ in range(2)]

    held_classifier, held = await run("held", True)
    free_classifier, free = await run("free", False)

    assert held_classifier.prompts[0] != held_classifier.prompts[1]  # the second carries the stored Signal
    assert held_classifier.prompts == free_classifier.prompts
    assert [{k: v for k, v in reply.items() if k != "position"} for reply in held] == free
