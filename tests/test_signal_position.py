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
    await set_position(url, "AAPL", quantity=4)
    second = await get_signal(url, fetcher)

    assert first["position"]["quantity"] == 10
    assert second["position"] == {
        "direction": "long",
        "quantity": 4,
        "avg_cost": 90.0,
        "unrealised_pnl": pytest.approx(40.0),
        "unrealised_pnl_pct": pytest.approx(40 / 360 * 100),
    }
    assert len(classifier.prompts) == 1


@pytest.mark.parametrize("change", [{"avg_cost": 95.0}, {"direction": "short"}, {"quantity": 0}])
async def test_changing_the_position_beyond_quantity_reclassifies(server_url, tmp_path, change):
    fetcher, classifier = QuotedFlatFetcher(), FakeClassifier(_signal())
    url = server_url(fetcher, classifier=classifier, db_path=tmp_path / "smm.db")
    await set_position(url, "AAPL")

    await get_signal(url, fetcher)
    await set_position(url, "AAPL", **change)
    await get_signal(url, fetcher)

    assert len(classifier.prompts) == 2


def _prompt(classifier, n=-1) -> dict:
    return json.loads(classifier.prompts[n])


@pytest.mark.parametrize(("direction", "pct"), [("long", 100 / 900 * 100), ("short", -100 / 900 * 100)])
async def test_prompt_carries_the_held_position_without_quantity(server_url, tmp_path, direction, pct):
    fetcher, classifier = QuotedFlatFetcher(), FakeClassifier(_signal())
    url = server_url(fetcher, classifier=classifier, db_path=tmp_path / "smm.db")
    await set_position(url, "AAPL", direction)
    await get_signal(url, fetcher)

    assert _prompt(classifier)["position"] == {
        "direction": direction,
        "avg_cost": 90.0,
        "unrealised_pnl_pct": pytest.approx(pct),
    }


async def test_prompt_says_no_position_for_an_unheld_ticker(server_url, tmp_path):
    fetcher, classifier = QuotedFlatFetcher(), FakeClassifier(_signal())
    url = server_url(fetcher, classifier=classifier, db_path=tmp_path / "smm.db")
    await set_position(url, "MSFT")
    await get_signal(url, fetcher)

    prompt = _prompt(classifier)
    assert "position" in prompt
    assert prompt["position"] is None


async def test_position_survives_row_trimming(server_url, tmp_path, monkeypatch):
    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "3000")
    fetcher, classifier = QuotedFlatFetcher(), FakeClassifier(_signal())
    url = server_url(fetcher, classifier=classifier, db_path=tmp_path / "smm.db")
    await set_position(url, "AAPL", "short")
    await get_signal(url, fetcher)

    prompt = _prompt(classifier)
    assert len(prompt["rows"]) < 60  # rows were dropped to fit
    assert prompt["position"]["direction"] == "short"
