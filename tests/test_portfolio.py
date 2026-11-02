"""Tests for the Portfolio (#58): the store directly, then set_position and get_portfolio over MCP."""

import json
import re

import pytest

from fakes import FakeFetcher, call_tool, text_of

from stock_market_mcp.portfolio import PortfolioStore, Position

# Store tests call PortfolioStore directly: the spec (#57) puts a storage seam under the tools, and an
# empty or broken file must fail the same way for every caller (get_signal reads it too in #59).


def test_store_set_then_load_round_trips(tmp_path):
    store = PortfolioStore(tmp_path / "smm.db")
    store.set("AAPL", "long", 10.5, 180.0)
    store.set("BBCA.JK", "short", 100, 9000)

    assert store.load() == {
        "AAPL": Position("long", 10.5, 180.0),
        "BBCA.JK": Position("short", 100, 9000),
    }


def test_store_set_replaces_the_tickers_position(tmp_path):
    store = PortfolioStore(tmp_path / "smm.db")
    store.set("AAPL", "long", 10, 180.0)
    store.set("AAPL", "short", 3, 200.0)

    assert store.load() == {"AAPL": Position("short", 3, 200.0)}


def test_store_delete_removes_one_and_missing_is_fine(tmp_path):
    store = PortfolioStore(tmp_path / "smm.db")
    store.set("AAPL", "long", 10, 180.0)
    store.set("MSFT", "long", 1, 400.0)

    store.delete("AAPL")
    store.delete("NVDA")

    assert store.load() == {"MSFT": Position("long", 1, 400.0)}


def test_store_empty_file_is_an_empty_portfolio(tmp_path):
    path = tmp_path / "smm.db"
    path.touch()

    assert PortfolioStore(path).load() == {}


def test_store_file_that_is_not_a_database_errors_naming_its_path(tmp_path):
    path = tmp_path / "smm.db"
    path.write_text("{not a database")

    with pytest.raises(ValueError, match=re.escape(str(path))):
        PortfolioStore(path).load()
    assert path.read_text() == "{not a database"


QUOTES = {
    "AAPL": {"price": 200.0, "currency": "USD"},
    "MSFT": {"price": 300.0, "currency": "USD"},
    "BBCA.JK": {"price": 9000.0, "currency": "IDR"},
}


def start(server_url, tmp_path, quotes=QUOTES):
    return server_url(FakeFetcher(quotes=quotes), db_path=tmp_path / "smm.db")


async def set_position(url, ticker, direction="long", quantity: float = 10, avg_cost: float = 180.0):
    return await call_tool(
        url, "set_position", {"ticker": ticker, "direction": direction, "quantity": quantity, "avg_cost": avg_cost}
    )


async def get_portfolio(url) -> list[dict]:
    result = await call_tool(url, "get_portfolio", {})
    assert not result.is_error, text_of(result)
    assert result.structured_content is not None
    return result.structured_content["result"]


async def test_set_position_upper_cases_and_returns_the_saved_position(server_url, tmp_path):
    result = await set_position(start(server_url, tmp_path), "aapl", "long", 10.5, 180.0)

    assert not result.is_error, text_of(result)
    assert json.loads(text_of(result)) == {"ticker": "AAPL", "direction": "long", "quantity": 10.5, "avg_cost": 180.0}


async def test_portfolio_prices_each_position_in_its_own_currency(server_url, tmp_path):
    # AAPL long 10 @ 180 at 200: +200 on 1800 cost. MSFT short 2 @ 400 at 300: +200 on 800. BBCA long 100 @ 9500 at 9000: -50000 on 950000.
    url = start(server_url, tmp_path)
    await set_position(url, "AAPL", "long", 10, 180.0)
    await set_position(url, "MSFT", "short", 2, 400.0)
    await set_position(url, "BBCA.JK", "long", 100, 9500.0)

    assert await get_portfolio(url) == [
        {
            "ticker": "AAPL", "direction": "long", "quantity": 10, "avg_cost": 180.0,
            "price": 200.0, "currency": "USD", "unrealised_pnl": pytest.approx(200.0),
            "unrealised_pnl_pct": pytest.approx(100 / 9),
        },
        {
            "ticker": "BBCA.JK", "direction": "long", "quantity": 100, "avg_cost": 9500.0,
            "price": 9000.0, "currency": "IDR", "unrealised_pnl": pytest.approx(-50000.0),
            "unrealised_pnl_pct": pytest.approx(-100 / 19),
        },
        {
            "ticker": "MSFT", "direction": "short", "quantity": 2, "avg_cost": 400.0,
            "price": 300.0, "currency": "USD", "unrealised_pnl": pytest.approx(200.0),
            "unrealised_pnl_pct": pytest.approx(25.0),
        },
    ]


@pytest.mark.parametrize(
    "args",
    [
        {"ticker": "NOPE"},
        {"direction": "flat"},
        {"quantity": -1},
        {"quantity": float("inf")},
        {"quantity": float("nan")},
        {"avg_cost": 0},
        {"avg_cost": -5.0},
        {"avg_cost": float("inf")},
        {"avg_cost": float("nan")},
    ],
)
async def test_bad_position_is_rejected_with_nothing_saved(server_url, tmp_path, args):
    url = start(server_url, tmp_path)
    await set_position(url, "AAPL", "long", 10, 180.0)

    result = await set_position(url, **{"ticker": "AAPL", "direction": "short", "quantity": 1, "avg_cost": 1.0, **args})

    assert result.is_error
    [entry] = await get_portfolio(url)
    assert (entry["direction"], entry["quantity"], entry["avg_cost"]) == ("long", 10, 180.0)


async def test_unknown_ticker_error_names_it(server_url, tmp_path):
    result = await set_position(start(server_url, tmp_path), "nope")

    assert result.is_error
    assert "NOPE" in text_of(result)


async def test_sixteenth_position_is_rejected_but_replacing_a_held_one_works(server_url, tmp_path):
    tickers = [f"T{i}" for i in range(16)]
    url = start(server_url, tmp_path, quotes={t: {"price": 1.0, "currency": "USD"} for t in tickers})
    for ticker in tickers[:15]:
        assert not (await set_position(url, ticker)).is_error

    rejected = await set_position(url, tickers[15])
    replaced = await set_position(url, tickers[0], "short", 1, 2.0)

    assert rejected.is_error
    assert "15" in text_of(rejected)
    assert not replaced.is_error, text_of(replaced)
    held = await get_portfolio(url)
    assert [e["ticker"] for e in held] == sorted(tickers[:15])
    assert held[0]["direction"] == "short"


async def test_quantity_zero_deletes_and_deleting_one_not_held_succeeds(server_url, tmp_path):
    url = start(server_url, tmp_path)
    await set_position(url, "AAPL")
    await set_position(url, "MSFT")

    deleted = await set_position(url, "aapl", "short", quantity=0, avg_cost=-5.0)
    not_held = await set_position(url, "GONE", quantity=0)

    assert not deleted.is_error, text_of(deleted)
    assert json.loads(text_of(deleted)) == {"ticker": "AAPL", "quantity": 0}
    assert not not_held.is_error, text_of(not_held)
    assert [e["ticker"] for e in await get_portfolio(url)] == ["MSFT"]


async def test_failing_quote_gets_its_own_error_entry(server_url, tmp_path):
    db = tmp_path / "smm.db"
    # Seeded through the store: set_position would reject a ticker whose quote fails.
    PortfolioStore(db).set("DEAD", "long", 5, 10.0)
    url = server_url(FakeFetcher(quotes=QUOTES), db_path=db)
    await set_position(url, "AAPL")

    aapl, dead = await get_portfolio(url)

    assert aapl["unrealised_pnl"] == pytest.approx(200.0)
    assert dead["ticker"] == "DEAD"
    assert "DEAD" in dead["error"]
    assert "unrealised_pnl" not in dead


async def test_empty_portfolio_lists_as_empty(server_url, tmp_path):
    assert await get_portfolio(start(server_url, tmp_path)) == []


async def test_file_that_is_not_a_database_fails_both_tools_naming_its_path(server_url, tmp_path):
    path = tmp_path / "smm.db"
    path.write_text("{not a database")
    url = start(server_url, tmp_path)

    for tool, args in [
        ("get_portfolio", {}),
        ("set_position", {"ticker": "AAPL", "direction": "long", "quantity": 1, "avg_cost": 1.0}),
    ]:
        result = await call_tool(url, tool, args)
        assert result.is_error
        assert str(path) in text_of(result)
    assert path.read_text() == "{not a database"


async def test_positions_survive_a_fresh_server(server_url, tmp_path):
    await set_position(start(server_url, tmp_path), "AAPL", "short", 2, 250.0)

    [entry] = await get_portfolio(start(server_url, tmp_path))
    assert (entry["ticker"], entry["direction"], entry["quantity"], entry["avg_cost"]) == ("AAPL", "short", 2, 250.0)
