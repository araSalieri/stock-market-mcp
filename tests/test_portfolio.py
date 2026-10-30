"""Tests for the Portfolio (#58): the store directly, then set_position and get_portfolio over MCP."""

import re

import pytest

from stock_market_mcp.portfolio import PortfolioStore

# Store tests call PortfolioStore directly: the spec (#57) puts a storage seam under the tools, and an
# empty or broken file must fail the same way for every caller (get_signal reads it too in #59).


def test_store_set_then_load_round_trips(tmp_path):
    store = PortfolioStore(tmp_path / "smm.db")
    store.set("AAPL", "long", 10.5, 180.0)
    store.set("BBCA.JK", "short", 100, 9000)

    assert store.load() == {
        "AAPL": {"direction": "long", "quantity": 10.5, "avg_cost": 180.0},
        "BBCA.JK": {"direction": "short", "quantity": 100, "avg_cost": 9000},
    }


def test_store_set_replaces_the_tickers_position(tmp_path):
    store = PortfolioStore(tmp_path / "smm.db")
    store.set("AAPL", "long", 10, 180.0)
    store.set("AAPL", "short", 3, 200.0)

    assert store.load() == {"AAPL": {"direction": "short", "quantity": 3, "avg_cost": 200.0}}


def test_store_delete_removes_one_and_missing_is_fine(tmp_path):
    store = PortfolioStore(tmp_path / "smm.db")
    store.set("AAPL", "long", 10, 180.0)
    store.set("MSFT", "long", 1, 400.0)

    store.delete("AAPL")
    store.delete("NVDA")

    assert store.load() == {"MSFT": {"direction": "long", "quantity": 1, "avg_cost": 400.0}}


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
