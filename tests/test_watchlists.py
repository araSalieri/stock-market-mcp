"""Tests for Watchlists: the store directly, then set_watchlist and get_watchlists over MCP."""

import json

import pytest
from fakes import FakeFetcher, call_tool, text_of

from stock_market_mcp.errors import InputError
from stock_market_mcp.watchlists import WatchlistStore

QUOTES = {t: {"price": 1.0, "currency": "IDR"} for t in ("BBCA.JK", "TLKM.JK", "ASII.JK")}


def start(server_url, tmp_path):
    return server_url(FakeFetcher(quotes=QUOTES), db_path=tmp_path / "smm.db")


async def get_watchlists(url):
    result = await call_tool(url, "get_watchlists", {})
    assert not result.is_error, text_of(result)
    return json.loads(text_of(result))


async def test_set_then_get_round_trips(server_url, tmp_path):
    url = start(server_url, tmp_path)
    result = await call_tool(url, "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK", "TLKM.JK"]})
    assert not result.is_error, text_of(result)

    assert await get_watchlists(url) == {"idx": ["BBCA.JK", "TLKM.JK"]}


def test_store_upper_cases_and_deduplicates_tickers_in_order(tmp_path):
    store = WatchlistStore(tmp_path / "smm.db")
    store.set("idx_2-a", ["tlkm.jk", "BBCA.JK", "TLKM.JK", "bbca.jk"])

    assert store.load() == {"idx_2-a": ["TLKM.JK", "BBCA.JK"]}


@pytest.mark.parametrize("name", ["", "a" * 33, "my list", "idx!", "ünï"])
def test_store_rejects_invalid_names(tmp_path, name):
    store = WatchlistStore(tmp_path / "smm.db")

    with pytest.raises(InputError, match="1-32 letters"):
        store.set(name, ["BBCA.JK"])
    assert store.load() == {}


def test_store_caps_unique_tickers_at_15(tmp_path):
    store = WatchlistStore(tmp_path / "smm.db")
    tickers = [f"T{i}.JK" for i in range(16)]

    with pytest.raises(InputError, match="at most 15 tickers, got 16"):
        store.set("big", tickers)
    store.set("big", tickers[:15] + ["t0.jk"])

    assert store.load() == {"big": tickers[:15]}


async def test_unknown_ticker_rejects_the_whole_save(server_url, tmp_path):
    url = start(server_url, tmp_path)
    await call_tool(url, "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK"]})

    result = await call_tool(
        url, "set_watchlist", {"name": "idx", "tickers": ["TLKM.JK", "BBCAA.JK", "NOPE.JK"]}
    )

    assert result.is_error
    assert "BBCAA.JK" in text_of(result)
    assert "NOPE.JK" in text_of(result)
    assert await get_watchlists(url) == {"idx": ["BBCA.JK"]}


async def test_empty_list_deletes_and_missing_delete_is_fine(server_url, tmp_path):
    url = start(server_url, tmp_path)
    await call_tool(url, "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK"]})
    await call_tool(url, "set_watchlist", {"name": "us", "tickers": ["TLKM.JK"]})

    result = await call_tool(url, "set_watchlist", {"name": "idx", "tickers": []})
    assert not result.is_error, text_of(result)
    assert await get_watchlists(url) == {"us": ["TLKM.JK"]}

    result = await call_tool(url, "set_watchlist", {"name": "gone", "tickers": []})
    assert not result.is_error, text_of(result)
    assert await get_watchlists(url) == {"us": ["TLKM.JK"]}


async def test_no_file_means_no_watchlists(server_url, tmp_path):
    assert await get_watchlists(start(server_url, tmp_path)) == {}


async def test_file_that_is_not_a_database_errors_naming_its_path(server_url, tmp_path):
    path = tmp_path / "smm.db"
    path.write_text("{not a database")
    url = start(server_url, tmp_path)

    for tool, args in [("get_watchlists", {}), ("set_watchlist", {"name": "idx", "tickers": ["BBCA.JK"]})]:
        result = await call_tool(url, tool, args)
        assert result.is_error
        assert str(path) in text_of(result)
    assert path.read_text() == "{not a database"


async def test_watchlists_survive_a_fresh_server(server_url, tmp_path):
    await call_tool(
        start(server_url, tmp_path), "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK"]}
    )

    assert await get_watchlists(start(server_url, tmp_path)) == {"idx": ["BBCA.JK"]}
