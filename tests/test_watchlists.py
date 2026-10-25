import json

import pytest
from fakes import FakeFetcher, call_tool, text_of

QUOTES = {t: {"price": 1.0, "currency": "IDR"} for t in ("BBCA.JK", "TLKM.JK", "ASII.JK")}


def start(server_url, tmp_path):
    return server_url(FakeFetcher(quotes=QUOTES), watchlist_path=tmp_path / "watchlists.json")


async def get_watchlists(url):
    result = await call_tool(url, "get_watchlists", {})
    assert not result.is_error, text_of(result)
    return json.loads(text_of(result))


async def test_set_then_get_round_trips(server_url, tmp_path):
    url = start(server_url, tmp_path)
    result = await call_tool(url, "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK", "TLKM.JK"]})
    assert not result.is_error, text_of(result)

    assert await get_watchlists(url) == {"idx": ["BBCA.JK", "TLKM.JK"]}


async def test_tickers_are_upper_cased_and_deduplicated_in_order(server_url, tmp_path):
    url = start(server_url, tmp_path)
    tickers = ["tlkm.jk", "BBCA.JK", "TLKM.JK", "bbca.jk"]
    result = await call_tool(url, "set_watchlist", {"name": "idx_2-a", "tickers": tickers})
    assert not result.is_error, text_of(result)

    assert await get_watchlists(url) == {"idx_2-a": ["TLKM.JK", "BBCA.JK"]}


@pytest.mark.parametrize("name", ["", "a" * 33, "my list", "idx!", "ünï"])
async def test_invalid_names_are_rejected(server_url, tmp_path, name):
    url = start(server_url, tmp_path)
    result = await call_tool(url, "set_watchlist", {"name": name, "tickers": ["BBCA.JK"]})

    assert result.is_error
    assert await get_watchlists(url) == {}


async def test_more_than_15_tickers_is_rejected(server_url, tmp_path):
    tickers = [f"T{i}.JK" for i in range(16)]
    fetcher = FakeFetcher(quotes={t: {"price": 1.0, "currency": "IDR"} for t in tickers})
    url = server_url(fetcher, watchlist_path=tmp_path / "watchlists.json")

    result = await call_tool(url, "set_watchlist", {"name": "big", "tickers": tickers})
    assert result.is_error
    assert "15" in text_of(result)

    result = await call_tool(url, "set_watchlist", {"name": "big", "tickers": tickers[:15]})
    assert not result.is_error, text_of(result)


async def test_unknown_ticker_rejects_the_whole_save(server_url, tmp_path):
    url = start(server_url, tmp_path)
    await call_tool(url, "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK"]})
    before = (tmp_path / "watchlists.json").read_bytes()

    result = await call_tool(
        url, "set_watchlist", {"name": "idx", "tickers": ["TLKM.JK", "BBCAA.JK", "NOPE.JK"]}
    )

    assert result.is_error
    assert "BBCAA.JK" in text_of(result)
    assert "NOPE.JK" in text_of(result)
    assert (tmp_path / "watchlists.json").read_bytes() == before


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


@pytest.mark.parametrize("content", ["{not json", '["idx"]', '{"idx": "BBCA.JK"}', '{"idx": [1]}'])
async def test_corrupt_file_errors_and_is_never_overwritten(server_url, tmp_path, content):
    path = tmp_path / "watchlists.json"
    path.write_text(content)
    url = start(server_url, tmp_path)

    assert (await call_tool(url, "get_watchlists", {})).is_error
    result = await call_tool(url, "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK"]})
    assert result.is_error
    assert path.read_text() == content


async def test_watchlists_survive_a_fresh_server(server_url, tmp_path):
    await call_tool(
        start(server_url, tmp_path), "set_watchlist", {"name": "idx", "tickers": ["BBCA.JK"]}
    )

    assert await get_watchlists(start(server_url, tmp_path)) == {"idx": ["BBCA.JK"]}
