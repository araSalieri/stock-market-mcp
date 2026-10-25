"""Container smoke: drive the deployed server through the published port.

Run against a container started from this checkout by `docker compose up -d --build`:

    uv run python scripts/container_smoke.py

Proves the data tools end to end with real Yahoo data, saves a throwaway
`smoke` Watchlist, runs `docker compose restart` and checks the Watchlist
survived on the volume, prints each Scan entry for eyeballing against a chart,
then makes real daily and 5m `get_signal` calls through the real classifier.
With `SMM_OPENROUTER_API_KEY` set on the host it validates the schema-enforced
verdict and its `trade_plan`; when the key is absent the missing-key tool error
is expected and the smoke stops with the exact key-setup step instead of faking
success.
"""

import asyncio
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx2
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from stock_market_mcp.classifier import validate_signal

URL = os.environ.get("SMM_SMOKE_URL", "http://127.0.0.1:8000/mcp")
TICKER = os.environ.get("SMM_SMOKE_TICKER", "AAPL")
WATCHLIST = "smoke"
WATCHLIST_TICKERS = ["BBCA.JK", "AAPL"]  # one IDX, one US
RESTART = ["docker", "compose", "restart"]
TOOLS = ["get_fundamentals", "get_quote", "get_signal", "get_watchlists", "scan", "set_watchlist"]

KEY_SETUP = (
    "STOP: real-classifier smoke needs an OpenRouter key on the host.\n"
    "  export SMM_OPENROUTER_API_KEY=<your-key>   # or put it in a .env file next to compose.yaml\n"
    "  then re-run: docker compose up -d --build && uv run python scripts/container_smoke.py"
)


class MissingKey(Exception):
    """Raised to stop the smoke when the host has no OpenRouter key."""


async def call(session, name, args):
    result = await session.call_tool(name, args)
    text = "".join(c.text for c in result.content if isinstance(c, types.TextContent))
    if result.is_error:
        if name == "get_signal" and "SMM_OPENROUTER_API_KEY" in text:
            raise MissingKey(f"{text}\n{KEY_SETUP}")
        raise RuntimeError(f"{name} failed: {text}")
    # A list result comes as one text block per item, so read the wrapped structured copy.
    if result.structured_content is not None:
        return result.structured_content["result"]
    return json.loads(text)


def check_scan_entry(entry):
    """No error entry; every field but ticker and data_date is a finite number or null."""
    assert "error" not in entry, entry
    for key, value in entry.items():
        if key in ("ticker", "data_date"):
            continue
        assert value is None or (isinstance(value, (int, float)) and math.isfinite(value)), (key, entry)


def check_signal(payload, interval):
    assert "trade_plan" in payload, "get_signal returned no trade_plan key"
    plan = payload.pop("trade_plan")
    price = payload.pop("price")
    assert isinstance(price, (int, float)) and math.isfinite(price), price
    verdict = validate_signal(payload)
    assert (plan is None) == (verdict["verdict"] == "hold"), plan
    if interval != "1d":
        # Intraday bars carry a time and the exchange offset.
        stamp = datetime.fromisoformat(verdict["data_date"])
        assert "T" in verdict["data_date"] and stamp.utcoffset() is not None, verdict["data_date"]
    print(
        f"get_signal {interval}: {verdict['verdict']} @ {price} "
        f"(confidence {verdict['confidence']}, data {verdict['data_date']}, plan {plan})"
    )


async def connected(steps):
    async with (
        streamable_http_client(URL) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        await steps(session)


async def before_restart(session):
    tools = await session.list_tools()
    names = sorted(t.name for t in tools.tools)
    assert names == TOOLS, names
    print(f"tools: {', '.join(names)}")

    quote = await call(session, "get_quote", {"ticker": TICKER})
    assert quote, "get_quote returned an empty payload"
    print(f"get_quote: {quote.get('ticker', TICKER)} @ {quote.get('price', quote)}")

    fundamentals = await call(session, "get_fundamentals", {"ticker": TICKER})
    assert fundamentals, "get_fundamentals returned an empty payload"
    print(f"get_fundamentals: {fundamentals.get('name', fundamentals)}")

    saved = await call(session, "set_watchlist", {"name": WATCHLIST, "tickers": WATCHLIST_TICKERS})
    assert saved["tickers"] == WATCHLIST_TICKERS, saved
    lists = await call(session, "get_watchlists", {})
    assert lists.get(WATCHLIST) == WATCHLIST_TICKERS, lists
    print(f"set_watchlist + get_watchlists: {WATCHLIST} = {lists[WATCHLIST]}")


async def after_restart(session):
    try:
        lists = await call(session, "get_watchlists", {})
        assert lists.get(WATCHLIST) == WATCHLIST_TICKERS, f"Watchlist lost on restart: {lists}"
        print(f"after restart: {WATCHLIST} = {lists[WATCHLIST]}")

        entries = await call(session, "scan", {"watchlist": WATCHLIST})
        assert [e["ticker"] for e in entries] == WATCHLIST_TICKERS, entries
        for entry in entries:
            print(f"scan: {json.dumps(entry)}")
            check_scan_entry(entry)
    finally:
        await call(session, "set_watchlist", {"name": WATCHLIST, "tickers": []})
    assert WATCHLIST not in await call(session, "get_watchlists", {})
    print(f"deleted Watchlist {WATCHLIST}")

    # Real classifier calls last, so a missing key still leaves the rest proven.
    for interval in ("1d", "5m"):
        check_signal(await call(session, "get_signal", {"ticker": TICKER, "interval": interval}), interval)


async def main():
    await connected(before_restart)

    print(f"running: {' '.join(RESTART)}")
    subprocess.run(RESTART, cwd=Path(__file__).resolve().parent.parent, check=True)
    for attempt in range(30):
        try:
            await connected(after_restart)
            break
        except* httpx2.TransportError:
            # The server isn't listening yet; anything else is a real failure.
            if attempt == 29:
                raise
            await asyncio.sleep(1)
    print("smoke passed")


if __name__ == "__main__":
    # Exit cleanly outside the MCP task group so the stop-and-ask message
    # never arrives wrapped in a BaseExceptionGroup traceback.
    try:
        asyncio.run(main())
    except* MissingKey as eg:
        # anyio wraps through two task-group layers; unwrap to the leaf.
        err = eg.exceptions[0]
        while isinstance(err, BaseExceptionGroup):
            err = err.exceptions[0]
        print(err)
        sys.exit(2)
