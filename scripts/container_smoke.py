"""Container smoke: drive the deployed server through the published port.

Run against a container started from this checkout by `docker compose up -d --build`:

    uv run python scripts/container_smoke.py

`SMM_SMOKE_URL` sets the endpoint and `SMM_SMOKE_PROJECT` the compose project
(`docker compose -p`) the smoke restarts. Before any MCP call it checks that the
project has a running container publishing the URL's port, so it never writes to
or restarts a server it didn't mean to.

Proves the data tools end to end with real Yahoo data, saves a throwaway
`smoke` Watchlist (aborting if one already exists), runs `docker compose restart`
and checks the Watchlist survived on the volume, prints each Scan entry for
eyeballing against a chart, deletes the Watchlist even if a step failed, then
makes real daily and 5m `get_signal` calls through the real classifier.
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
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from stock_market_mcp.classifier import validate_signal

URL = os.environ.get("SMM_SMOKE_URL", "http://127.0.0.1:8000/mcp")
PROJECT = os.environ.get("SMM_SMOKE_PROJECT")
TICKER = os.environ.get("SMM_SMOKE_TICKER", "AAPL")
WATCHLIST = "smoke"
WATCHLIST_TICKERS = ["BBCA.JK", "AAPL"]  # one IDX, one US
ROOT = Path(__file__).resolve().parent.parent
# Without a project, compose derives it from ROOT, as in the README run.
COMPOSE = ["docker", "compose", *(["-p", PROJECT] if PROJECT else [])]
RETRIES = 30  # one-second tries for the server to come back after the restart
TOOLS = ["get_fundamentals", "get_portfolio", "get_quote", "get_signal", "get_track_record", "get_watchlists", "scan", "set_position", "set_watchlist"]

KEY_SETUP = (
    "STOP: real-classifier smoke needs an OpenRouter key on the host.\n"
    "  export SMM_OPENROUTER_API_KEY=<your-key>   # or put it in a .env file next to compose.yaml\n"
    "  then re-run: docker compose up -d --build && uv run python scripts/container_smoke.py"
)

CLEANUP = (
    f"CLEANUP: could not delete the {WATCHLIST!r} Watchlist. Once the container is up, run in {ROOT}:\n"
    f"  {' '.join(COMPOSE)} exec stock-market-mcp /app/.venv/bin/python -c "
    "'import os; from stock_market_mcp.watchlists import WatchlistStore; "
    f'WatchlistStore(os.environ["SMM_DB_PATH"]).set("{WATCHLIST}", [])\''
)


class MissingKey(Exception):
    """Raised to stop the smoke when the host has no OpenRouter key."""


def compose(*args, capture=False):
    return subprocess.run([*COMPOSE, *args], cwd=ROOT, check=True, capture_output=capture, text=True).stdout


def check_target():
    """Exit unless the project we restart runs a container publishing the URL's port."""
    url = urlsplit(URL)
    port = url.port or (443 if url.scheme == "https" else 80)
    lines = compose("ps", "--format", "json", capture=True).splitlines()
    running = [c for c in map(json.loads, filter(None, lines)) if c["State"] == "running"]
    if not any(p["PublishedPort"] == port for c in running for p in c.get("Publishers") or []):
        sys.exit(
            f"ABORT: `{' '.join(COMPOSE)} ps` in {ROOT} shows no running container publishing "
            f"port {port} from SMM_SMOKE_URL={URL}. Start it with `docker compose up -d --build`, "
            "or set SMM_SMOKE_PROJECT and SMM_SMOKE_URL to the project and port to test."
        )
    print(f"target: {', '.join(c['Name'] for c in running)} on port {port}")


def wait_until_up():
    """Poll the URL until the server answers; any HTTP status means it's listening."""
    for attempt in range(RETRIES):
        try:
            urllib.request.urlopen(URL, timeout=2).close()
            return
        except urllib.error.HTTPError:
            return
        except OSError:
            # Refused or reset while the server comes back; give up after RETRIES.
            if attempt == RETRIES - 1:
                raise
            time.sleep(1)


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


async def check_watchlist(session, label, message="Watchlist mismatch"):
    watchlists = await call(session, "get_watchlists", {})
    assert watchlists.get(WATCHLIST) == WATCHLIST_TICKERS, f"{message}: {watchlists}"
    print(f"{label}: {WATCHLIST} = {watchlists[WATCHLIST]}")


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
    min_confidence = payload.pop("min_confidence")
    payload.pop("hit_rate")
    payload.pop("position", None)
    verdict = validate_signal(payload)
    # A buy or sell below the Minimum Confidence has no Trade Plan either.
    assert plan is None or verdict["verdict"] != "hold", plan
    assert plan is None or (min_confidence is not None and verdict["confidence"] >= min_confidence), plan
    if interval != "1d":
        # Intraday bars carry a time and the exchange offset.
        stamp = datetime.fromisoformat(verdict["data_date"])
        assert "T" in verdict["data_date"] and stamp.utcoffset() is not None, verdict["data_date"]
    print(
        f"get_signal {interval}: {verdict['verdict']} @ {price} "
        f"(confidence {verdict['confidence']}, data {verdict['data_date']}, plan {plan})"
    )


async def in_session(steps):
    """Open one MCP session, run steps in it and return their result."""
    async with (
        streamable_http_client(URL) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await steps(session)


async def check_data_tools(session):
    """Return True if a Watchlist named WATCHLIST already exists."""
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

    return WATCHLIST in await call(session, "get_watchlists", {})


async def save_watchlist(session):
    saved = await call(session, "set_watchlist", {"name": WATCHLIST, "tickers": WATCHLIST_TICKERS})
    assert saved["tickers"] == WATCHLIST_TICKERS, saved
    await check_watchlist(session, "set_watchlist + get_watchlists")


async def check_persisted(session):
    await check_watchlist(session, "after restart", "Watchlist lost on restart")
    entries = await call(session, "scan", {"watchlist": WATCHLIST})
    assert [e["ticker"] for e in entries] == WATCHLIST_TICKERS, entries
    for entry in entries:
        print(f"scan: {json.dumps(entry)}")
        check_scan_entry(entry)


async def delete_watchlist(session):
    await call(session, "set_watchlist", {"name": WATCHLIST, "tickers": []})
    assert WATCHLIST not in await call(session, "get_watchlists", {})
    print(f"deleted Watchlist {WATCHLIST}")


async def check_signals(session):
    for interval in ("1d", "5m"):
        check_signal(await call(session, "get_signal", {"ticker": TICKER, "interval": interval}), interval)
    record = await call(session, "get_track_record", {"ticker": TICKER})
    assert "bands" in record and "min_confidence" in record, record
    print(f"get_track_record: {len(record['bands'])} bands, min_confidence {record['min_confidence']}")


async def main():
    check_target()
    if await in_session(check_data_tools):
        sys.exit(f"ABORT: a Watchlist named {WATCHLIST!r} already exists on {URL}; not touching it.")
    try:
        await in_session(save_watchlist)
        print(f"running: {' '.join(COMPOSE)} restart")
        compose("restart")
        wait_until_up()
        await in_session(check_persisted)
    finally:
        try:
            wait_until_up()
            await in_session(delete_watchlist)
        except Exception:
            print(CLEANUP, file=sys.stderr)
            raise
    # Real classifier calls last, so a missing key still leaves the rest proven.
    await in_session(check_signals)
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
