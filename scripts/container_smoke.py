"""Container smoke: drive the deployed server through the published port.

Run against a container started by `docker compose up -d --build`:

    uv run python scripts/container_smoke.py

Proves the three data tools end to end, then makes one real `get_signal`
call through the real classifier. With `SMM_OPENROUTER_API_KEY` set on the
host it validates the schema-enforced verdict; when the key is absent the
missing-key tool error is expected and the smoke stops with the exact
key-setup step instead of faking success.
"""

import asyncio
import json
import os
import sys

from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from stock_market_mcp.classifier import validate_signal

URL = os.environ.get("SMM_SMOKE_URL", "http://127.0.0.1:8000/mcp")
TICKER = os.environ.get("SMM_SMOKE_TICKER", "AAPL")

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
        raise RuntimeError(f"{name} failed: {text}")
    return text


async def main():
    async with (
        streamable_http_client(URL) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        tools = await session.list_tools()
        names = sorted(t.name for t in tools.tools)
        assert names == ["get_fundamentals", "get_history", "get_quote", "get_signal"], names
        print(f"tools: {', '.join(names)}")

        quote = json.loads(await call(session, "get_quote", {"ticker": TICKER}))
        assert quote, "get_quote returned an empty payload"
        print(f"get_quote: {quote.get('ticker', TICKER)} @ {quote.get('price', quote)}")

        fundamentals = json.loads(await call(session, "get_fundamentals", {"ticker": TICKER}))
        assert fundamentals, "get_fundamentals returned an empty payload"
        print(f"get_fundamentals: {fundamentals.get('name', fundamentals)}")

        history = json.loads(
            await call(session, "get_history", {"ticker": TICKER, "indicators": ["sma", "rsi"]})
        )
        assert history["rows"], "get_history returned no rows"
        print(f"get_history: {len(history['rows'])} rows, last {history['rows'][-1]['Date'][:10]}")

        # One real classifier call; the missing-key tool error is the pass
        # condition when the host has no key, and we stop and ask instead.
        result = await session.call_tool("get_signal", {"ticker": TICKER})
        text = "".join(c.text for c in result.content if isinstance(c, types.TextContent))
        if result.is_error:
            if "SMM_OPENROUTER_API_KEY" in text:
                raise MissingKey(f"{text}\n{KEY_SETUP}")
            raise RuntimeError(f"get_signal failed: {text}")
        verdict = validate_signal(json.loads(text))
        print(
            f"get_signal: {verdict['verdict']} "
            f"(confidence {verdict['confidence']}, data {verdict['data_date']})"
        )
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
