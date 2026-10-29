import logging
import re

from fakes import FakeFetcher, call_tool, text_of
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


def audit_lines(caplog):
    return [r for r in caplog.records if " outcome=" in r.getMessage()]


async def test_ok_call_writes_one_info_audit_line(server_url, caplog):
    caplog.set_level(logging.INFO, logger="stock_market_mcp")
    fetcher = FakeFetcher(quotes={"AAPL": {"price": 190.0, "currency": "USD"}})
    await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    [line] = audit_lines(caplog)
    assert line.levelno == logging.INFO
    assert re.fullmatch(r"tool=get_quote ticker=AAPL ms=\d+ outcome=ok", line.getMessage())


async def test_failing_call_writes_error_audit_line_and_keeps_tool_error_text(server_url, caplog):
    caplog.set_level(logging.INFO, logger="stock_market_mcp")
    fetcher = FakeFetcher(error=RuntimeError("yahoo is down"))
    result = await call_tool(server_url(fetcher), "get_quote", {"ticker": "AAPL"})

    assert result.is_error
    assert text_of(result) == "Error executing tool get_quote: yahoo is down"
    [line] = audit_lines(caplog)
    assert re.fullmatch(r"tool=get_quote ticker=AAPL ms=\d+ outcome=error", line.getMessage())


# One call per registered tool; a new tool must be added here, so it can't skip the audit line.
EVERY_TOOL_ARGS = {
    "get_quote": {"ticker": "AAPL"},
    "get_fundamentals": {"ticker": "AAPL"},
    "get_signal": {"ticker": "AAPL"},
    "scan": {"tickers": ["AAPL"]},
    "set_watchlist": {"name": "tech", "tickers": ["AAPL"]},
    "get_watchlists": {},
}


async def test_every_registered_tool_writes_an_audit_line(server_url, caplog, tmp_path):
    caplog.set_level(logging.INFO, logger="stock_market_mcp")
    url = server_url(FakeFetcher(), watchlist_path=tmp_path / "watchlists.json")
    async with streamable_http_client(url) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        listed = {t.name for t in (await session.list_tools()).tools}
        assert listed == set(EVERY_TOOL_ARGS)
        for name, args in EVERY_TOOL_ARGS.items():
            await session.call_tool(name, args)

    audited = [r.getMessage().split()[0] for r in audit_lines(caplog)]
    assert audited == [f"tool={name}" for name in EVERY_TOOL_ARGS]


async def test_audit_line_includes_defaulted_arguments(server_url, caplog):
    caplog.set_level(logging.INFO, logger="stock_market_mcp")
    await call_tool(server_url(FakeFetcher()), "get_signal", {"ticker": "AAPL"})

    [line] = audit_lines(caplog)
    assert line.getMessage().startswith(
        "tool=get_signal ticker=AAPL interval=5m period=None atr_multiple=2.0 reward_ratio=2.0 min_confidence=0.6 ms="
    )


async def test_tool_schema_keeps_argument_names_defaults_and_description(server_url):
    async with streamable_http_client(server_url(FakeFetcher())) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        [tool] = [t for t in (await session.list_tools()).tools if t.name == "get_signal"]

    props = tool.input_schema["properties"]
    assert list(props) == ["ticker", "interval", "period", "atr_multiple", "reward_ratio", "min_confidence"]
    assert tool.input_schema["required"] == ["ticker"]
    assert {k: v.get("default") for k, v in props.items()} == {
        "ticker": None,
        "interval": "5m",
        "period": None,
        "atr_multiple": 2.0,
        "reward_ratio": 2.0,
        "min_confidence": 0.6,
    }
    assert tool.description is not None
    assert tool.description.startswith("Advisory buy/sell/hold Signal")
