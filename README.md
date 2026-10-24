# stock-market-mcp

A read-only MCP server that gives coding agents stock market data, indicators, and advisory trading guidance. Data comes from Yahoo Finance via yfinance. It never places orders.

## Tools

| Tool | Arguments | Returns |
|------|-----------|---------|
| `get_quote` | `ticker` | Current price and currency |
| `get_fundamentals` | `ticker` | Name, sector, industry, market cap |
| `get_history` | `ticker`, `period="1y"`, `indicators=[]` | OHLCV rows, optionally joined with indicators: `sma`, `ema`, `rsi`, `macd`, `bbands`, `atr`, `adx`, `vwap` |
| `get_signal` | `ticker`, `period="1y"` | Advisory Signal: a buy/sell/hold Verdict with confidence, rationale, and data date. Needs an OpenRouter key. |
| `get_plan` | `ticker`, `direction` (`long` or `short`), `atr_multiple=2`, `reward_ratio=2` | Advisory Trade Plan: Entry (last close), Stop (`atr_multiple` × ATR(14) away), and Target (`reward_ratio` × the Stop distance) |

Results are cached in memory: quotes for 60 seconds, everything else for 5 minutes. Bad input, unknown tickers, and missing data return a tool error rather than an empty result. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).

Free Yahoo data is delayed, so check `data_date` before acting on a Signal or Trade Plan.

## Run

With Docker (published image):

```sh
cp .env.example .env   # add your OpenRouter key
docker compose up -d
```

From source (needs [uv](https://docs.astral.sh/uv/)):

```sh
uv run stock-market-mcp
```

The server speaks streamable HTTP at `http://127.0.0.1:8000/mcp`.

The compose setup listens on all interfaces, so other machines on your LAN can reach it. There is no auth.

## Connect an agent

Claude Code:

```sh
claude mcp add --transport http stock-market http://127.0.0.1:8000/mcp
```

Other clients: point them at the same URL as a streamable HTTP server.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SMM_OPENROUTER_API_KEY` | none | Required for `get_signal`. Other tools work without it. |
| `SMM_CLASSIFIER_MODEL` | `typesafe/jev-1.13` | Model used for `get_signal` |
| `SMM_HOST` | `127.0.0.1` | Listen address (compose sets `0.0.0.0`) |
| `SMM_PORT` | `8000` | Listen port |
| `SMM_CLASSIFIER_MAX_STATE_CHARS` | `32000` | Max size of the market data sent to the classifier |

compose passes only the first two from `.env`. When running from source, export variables in your shell.

## Development

```sh
uv run pytest
uv run pyright
```

CI runs both on every push and pull request. Pushing a `v*` tag runs them, then publishes the image to `ghcr.io/arasalieri/stock-market-mcp`.

To smoke-test a running container end to end: `uv run python scripts/container_smoke.py`.
