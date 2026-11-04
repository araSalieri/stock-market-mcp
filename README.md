> [!WARNING]
> This repository is written by AI, except for this text :P. Its just for my own curiosity how to use [mattpock skills](https://github.com/mattpocock/skills) and how MCP works

# stock-market-mcp

An MCP server that gives coding agents stock market data, indicators and advisory trading guidance from Yahoo Finance. It never places orders. It writes only its own state: your Watchlists, your Portfolio and a log of past Signals.

## Tools

| Tool | What it does |
|------|--------------|
| `get_quote` | Current price and currency |
| `get_fundamentals` | Name, sector, industry, market cap |
| `get_signal` | Long-only advice for a ticker: buy or hold, plus cut (exit now) if you hold it. Comes with a confidence, a rationale and, for a confident buy, a Trade Plan (entry, stop, target). After a confident cut, buys get no Trade Plan until the price recovers (Re-entry Lock). On a held ticker it also shows your Unrealised P&L. Needs an OpenRouter key. |
| `get_track_record` | How past Signals turned out, by confidence and Verdict, optionally for one ticker or Interval |
| `set_watchlist` | Save a Watchlist of up to 15 tickers. An empty list deletes it. |
| `get_watchlists` | Every saved Watchlist |
| `scan` | Quick numbers (price, ATR, RSI, Gap, Relative Volume, VWAP distance) for a Watchlist or list of tickers, to pick which deserve a Signal |
| `set_position` | Save the quantity and average cost you hold in a ticker. Quantity 0 deletes it. At most 15 Positions. |
| `get_portfolio` | Every Position with its latest price and Unrealised P&L |

Each tool's arguments and output are described in its MCP schema. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).

Free Yahoo data is delayed, so check `data_date` before acting on a Signal or Trade Plan. Results are cached in memory for 60 seconds (quotes, history) or 5 minutes (fundamentals, Signals).

## Run

With Docker:

```sh
cp .env.example .env   # add your OpenRouter key
docker compose up -d
```

From source (needs [uv](https://docs.astral.sh/uv/)):

```sh
uv run stock-market-mcp
```

The server listens at `http://127.0.0.1:8000/mcp` (streamable HTTP). The compose setup listens on all interfaces and has no auth, so anyone on your LAN can read and replace your Watchlists and Portfolio. Both are kept in a SQLite file on a named volume and survive restarts. Older `watchlists.json` files are not imported: re-create those lists with `set_watchlist`.

## Connect an agent

Start the server first; each agent connects to `http://127.0.0.1:8000/mcp`.

**Claude Code:**

```sh
claude mcp add --transport http stock-market http://127.0.0.1:8000/mcp
```

**Pi:** Pi has no built-in MCP support, so install [pi-mcp-adapter](https://github.com/nicobailon/pi-mcp-adapter) with `pi install npm:pi-mcp-adapter` and restart Pi. Then add the server to `.mcp.json` in your project or `~/.config/mcp/mcp.json` for all projects:

```json
{
  "mcpServers": {
    "stock-market": { "url": "http://127.0.0.1:8000/mcp" }
  }
}
```

**Hermes:** add the server to `~/.hermes/config.yaml`, then run `/reload-mcp` or restart Hermes:

```yaml
mcp_servers:
  stock-market:
    url: "http://127.0.0.1:8000/mcp"
```

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SMM_OPENROUTER_API_KEY` | none | Required for `get_signal` |
| `SMM_CLASSIFIER_MODEL` | `typesafe/jev-1.13` | Model used for `get_signal` |
| `SMM_HOST` | `127.0.0.1` | Listen address (compose sets `0.0.0.0`) |
| `SMM_PORT` | `8000` | Listen port |
| `SMM_CLASSIFIER_MAX_STATE_CHARS` | `32000` | Max size of the market data sent to the classifier |
| `SMM_DB_PATH` | `stock-market-mcp.db` | SQLite file for server state (compose sets a path on its volume) |
| `SMM_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

compose reads `SMM_OPENROUTER_API_KEY`, `SMM_CLASSIFIER_MODEL` and `SMM_LOG_LEVEL` from `.env`.

## Development

```sh
scripts/check.sh
```

It runs the tests, pyright and the en dash check. CI runs it on every push and pull request. Pushing a `v*` tag also publishes the image to `ghcr.io/arasalieri/stock-market-mcp`.

Before tagging, run the smoke test against a local build: `docker compose up -d --build`, then `uv run python scripts/container_smoke.py`. Run it once during an IDX or US session and once outside one, since Gap, Relative Volume and VWAP distance take different paths.
