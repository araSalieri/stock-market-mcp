# stock-market-mcp

An MCP server that gives coding agents stock market data, indicators and advisory trading guidance from Yahoo Finance. It never places orders. It writes only its own state: your Watchlists, your Portfolio and a log of past Signals.

## Tools

| Tool | What it does |
|------|--------------|
| `get_quote` | Current price and currency |
| `get_fundamentals` | Name, sector, industry, market cap |
| `get_signal` | Buy/sell/hold Verdict with confidence, rationale and a Trade Plan (entry, stop, target). Only a buy or sell at or above the Minimum Confidence gets a Trade Plan. The server sets that from past Signals' Hit Rates (there is no `min_confidence` argument) and shows it as `min_confidence`; null means no buy or sell gets a Trade Plan. `hit_rate` is the Hit Rate of past Signals of the same kind for the ticker and Interval, null until enough are scored. On a held ticker, `position` carries the Position, its Unrealised P&L at the Signal's Price, and `cut_loss`: true when the Verdict points against a losing Position at or above the Minimum Confidence (a Cut Loss), and `cut_loss_hit_rate`: the Cut Loss Hit Rate over scored past Signals that pointed against a losing Position at any Confidence, across all tickers and Intervals, null until 30 are scored. The Trade Plan is unchanged. Needs an OpenRouter key. |
| `get_track_record` | Track Record: how the current prompt version's Signals turned out, per 0.1 Confidence band (ascending, bands with Signals only) and Verdict. Buy and sell show Target, Stop, Expired and pending counts and a Hit Rate (Targets / (Targets + Stops)); hold shows Quiet, Missed and pending and a Hit Rate (Quiet / (Quiet + Missed)). Each Hit Rate is null until 10 are scored. Optional `ticker` and `interval` narrow it; Signals with no default Trade Plan are left out. Reads the Signal log only, no market data or OpenRouter call. |
| `set_watchlist` | Save a Watchlist of up to 15 tickers. An empty list deletes it. |
| `get_watchlists` | Every saved Watchlist |
| `scan` | Price, previous day's range, ATR, RSI, Gap, Relative Volume and VWAP distance for a Watchlist or list of tickers |
| `set_position` | Save the Position for a ticker: long or short, quantity and average cost. Replaces any held one; quantity 0 deletes it and replies with just the ticker and quantity 0. At most 15 Positions. |
| `get_portfolio` | Every Position with its latest price and Unrealised P&L in the ticker's own currency, with no total |

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

The server listens at `http://127.0.0.1:8000/mcp` (streamable HTTP). Connect Claude Code with:

```sh
claude mcp add --transport http stock-market http://127.0.0.1:8000/mcp
```

The compose setup listens on all interfaces and has no auth, so anyone on your LAN can read and replace your Watchlists and Portfolio. Both are kept in a SQLite file on a named volume and survive restarts. Older `watchlists.json` files are not imported: re-create those lists with `set_watchlist`.

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
uv run pytest
uv run pyright
```

CI runs both on every push and pull request. Pushing a `v*` tag also publishes the image to `ghcr.io/arasalieri/stock-market-mcp`.

Before tagging, run the smoke test against a local build: `docker compose up -d --build`, then `uv run python scripts/container_smoke.py`. Run it once during an IDX or US session and once outside one, since Gap, Relative Volume and VWAP distance take different paths.
