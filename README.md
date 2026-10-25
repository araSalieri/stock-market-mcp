# stock-market-mcp

An MCP server that gives coding agents stock market data, indicators, and advisory trading guidance. Data comes from Yahoo Finance via yfinance. It never places orders. Its only write is saving your Watchlists.

## Tools

| Tool | Arguments | Returns |
|------|-----------|---------|
| `get_quote` | `ticker` | Current price and currency |
| `get_fundamentals` | `ticker` | Name, sector, industry, market cap |
| `get_signal` | `ticker`, `interval="1d"`, `period` (default per Interval), `atr_multiple=2`, `reward_ratio=2` | Advisory Signal: a buy/sell/hold Verdict with confidence, rationale, `data_date`, `price` (last close), and `trade_plan`. Needs an OpenRouter key. |
| `set_watchlist` | `name`, `tickers` | Saves the Watchlist, replacing its tickers. An empty list deletes it. |
| `get_watchlists` | none | Every Watchlist as `{name: [tickers]}` |
| `scan` | `tickers` (or `watchlist`, not supported yet) | One Scan entry per ticker, in input order. No Verdict, no OpenRouter key. |

`get_signal` computes `sma`, `ema`, `rsi`, `macd`, `bbands`, `atr`, `adx` and `vwap` over `period` on bars of the given `interval` and sends them to the classifier. `trade_plan` is `null` for hold. For buy (long) or sell (short) it holds `direction`, `entry` (last close), `stop` (`atr_multiple` × ATR(14) away), `target` (`reward_ratio` × the Stop distance), `atr`, `atr_multiple` and `reward_ratio`. `atr_multiple` and `reward_ratio` must be numbers in (0, 10].

`interval` is one of `1m`, `5m`, `15m`, `1h` or `1d`. Without `period` it defaults to `5d` for 1m, `1mo` for 5m and 15m, `6mo` for 1h, and `1y` for 1d. Yahoo keeps 7 days of 1m bars, 60 days of 5m and 15m bars, and 730 days of 1h bars, so a longer `period` is a tool error. On intraday Intervals the last bar may still be forming: `price` is its close, `data_date` is its timestamp, and the ATR comes from those bars, so the Stop and Target are intraday-sized.

`set_watchlist` takes a `name` (1–32 letters, digits, `-` or `_`) and up to 15 `tickers`. Tickers are upper-cased and de-duplicated, and each is checked with a quote fetch. If any fail, nothing is saved. Watchlists are stored in one JSON file at `SMM_WATCHLIST_PATH`. If that file is corrupt, both tools return an error and leave it alone so you can fix it by hand.

`scan` takes exactly one of `watchlist` or `tickers`. Tickers are upper-cased and de-duplicated, up to 15. Each entry is built from 3 months of daily bars and holds `ticker`, `price` (latest close), `data_date` (latest bar's timestamp), `prev_high`, `prev_low` and `prev_close` (the last daily bar before today in the exchange's timezone, so during the session today's forming bar is left out), `atr_pct` (ATR(14) as a percent of `prev_close`) and `rsi` (RSI(14)). A ticker that fails gets `{ticker, error}` instead, and the others still return. Results aren't ranked or filtered.

Results are cached in memory: quotes and history for 60 seconds, fundamentals and Signals for 5 minutes. Bad input, unknown tickers, and missing data return a tool error rather than an empty result. Domain terms are defined in [GLOSSARY.md](GLOSSARY.md).

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

The compose setup listens on all interfaces, so other machines on your LAN can reach it. There is no auth, so anyone on your LAN can read and replace your Watchlists. compose keeps Watchlists on a named volume, so they survive restarts and image updates.

## Connect an agent

Claude Code:

```sh
claude mcp add --transport http stock-market http://127.0.0.1:8000/mcp
```

Other clients: point them at the same URL as a streamable HTTP server.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `SMM_OPENROUTER_API_KEY` | none | Required for `get_signal`, so every Trade Plan needs it. Other tools work without it. |
| `SMM_CLASSIFIER_MODEL` | `typesafe/jev-1.13` | Model used for `get_signal` |
| `SMM_HOST` | `127.0.0.1` | Listen address (compose sets `0.0.0.0`) |
| `SMM_PORT` | `8000` | Listen port |
| `SMM_CLASSIFIER_MAX_STATE_CHARS` | `32000` | Max size of the market data sent to the classifier |
| `SMM_WATCHLIST_PATH` | `watchlists.json` in the working directory | Watchlist file (compose sets a path on its volume) |

compose passes only the first two from `.env`. When running from source, export variables in your shell.

## Development

```sh
uv run pytest
uv run pyright
```

CI runs both on every push and pull request. Pushing a `v*` tag runs them, then publishes the image to `ghcr.io/arasalieri/stock-market-mcp`.

To smoke-test a running container end to end: `uv run python scripts/container_smoke.py`.
