# Server stores Watchlists

The server keeps named Watchlists in a JSON file (`SMM_WATCHLIST_PATH`, a Docker volume in compose), edited through `set_watchlist`, instead of the agent passing the tickers on every Scan. That makes it the server's only state and its only write. It still never places orders. Lists persisted on the server outlive any one agent session, and it's the user's choice to keep them there rather than in prompts.

## Consequences

- There is no auth, and compose listens on all interfaces, so anyone on the LAN can read or replace your Watchlists.
- A Watchlist holds at most 15 tickers, so one Scan stays within what free Yahoo tolerates.
