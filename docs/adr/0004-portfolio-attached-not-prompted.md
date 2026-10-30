# Portfolio is attached to Signals, not fed to the classifier

The server stores one Portfolio of Positions in the shared SQLite file (ADR-0003), set through `set_position` and read through `get_portfolio`. When `get_signal` is called for a ticker the user holds, the reply carries the Position and its Unrealised P&L at the Signal's Price. The classifier never sees the Position: its prompt is unchanged, so every Signal is still judged by one yardstick and Hit Rates stay comparable across held and unheld tickers. The agent reads the Verdict against the Position, e.g. sell on a held long means exit.

The alternative was putting the Position in the prompt so the Verdict itself is about the holding. That needs a new prompt version, resets the Hit Rate and Minimum Confidence to their defaults, and mixes holding-aware and plain Signals in one score.

## Consequences

- One unnamed Portfolio; a ticker has at most one Position, long or short.
- At most 15 Positions, so `get_portfolio` stays within what free Yahoo tolerates (same limit as a Watchlist).
- Unrealised P&L is per Position in the ticker's currency; there is no Portfolio total and no FX data.
- Like Watchlists, Positions have no auth: anyone on the LAN can read or replace them.
