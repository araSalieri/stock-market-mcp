# Server logs and scores Signals

The server stores every Signal it gets from the classifier in a SQLite file and scores it later, so Confidence can be checked against a measured Hit Rate and the Minimum Confidence can be set from data. This amends ADR-0002: Watchlists are no longer the server's only state.

- Each cache miss logs one Signal with its prompt version and the default Trade Plans (2 ATR Stop, Reward Ratio 2) for both Directions, whatever the caller asked for, so all Signals are judged by one yardstick.
- Scoring runs inside `get_signal`, on the history it already fetched: pending Signals for the same ticker and Interval get an Outcome within 20 bars. No extra Yahoo calls, no scheduler.
- The most recent Signal for the ticker and Interval (Verdict, Confidence, Outcome or pending) is always in the classifier's state.
- The Minimum Confidence is set by the server: the lowest 0.1 Confidence band whose buy/sell Hit Rate is above 33% (break-even at Reward Ratio 2) with at least 30 scored Signals, over the current prompt version. The `min_confidence` parameter is removed.

SQLite over a JSON file because Signals grow without bound and Hit Rates need grouping by ticker, Interval, Confidence band and prompt version. Watchlists move into the same file (`SMM_DB_PATH`, replacing `SMM_WATCHLIST_PATH`), so the server has one state file; an existing `watchlists.json` is not imported.

## Consequences

- Signals for tickers or Intervals never asked for again stay unscored, as do ones whose window falls outside the fetched history. Yahoo keeps 1m bars for about 7 days.
- Feeding the previous Signal back in has no baseline to compare against; the model may anchor on its own last answer.
- After a prompt change, the Hit Rate in the reply is null and the Minimum Confidence is 0.6 until the new version has enough scored Signals.
- If enough data shows no Confidence band breaks even, no Signal gets a Trade Plan.
- Agents still passing `min_confidence` get no error: the server ignores unknown arguments.
- Watchlists saved before the move are lost on upgrade and must be re-created.
