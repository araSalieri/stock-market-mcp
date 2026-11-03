# Classifier sees the Position, so it can call a Cut Loss

Partly superseded by ADR-0006.

Supersedes ADR-0004. The `get_signal` prompt now carries the Position for a held ticker (Direction, average cost, Unrealised P&L % at the Price; no quantity) and `"position": null` otherwise, so the Verdict on a held ticker is about the holding. A Cut Loss is a sell on a losing long or a buy on a losing short at or above the Minimum Confidence. We chose this so the classifier, not the agent, judges whether to cut, at the cost of the single yardstick ADR-0004 kept.

## Considered Options

- Keep the classifier blind and let the server flag an opposing Verdict on a losing Position (ADR-0004 unchanged). Rejected: the Verdict ignores that the user holds the stock.
- A second classifier call with a yes/no cut-loss question, leaving the Verdict blind. Rejected: an extra call, a Confidence that isn't calibrated, and a new Outcome rule.

## Consequences

- `PROMPT_VERSION` is 2: Hit Rate and Minimum Confidence reset to their defaults until v2 has enough scored Signals.
- Held and unheld Signals share one Hit Rate and Minimum Confidence. Each Signal stores `cut_loss` (1 against a losing Position at any Confidence, 0 held otherwise, null not held), so they can be split later.
- Cut Loss Hit Rate reuses existing Outcomes: it is the Hit Rate over v2 Signals with `cut_loss = 1`, across all tickers and Intervals.
- Quantity stays out of the prompt so size doesn't sway the Verdict and changing it doesn't bust the Signal cache; any other Position change does.
- A Cut Loss sell still gets its short Trade Plan; `cut_loss: true` in the `position` key tells the agent to close the long.
- Existing database files gain the `cut_loss` column on open; v1 rows stay null.
