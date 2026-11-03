# Long-only, a cut Verdict, and the Re-entry Lock

Partly supersedes ADR-0005: the classifier still sees the Position, but the Cut Loss rule changes.

The server is long-only: there are no short Trade Plans and no short Positions. The Verdict is buy or hold on a ticker not held, and buy, hold, or cut on a held one. A confident cut starts a Re-entry Lock for that ticker and Interval: buys get no Trade Plan until a later bar closes above both the 10-bar EMA and the cut's Price. We chose this so that after a cut you re-enter on a structural trigger, not because the price is now below your cost. The server checks the trigger, and the classifier still has to say buy at or above the Minimum Confidence.

## Considered Options

- Let the classifier decide re-entry from a prompt that mentions the cut. Rejected: the check is self-judged with uncalibrated Confidence, and it busts the Signal cache whenever the lock changes.
- A separate yes/no "cut now?" question next to a buy/hold Verdict. Rejected: a second Confidence with no Minimum Confidence calibrated for it.
- A time-based cooldown after a cut. Rejected: time passing is not structure.
- Keep sell for unheld tickers as "stay out". Rejected: it has no Trade Plan to act on.

## Consequences

- `PROMPT_VERSION` goes to 3, so Hit Rates and the Minimum Confidence reset to their defaults.
- The Minimum Confidence is set from buy Hit Rates only, and cut uses the same threshold. The Cut Loss Hit Rate is reported but sets nothing.
- Sell is no longer scored. A cut is scored against the default Trade Plan distances mirrored below the Price (the old short Stop and Target).
- A cut can come at any Unrealised P&L, and every confident cut starts a Re-entry Lock. Cut Loss means a confident cut while the Position is losing.
- The lock is per ticker and Interval and is never in the prompt. Once lifted it stays lifted until the next confident cut.
- A Portfolio file holding a short Position fails loudly on load and names the tickers. `set_position` with quantity 0 still has to work, so the user can clear them.
