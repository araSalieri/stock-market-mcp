# The classifier picks a cut's Buyback Price and Cut Stop

Partly supersedes ADR-0006 and ADR-0007: a cut is no longer scored against the default Trade Plan distances mirrored below the Price, and the Buyback Price is no longer the cut's Price minus the default Target distance.

The 4 ATR Buyback Price was rarely reached: of 69 scored confident cuts under prompt version 3, price fell that far first in 9. So on a held ticker the classifier now answers two more choice questions next to the Verdict: which chart level below the Price is the Buyback Price, and which level above it is the Cut Stop. The server works out the candidates from the bars (20-bar and 50-bar lows or highs, the lower or upper Bollinger Band, the 10-bar EMA, the SMA, VWAP), offers only those on the right side of the Price, plus "no level". A side with no candidate isn't asked, since a choice of "no level" alone may be rejected, and counts as "no level". The Decisions API can't return a free-form price, only pick from options, which is why the server supplies the levels. A cut is then scored against the two picked levels, so the cut Hit Rate measures whether the classifier's levels work.

## Considered Options

- Cut Price minus 2 ATR. Rejected: still a fixed distance that ignores the chart.
- Classifier picks an ATR multiple on a score scale. Rejected in favour of levels traders recognise on the chart.
- Ask for the levels in a second request only on a confident cut, keeping the prompt unchanged. Rejected: two calls and an extra failure point per cut.
- Score cuts at the old mirrored distances when a level is missing. Rejected: the cut Hit Rate would mix two rules.

## Consequences

- `PROMPT_VERSION` goes to 4, so every Hit Rate and the Minimum Confidence reset to their defaults.
- Every held Signal asks up to three questions; the levels are used only on a cut. Every cut stores its picked levels as its scoring Target and Stop, at any Confidence; a cut with "no level" for either is never scored.
- A confident cut's reply has `buyback_price`, `buyback_quantity`, `cut_stop` and the confidence of each pick, all null when no level was picked. The pick confidences are not calibrated and set nothing.
- A cut's distances now vary, so the cut Hit Rate has no fixed break-even. It sets no Minimum Confidence, so nothing else moves.
- Hold and buy scoring keep the default Trade Plan distances.
