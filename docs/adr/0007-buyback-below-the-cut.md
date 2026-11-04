# Buyback below the cut

Partly supersedes ADR-0006: the server stays long-only and a confident cut still starts a Re-entry Lock, but the lock can now lift below the cut as well as above it.

A cut can work like a short made with your own shares: sell, then buy back more shares lower with the same cash. So the Re-entry Lock lifts on whichever comes first: a bar's low touches the Buyback Price (the cut's Price minus the default Target distance), or a completed bar closes above the Re-entry Price. We picked the cut's mirrored Target because it is already the price that scores the cut as a Target, so "buyback opened" and "the cut was right" always agree. Either lift only opens re-entry: a Trade Plan still needs a buy Verdict at or above the Minimum Confidence, because price falling to a level says nothing about whether it has stopped falling.

## Considered Options

- Keep the close-above lift only (ADR-0006). Rejected: it forbids the buy-lower trade the cut is often made for.
- Buyback Price at a recent support level or a fixed percent below the cut. Rejected: a new rule with no Outcome scoring behind it.
- Reaching the Buyback Price gets a Trade Plan on its own. Rejected: no Hit Rate behind it, and it buys into a falling price.
- Real short Positions. Rejected: out of scope; ADR-0006's long-only rule stands.

## Consequences

- The lock is shown from the cut with its Buyback Price; the Buyback Quantity (whole shares the cut's proceeds buy at the Buyback Price) appears only while the ticker is not held.
- The lock stores the cut quantity, and a lifted lock records which way it lifted.
- The Re-entry Hit Rate is given separately for buyback and re-entry lifts.
- The lock is still never in the prompt, so `PROMPT_VERSION` doesn't change.
