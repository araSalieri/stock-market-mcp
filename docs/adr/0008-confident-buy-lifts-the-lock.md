# A confident buy lifts the Re-entry Lock

Partly supersedes ADR-0006 and ADR-0007: the price triggers stay, but they now arm the Re-entry Lock instead of lifting it.

A lift on price alone said nothing about whether the classifier agreed, so a lock could show as lifted and count toward the Re-entry Hit Rate while every buy after it was below the Minimum Confidence. Now the lock goes locked, then armed, then lifted. It is armed when a bar's low touches the Buyback Price or a completed bar closes above the Re-entry Price, by whichever comes first, and stays armed until the next confident cut. It lifts at the first buy at or above the Minimum Confidence while armed. That buy's Confidence is the lift's Confidence, so no new classifier question is asked. Trade Plans come out the same as before; what changes is what "lifted" means and what the Re-entry Hit Rate counts.

## Considered Options

- Keep the price-only lift (ADR-0007). Rejected: "lifted" and the Re-entry Hit Rate included re-entries the classifier never backed.
- A separate "has the move finished?" question with its own Confidence. Rejected for the reason ADR-0006 gave: a second Confidence with no calibrated Minimum Confidence, plus a `PROMPT_VERSION` bump.
- Make each Signal re-check the trigger. Rejected: a buyback touch is a past event, and making price touch it again would throw it away.
- Set a separate Minimum Confidence for the lifting buy from the Re-entry Hit Rate. Rejected: too few lifts to calibrate it. The rate stays reported only.

## Consequences

- `re_entry_lock` gains `state: armed` with `armed_by` and `armed_date`. A lifted lock's `lifted_date` is the lifting buy's date, and `lifted_by` is the path that armed it.
- The Re-entry Hit Rate counts the lifting buy, which is confident by definition, per path.
- Locks already lifted on price stay lifted. Their first-buy tags drop out of the Re-entry Hit Rate.
- While the Minimum Confidence is null, no lock lifts.
- The lock is still never in the prompt, so `PROMPT_VERSION` doesn't change.
