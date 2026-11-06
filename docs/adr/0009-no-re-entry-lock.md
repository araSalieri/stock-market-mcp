# No Re-entry Lock

Supersedes the Re-entry Lock parts of ADR-0006, ADR-0007 and ADR-0008. The rest of ADR-0006 stands: the server is long-only, cut is a Verdict on a held Position, and Cut Loss and the Cut Loss Hit Rate stay. ADR-0010 changes how a cut is scored.

The lock changed a Trade Plan in one case only: a confident buy while locked got none. Its armed and lifted states and the Re-entry Hit Rate only fed reporting, yet took three ADRs, four column migrations and a second failure path in every Scan. So a confident cut no longer starts anything stored, and a buy gets a Trade Plan whenever its Confidence is at or above the Minimum Confidence.

The buyback from ADR-0007 stays, without lock state: a confident cut's Signal carries the Buyback Price and the Buyback Quantity (ADR-0010 says how the price is picked). Later Signals and Scan don't show it, and reaching it opens or blocks nothing.

## Considered Options

- Go back to ADR-0006's lock (locked until a completed bar closes above the Re-entry Price, then lifted). Rejected: still a stored per-ticker state with no Hit Rate showing it helps.
- Keep the lock and trim only its reporting. Rejected: the states and arming paths, which are the hard part, would stay.
- Store the last confident cut so later Signals and Scan show its Buyback Price. Rejected: brings back a per-ticker table for a number the cut reply already gives.

## Consequences

- Nothing stops a confident buy right after a confident cut. The Minimum Confidence is the only gate on a Trade Plan.
- `get_signal` drops `re_entry_lock` and `re_entry_hit_rate`; `get_track_record` drops `re_entry_hit_rate`, and `scan` drops `re_entry_locks` and `re_entry_locks_error`.
- On open, the server drops the `reentry_locks` table and the `signals.re_entry` column. Lock history is lost.
- The lock was never in the prompt, so `PROMPT_VERSION` doesn't change and Hit Rates keep counting.
