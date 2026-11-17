# Cuts get their own Minimum Confidence per Interval

Partly supersedes ADR-0010: the cut Hit Rate now sets something, a Cut Minimum Confidence per Interval, measured against the Cut Break-even instead of a fixed rate.

Under prompt version 4, 23 of 133 scored cuts reached the Buyback Price first (1h: 16 of 126). The picked Cut Stop sits about half as far from the Price as the Buyback Price (median 1.65% against 3.14% on 1h), so a raw Hit Rate means little: a random cut would win about a third of the time. Each cut's break-even is its Cut Stop distance over the sum of both distances, and a band of cuts is worth acting on only when its Hit Rate beats the average of those. On 1h the 0.8 and 0.9 bands made 6 Targets each against about 15 expected at break-even, so cuts there were worse than chance. The 1d sample (7 of 7) is too small to judge, which is why the gate is per Interval.

The rule copies buy's: the lowest 0.1 Confidence band with at least 30 Targets plus Stops whose Hit Rate beats its Cut Break-even; 0.6 until a band has that many; none when some do and none beats it. It counts every scored cut on the Interval, on winning and losing Positions alike. A cut below it keeps its Verdict and is still scored, but is not a Cut Loss and carries no Buyback Price, Cut Stop or Buyback Quantity, the same way a weak buy carries no Trade Plan.

## Considered Options

- Raw cut Hit Rate above 50%. Rejected: rewards picking a far Cut Stop and a near Buyback Price instead of cutting well.
- Go back to fixed ATR distances so break-even is constant. Rejected: undoes ADR-0010's chart levels.
- Change the prompt first (higher-Interval trend, confirmation close). Rejected for now: resets every Hit Rate, and the gate is measurable without it. Still open as a next step.
- One server-wide cut gate. Rejected: 1h's record would silence 1d.
- Rewrite a gated cut to hold. Rejected: hides what the classifier said and mixes cuts into hold scoring.
- Count only cuts on losing Positions. Rejected: whether price falls next doesn't depend on the holder's cost, and the sample fills slower.

## Consequences

- With today's log, 1h has no Cut Minimum Confidence, so no 1h cut is a Cut Loss; 1d stays at 0.6 until it has 30 scored cuts in a band.
- `get_signal` on a held ticker adds `cut_min_confidence` for its Interval. `get_track_record` adds `break_even` to each band's cut entry and `cut_min_confidence` when an interval is given.
- Buy's Minimum Confidence and the Cut Loss Hit Rate don't change.
- Nothing in the prompt changes, so `PROMPT_VERSION` stays at 4 and Hit Rates keep counting.
