# Backfilled Signals count toward Hit Rates

A 1d Signal takes 20 bars, about four weeks, to get an Outcome, and a Confidence band needs 30 Targets plus Stops before it can set a Minimum Confidence. On 1d that takes months of live calls. `backfill_signals(ticker, interval, since)` makes Signals on past bars whose next 20 bars are already known, scores them right away, and counts them with live Signals under the same Prompt Version.

- `since` is required, with no default. The prompt shows the real ticker and dates, so a bar from before the classifier's training cutoff may be one whose outcome the model already knows. The caller decides how far back is safe; the server doesn't know the cutoff, and the model can be swapped with `SMM_CLASSIFIER_MODEL`.
- Each replayed bar sees what a live Signal would: the Interval's default period ending at that bar, no Position, and no previous Signal, the same as a ticker's first live Signal. The prompt is the same as a live one, so `PROMPT_VERSION` doesn't change.
- The server replays only tickers that aren't held, so it produces buy, dip and hold. A cut needs a Position at a past date, and none existed, so the Cut Minimum Confidence still fills from live cuts only.
- Each call replays up to 50 bars, oldest first, and skips any bar that already has a Signal under the current Prompt Version for that ticker and Interval. It returns how many it logged and how many are left.
- Each row is marked as backfilled. Nothing reads the mark yet; it exists so these rows can be split out or deleted later.

## Considered Options

- Let the agent set the Minimum Confidence. Rejected: undoes ADR-0003, so the number would be a guess, not a measurement.
- Score 1d over fewer bars. Rejected: changes what an Outcome means on one Interval only.
- Hide the ticker and dates in every prompt, so any date is safe. Rejected for now: bumps the Prompt Version, resets every Hit Rate, and intraday prompts lose time of day.
- Keep backfilled Signals in a separate count that sets nothing. Rejected: doesn't shorten the 1d wait.
- Read the cutoff from an env var. Rejected in favour of a required argument, chosen by the user.

## Consequences

- If a caller passes a `since` from before the model's cutoff, Hit Rates can come out too high, and nothing catches it.
- Replayed bars are always completed bars, while a live Signal during the session sees a bar that is still forming.
- Each replayed bar costs one classifier call.
- How deep backfill can go depends on how long Yahoo keeps bars for each Interval: 1d has no limit, 1h keeps 730 days, and 5m and 15m keep 60 days, which leaves about a month after a 1mo lookback.
