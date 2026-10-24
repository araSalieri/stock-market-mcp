# Signal carries the Trade Plan; no raw history tool

`get_signal` returns the Trade Plan for its own Verdict (buy is long, sell is short, hold has none), and the separate `get_plan` and `get_history` tools are removed. As separate tools, the plan and the Signal could disagree (a long plan next to a hold Verdict), and a year of raw history was too large for an agent to use in one call.

## Consequences

- Every Trade Plan now needs the OpenRouter key; there is no key-free plan.
- An agent can't plan a Direction the classifier didn't pick, and can't inspect raw OHLCV or indicators to check a Verdict.
- A sell Verdict gives a short plan even on markets where retail shorting isn't available (e.g. IDX); the agent decides whether it applies.
