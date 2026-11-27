# Sentiment is a separate tool, never fed into the Signal

`get_sentiment` reads X posts about a ticker through xAI's `x_search` and returns a Reading (bullish, bearish, mixed or quiet), logged and scored in its own table. It never goes into the classifier prompt and never changes a Verdict, Confidence or Minimum Confidence. Putting X posts in the prompt would bump the Prompt Version and reset every Hit Rate. It would add an X charge to every Signal and every backfilled bar. And Backfilled Signals would break, since a search of past posts run by a model that knows what happened next leaks hindsight.

## Considered Options

- Feed the X summary into the Signal prompt. Rejected for the reasons above, and because nothing yet shows X chatter improves Hit Rates.
- Stateless like Scan. Rejected: the user wants evidence of whether a Reading is worth anything, so bullish and bearish Sentiments are scored like buy and its mirror, per ticker, Interval and Reading.
- Reach Grok through OpenRouter. Rejected: xAI documents `x_search` only on its own Responses API, so the server calls `api.x.ai` with `SMM_XAI_API_KEY`.

## Consequences

- Sentiment Hit Rates set nothing. If they show a real edge, feeding Readings into the Signal is a later decision with its own Prompt Version bump.
- There is no backfill, so Sentiment Hit Rates fill from live calls only, and a 1d Sentiment takes about four weeks to score.
- A pending Sentiment is scored only when `get_sentiment` is called again for that ticker and Interval.
