# OpenRouter Decisions API

What the classifier (`typesafe/jev-1.13`) can answer. Read before designing anything the classifier decides. Source: [OpenRouter's Jev tutorial](https://openrouter.ai/docs/guides/community/jev-tutorial.md); the full API reference wasn't available when this was written.

## Request

`POST https://openrouter.ai/api/alpha/decisions` with `model`, `state` (the prompt, one string) and `questions`, a map of question name to question. The model rejects `chat/completions`.

## Question types

| Type | Request | Answer |
|---|---|---|
| `choice` | `instructions`, `criteria`: option name to description | `choice` (one of the option names), `confidence`, `probabilities` per option |
| `score` | `instructions`, `criteria`: ordered list of level descriptions | `score` (an index), `confidence`, `probabilities` by index, `legend` |
| `noul` | `instructions`, `criteria`: `true` and `false` descriptions | `noul`, the probability of yes in [0, 1]; no separate confidence |

Answers come back under `answers`, keyed by question name.

## Limits

- **No free-form values.** No type returns a price, a number of your choosing, or text. To get a price, offer priced options (ADR-0010 does this with chart levels). The server writes the rationale itself.
- **Each question is answered on every call.** A question only some Verdicts need is still asked and answered.

## Unverified

Check these live (the container smoke's held `get_signal` call is the cheapest test) before relying on them:

- Whether a `choice` question with one option is accepted. The server never sends one: a level side with no candidate isn't asked.
- Whether cost scales with the number of questions.
- The state size cap: about 32k input tokens returns a 400 `max_tokens_exceeded` (verified live; see `DEFAULT_MAX_STATE_CHARS`).
