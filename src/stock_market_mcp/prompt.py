"""Pure prompt builder for get_signal: history + indicators to a compact classifier prompt.

No model, no I/O – serialization only, so NaN handling and date formats are
unit-tested with literals as truth.
"""

import json
import os

import pandas as pd

# typesafe/jev-1.13 rejects states over ~32k input tokens with a 400
# max_tokens_exceeded (verified live); this payload tokenizes at ~1.2 chars per
# token, so a 32k-char budget lands near 27k tokens – under the cap with headroom.
DEFAULT_MAX_STATE_CHARS = 32_000


def build_signal_prompt(
    ticker: str,
    period: str,
    df: pd.DataFrame,
    *,
    interval: str = "1d",
    position: dict | None = None,
    max_chars: int | None = None,
) -> str:
    """Serialize OHLCV history plus joined indicator columns for the classifier.

    The frame must already carry the indicator columns the tool computed; rows
    are emitted as ISO date strings, with null for NaN (warmup rows of later
    lookbacks).

    position (average cost, Unrealised P&L %) sits in the head, null when not held,
    so row trimming never drops it.

    If the serialized payload exceeds the character budget, the oldest rows are
    dropped until it fits: recency is the point of a timing signal, so the most
    recent rows are the ones kept.
    """
    if max_chars is None:
        # Read at call time; empty counts as unset (compose passthrough pattern).
        max_chars = int(os.environ.get("SMM_CLASSIFIER_MAX_STATE_CHARS") or DEFAULT_MAX_STATE_CHARS)
    # Yahoo names the index Date on daily bars and Datetime on intraday; every row
    # carries Date so the classifier reads the same key on both.
    frame = df.rename_axis("Date").reset_index()
    # Bar times keep the exchange's offset, as scan gives them; to_json would turn them into UTC.
    frame["Date"] = [t.isoformat() for t in frame["Date"]]
    payload = frame.to_json(orient="records", date_format="iso")
    if payload is None:
        raise ValueError("could not serialize history rows")
    rows = json.loads(payload)
    head = {"ticker": ticker, "period": period, "interval": interval, "position": position}
    if not rows:
        return json.dumps({**head, "rows": []})
    # Serialize each row once and count back from the newest: json.dumps joins rows with ", ".
    size = len(json.dumps({**head, "rows": []})) + len(json.dumps(rows[-1]))
    start = len(rows) - 1
    while start > 0:
        size += len(json.dumps(rows[start - 1])) + 2
        if size > max_chars:
            break
        start -= 1
    return json.dumps({**head, "rows": rows[start:]})


def with_previous_signal(prompt: str, previous_signal: dict) -> str:
    """prompt with previous_signal (Verdict, Confidence, Outcome) in its head, where row trimming never drops it.

    The rows were trimmed without it, so the prompt may run that many characters over the budget.
    """
    payload = json.loads(prompt)
    rows = payload.pop("rows")
    return json.dumps({**payload, "previous_signal": previous_signal, "rows": rows})
