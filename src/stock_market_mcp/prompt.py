"""Pure prompt builder for get_signal: history + indicators to a compact classifier prompt.

No model, no I/O — serialization only, so NaN handling and date formats are
unit-tested with literals as truth.
"""

import json
import os

import pandas as pd

# typesafe/jev-1.13 rejects states over ~32k input tokens with a 400
# max_tokens_exceeded (verified live); this payload tokenizes at ~1.2 chars per
# token, so a 32k-char budget lands near 27k tokens — under the cap with headroom.
DEFAULT_MAX_STATE_CHARS = 32_000


def build_signal_prompt(
    ticker: str,
    period: str,
    df: pd.DataFrame,
    *,
    interval: str = "1d",
    max_chars: int | None = None,
) -> str:
    """Serialize OHLCV history plus joined indicator columns for the classifier.

    The frame must already carry the indicator columns the tool computed; rows
    are emitted as ISO date strings, with null for NaN (warmup rows of later
    lookbacks).

    If the serialized payload exceeds the character budget, the oldest rows are
    dropped until it fits: recency is the point of a timing signal, so the most
    recent rows are the ones kept.
    """
    if max_chars is None:
        # Read at call time; empty counts as unset (compose passthrough pattern).
        max_chars = int(os.environ.get("SMM_CLASSIFIER_MAX_STATE_CHARS") or DEFAULT_MAX_STATE_CHARS)
    # Yahoo names the index Date on daily bars and Datetime on intraday; every row
    # carries Date so data_date reads the same key on both.
    payload = df.rename_axis("Date").reset_index().to_json(orient="records", date_format="iso")
    if payload is None:
        raise ValueError("could not serialize history rows")
    rows = json.loads(payload)
    head = {"ticker": ticker, "period": period, "interval": interval}
    prompt = json.dumps({**head, "rows": rows})
    while len(prompt) > max_chars and len(rows) > 1:
        rows = rows[1:]
        prompt = json.dumps({**head, "rows": rows})
    return prompt
