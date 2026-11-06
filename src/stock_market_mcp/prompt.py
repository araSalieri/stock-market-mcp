"""Prompt module: everything the classifier sees, and the Prompt Version that names it.

No model, no I/O – serialization only, so NaN handling and date formats are
unit-tested with literals as truth.
"""

import json
import os
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from stock_market_mcp.portfolio import PricedPosition

# Stored with each Signal; bump whenever anything below changes what the classifier sees (the golden prompt test checks).
PROMPT_VERSION = 3

# typesafe/jev-1.13 rejects states over ~32k input tokens with a 400
# max_tokens_exceeded (verified live); this payload tokenizes at ~1.2 chars per
# token, so a 32k-char budget lands near 27k tokens – under the cap with headroom.
DEFAULT_MAX_STATE_CHARS = 32_000


class PreviousSignal(NamedTuple):
    """The previous Signal as the classifier sees it; outcome is None while pending."""

    verdict: str
    confidence: float
    outcome: str | None

_BUY = "Bullish momentum or trend evidence in the supplied data."
_HOLD = "Mixed or insufficient evidence in the supplied data."

# The single question the Verdict hinges on: _NOT_HELD_QUESTIONS without a Position, _HELD_QUESTIONS with one.
_NOT_HELD_QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "Given the supplied price history and indicators, is now the right "
            "time to buy this stock, or to hold off? The signal is advisory."
        ),
        "criteria": {"buy": _BUY, "hold": _HOLD},
    },
}
_HELD_QUESTIONS = {
    "verdict": {
        "type": "choice",
        "instructions": (
            "A long position in this stock is supplied. Given the price history, "
            "indicators and the position, should the holder buy more, hold, or cut "
            "(exit the long now)? The signal is advisory."
        ),
        "criteria": {
            "buy": _BUY,
            "hold": _HOLD,
            "cut": "Bearish evidence in the supplied data: exit the long now, at a gain or a loss.",
        },
    },
}


def questions(held: bool) -> dict:
    """The Decisions questions for a state with (held) or without a Position."""
    return _HELD_QUESTIONS if held else _NOT_HELD_QUESTIONS


@dataclass(frozen=True)
class SignalPrompt:
    """cache_key is the state without the previous Signal, so storing a Signal doesn't defeat the Signal cache.

    questions are what the classifier is asked about the state."""

    cache_key: str
    state: str
    questions: dict


def signal_prompt(
    ticker: str,
    period: str,
    df: pd.DataFrame,
    *,
    interval: str = "1d",
    position: PricedPosition | None = None,
    previous: PreviousSignal | None = None,
    max_chars: int | None = None,
) -> SignalPrompt:
    """What the classifier reads: OHLCV history plus joined indicator columns, the Position and the previous Signal.

    The frame must already carry the indicator columns the tool computed; rows
    are emitted as ISO date strings, with null for NaN (warmup rows of later
    lookbacks).

    The Position (average cost and Unrealised P&L %, no quantity, so size doesn't
    sway the Verdict or bust the cache: ADR-0005) sits in the head, null when not
    held, so row trimming never drops it. previous is the previous Signal's
    Verdict, Confidence and Outcome (None while pending); it goes in the head of
    the state only.

    If the payload without the previous Signal exceeds the character budget, the
    oldest rows are dropped until it fits: recency is the point of a timing
    signal, so the most recent rows are the ones kept. The state may run the
    previous Signal's length over the budget.
    """
    view = None if position is None else {"avg_cost": position.avg_cost, "unrealised_pnl_pct": position.unrealised_pnl_pct}
    head = {"ticker": ticker, "period": period, "interval": interval, "position": view}
    rows = _rows(df, head, max_chars)
    cache_key = json.dumps({**head, "rows": rows})
    state = cache_key
    if previous is not None:
        previous_view = {"verdict": previous.verdict, "confidence": previous.confidence, "outcome": previous.outcome or "pending"}
        state = json.dumps({**head, "previous_signal": previous_view, "rows": rows})
    return SignalPrompt(cache_key, state, questions(position is not None))


def _rows(df: pd.DataFrame, head: dict, max_chars: int | None) -> list:
    """The newest rows that fit the character budget alongside head."""
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
    if not rows:
        return []
    # Serialize each row once and count back from the newest: json.dumps joins rows with ", ".
    size = len(json.dumps({**head, "rows": []})) + len(json.dumps(rows[-1]))
    start = len(rows) - 1
    while start > 0:
        size += len(json.dumps(rows[start - 1])) + 2
        if size > max_chars:
            break
        start -= 1
    return rows[start:]

