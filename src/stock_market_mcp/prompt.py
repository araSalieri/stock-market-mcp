"""Prompt module: everything the classifier sees, and the Prompt Version that names it.

No model, no I/O – serialization only, so NaN handling and date formats are
unit-tested with literals as truth.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd

from stock_market_mcp.portfolio import PricedPosition

# Stored with each Signal; bump whenever anything below changes what the classifier sees (the golden prompt test checks).
PROMPT_VERSION = 4

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

# The question the Verdict hinges on: _NOT_HELD_QUESTIONS without a Position, _HELD_QUESTIONS with one.
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


# The questions a held ticker is also asked (ADR-0010): each picks a chart level on its side of the Price, or none.
_LEVEL_INSTRUCTIONS = {
    "buyback": (
        "If the holder cuts now, which level below the price is the buyback price: where buying back is cheaper "
        "and price is likely to reach? Pick none if no level fits."
    ),
    "cut_stop": (
        "If the holder cuts now, which level above the price is the cut stop: where the cut is proven wrong? "
        "Pick none if no level fits."
    ),
}
_NONE = {"buyback": "No level below the Price fits.", "cut_stop": "No level above the Price fits."}


def questions(levels: dict[str, dict[str, float]] | None) -> dict:
    """The Decisions questions: the Verdict alone without a Position (levels None), else also one per side of
    levels, offering its candidates by name with the price in the description, plus none."""
    if levels is None:
        return _NOT_HELD_QUESTIONS
    return _HELD_QUESTIONS | {
        question: {
            "type": "choice",
            "instructions": _LEVEL_INSTRUCTIONS[question],
            "criteria": {
                **{name: f"{_LEVEL_NAMES[name]} at {price:.6g}" for name, price in candidates.items()},
                "none": _NONE[question],
            },
        }
        for question, candidates in levels.items()
    }


_LEVEL_NAMES = {
    "low_20": "20-bar low",
    "low_50": "50-bar low",
    "high_20": "20-bar high",
    "high_50": "50-bar high",
    "lower_band": "Lower Bollinger Band",
    "upper_band": "Upper Bollinger Band",
    "ema": "10-bar EMA",
    "sma": "SMA",
    "vwap": "VWAP",
}
# Indicator columns read by prefix, since pandas-ta suffixes them with their parameters.
_SHARED_LEVELS = (("ema", "EMA_"), ("sma", "SMA_"), ("vwap", "VWAP_"))


def levels(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Candidate Buyback Prices and Cut Stops as of the latest bar: finite levels strictly below (buyback) or above
    (cut_stop) its close, by name, the first of any at the same price kept."""
    price = float(df["Close"].iloc[-1])
    last = df.iloc[-1]

    def column(prefix: str) -> float:
        return next((float(last[c]) for c in df.columns if str(c).startswith(prefix)), math.nan)

    def extreme(side: str, bars: int) -> float:
        window = df[side].iloc[-bars:]
        return float(window.min() if side == "Low" else window.max()) if len(df) >= bars else math.nan

    shared = [(name, column(prefix)) for name, prefix in _SHARED_LEVELS]
    sides = {
        "buyback": [("low_20", extreme("Low", 20)), ("low_50", extreme("Low", 50)), ("lower_band", column("BBL_")), *shared],
        "cut_stop": [("high_20", extreme("High", 20)), ("high_50", extreme("High", 50)), ("upper_band", column("BBU_")), *shared],
    }
    picked: dict[str, dict[str, float]] = {}
    for question, candidates in sides.items():
        below = question == "buyback"
        offered: dict[str, float] = {}
        for name, value in candidates:
            if math.isfinite(value) and (value < price if below else value > price) and value not in offered.values():
                offered[name] = value
        picked[question] = offered
    return picked


@dataclass(frozen=True)
class SignalPrompt:
    """cache_key is the state without the previous Signal, so storing a Signal doesn't defeat the Signal cache.

    questions are what the classifier is asked about the state; levels the candidate prices the level questions
    offer, by question and name, None when not held."""

    cache_key: str
    state: str
    questions: dict
    levels: dict[str, dict[str, float]] | None


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
    offered = None if position is None else levels(df)
    return SignalPrompt(cache_key, state, questions(offered), offered)


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

