"""Signal log: one row per fresh classifier answer in the server database (ADR-0003)."""

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pandas as pd

from stock_market_mcp.db import transaction

# Bars after a Signal's own bar within which it gets an Outcome.
OUTCOME_BARS = 20
# Scored Signals of one kind (Targets plus Stops, or scored holds) needed before a Hit Rate is reported.
HIT_RATE_MIN_SCORED = 10
# Targets plus Stops a Confidence band needs before its Hit Rate can set the Minimum Confidence.
BAND_MIN_SCORED = 30
# Targets plus Stops among Cut Loss Signals needed before the Cut Loss Hit Rate is reported.
CUT_LOSS_MIN_SCORED = 30
# Minimum Confidence until a band has BAND_MIN_SCORED.
MIN_CONFIDENCE = 0.6

# Outcome is null while pending; plan prices are the default Trade Plans, null when one has non-positive prices.
# cut_loss: 1 when the Verdict opposed a losing Position at any Confidence, 0 when held otherwise, null when not held.
SCHEMA = """CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY,
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    prompt_version INTEGER NOT NULL,
    data_date TEXT NOT NULL,
    verdict TEXT NOT NULL,
    confidence REAL NOT NULL,
    price REAL NOT NULL,
    long_stop REAL,
    long_target REAL,
    short_stop REAL,
    short_target REAL,
    outcome TEXT,
    scored_at TEXT,
    cut_loss INTEGER
)"""


class SignalLog:
    def __init__(self, path: str | os.PathLike):
        self.path = path

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A transaction on the signals table, adding cut_loss to a file from before it existed (null on old rows)."""
        with transaction(self.path, SCHEMA) as conn:
            # shortcut: two servers upgrading one file at once can fail one call on a duplicate column; retry succeeds.
            if "cut_loss" not in {column[1] for column in conn.execute("PRAGMA table_info(signals)")}:
                conn.execute("ALTER TABLE signals ADD COLUMN cut_loss INTEGER")
            yield conn

    def add(self, **signal: object) -> None:
        """Store one Signal; keys are the column names, Outcome and scored-at left null."""
        columns = ", ".join(signal)
        marks = ", ".join(f":{k}" for k in signal)
        with self._open() as conn:
            conn.execute(f"INSERT INTO signals ({columns}) VALUES ({marks})", signal)

    def score(self, ticker: str, interval: str, history: pd.DataFrame, scored_at: str) -> None:
        """Give pending Signals for the ticker and Interval an Outcome from the bars after their own.

        A Signal whose bar isn't in history, or without enough later bars or a default plan, stays pending.
        """
        with self._open() as conn:
            pending = conn.execute(
                "SELECT id, data_date, verdict, long_stop, long_target, short_stop, short_target FROM signals "
                "WHERE ticker = ? AND interval = ? AND outcome IS NULL",
                (ticker, interval),
            ).fetchall()
            for id_, data_date, verdict, long_stop, long_target, short_stop, short_target in pending:
                bar = pd.Timestamp(data_date)
                if bar not in history.index:
                    continue
                later = cast("pd.DataFrame", history[history.index > bar])
                if verdict == "hold":
                    outcome = _hold_outcome(later, long_target, short_target)
                elif verdict == "buy":
                    outcome = _outcome(later, long_stop, long_target, long=True)
                else:
                    outcome = _outcome(later, short_stop, short_target, long=False)
                if outcome is not None:
                    conn.execute(
                        "UPDATE signals SET outcome = ?, scored_at = ? WHERE id = ?", (outcome, scored_at, id_)
                    )

    def hit_rate(self, ticker: str, interval: str, prompt_version: int, verdict: str) -> float | None:
        """Hit Rate of this prompt version's Signals of the verdict's kind; None under HIT_RATE_MIN_SCORED.

        Buy/sell: Targets / (Targets + Stops). Hold: Quiet / scored holds.
        """
        hold = verdict == "hold"
        hits, misses = ("quiet", "missed") if hold else ("target", "stop")
        with self._open() as conn:
            right, wrong = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
                "FROM signals WHERE ticker = ? AND interval = ? AND prompt_version = ? "
                f"AND verdict {'=' if hold else '!='} 'hold'",
                (hits, misses, ticker, interval, prompt_version),
            ).fetchone()
        if right + wrong < HIT_RATE_MIN_SCORED:
            return None
        return right / (right + wrong)

    def min_confidence(self, prompt_version: int, break_even: float) -> float | None:
        """Lowest 0.1 Confidence band of this prompt version whose buy/sell Hit Rate is above break_even.

        Only bands with BAND_MIN_SCORED Targets plus Stops count: MIN_CONFIDENCE while none has that many,
        None when some do but none is above break_even.
        """
        with self._open() as conn:
            bands = conn.execute(
                # The epsilon keeps 0.6 in band 6 despite float error.
                "SELECT CAST(confidence * 10 + 1e-9 AS INTEGER) AS band, "
                "COUNT(*) FILTER (WHERE outcome = 'target'), COUNT(*) FILTER (WHERE outcome = 'stop') "
                "FROM signals WHERE prompt_version = ? AND verdict IN ('buy', 'sell') GROUP BY band ORDER BY band",
                (prompt_version,),
            ).fetchall()
        for band, targets, stops in bands:
            if targets + stops >= BAND_MIN_SCORED and targets / (targets + stops) > break_even:
                return band / 10
        if any(targets + stops >= BAND_MIN_SCORED for _, targets, stops in bands):
            return None
        return MIN_CONFIDENCE

    def cut_loss_hit_rate(self, prompt_version: int) -> float | None:
        """Targets / (Targets + Stops) over this prompt version's Signals stored with cut_loss = 1, all tickers and
        Intervals; None under CUT_LOSS_MIN_SCORED."""
        with self._open() as conn:
            targets, stops = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE outcome = 'target'), COUNT(*) FILTER (WHERE outcome = 'stop') "
                "FROM signals WHERE prompt_version = ? AND cut_loss = 1",
                (prompt_version,),
            ).fetchone()
        if targets + stops < CUT_LOSS_MIN_SCORED:
            return None
        return targets / (targets + stops)

    def latest(self, ticker: str, interval: str) -> dict | None:
        """The most recent Signal for the ticker and Interval as the classifier sees it, or None."""
        with self._open() as conn:
            row = conn.execute(
                "SELECT verdict, confidence, outcome FROM signals WHERE ticker = ? AND interval = ? "
                "ORDER BY id DESC LIMIT 1",
                (ticker, interval),
            ).fetchone()
        if row is None:
            return None
        verdict, confidence, outcome = row
        return {"verdict": verdict, "confidence": confidence, "outcome": outcome or "pending"}


def _outcome(later: pd.DataFrame, stop: float | None, target: float | None, *, long: bool) -> str | None:
    """Target, Stop (also when both fall in one bar) or Expired within OUTCOME_BARS; None while pending."""
    if stop is None or target is None:
        return None
    for high, low in zip(later["High"].iloc[:OUTCOME_BARS], later["Low"].iloc[:OUTCOME_BARS], strict=True):
        if (low <= stop) if long else (high >= stop):
            return "stop"
        if (high >= target) if long else (low <= target):
            return "target"
    return "expired" if len(later) >= OUTCOME_BARS else None


def _hold_outcome(later: pd.DataFrame, long_target: float | None, short_target: float | None) -> str | None:
    """Missed if either default Target is reached within OUTCOME_BARS, else Quiet; None while pending."""
    if long_target is None or short_target is None:
        return None
    window = later.iloc[:OUTCOME_BARS]
    if (window["High"] >= long_target).any() or (window["Low"] <= short_target).any():
        return "missed"
    return "quiet" if len(later) >= OUTCOME_BARS else None
