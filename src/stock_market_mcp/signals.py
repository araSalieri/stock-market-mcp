"""Signal log: one row per fresh classifier answer in the server database (ADR-0003)."""

import os
from typing import cast

import pandas as pd

from stock_market_mcp.db import transaction

# Bars after a Signal's own bar within which it gets an Outcome.
OUTCOME_BARS = 20
# Targets plus Stops needed before a Hit Rate is reported.
HIT_RATE_MIN_SCORED = 10

# Outcome is null while pending; plan prices are the default Trade Plans, null when one has non-positive prices.
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
    scored_at TEXT
)"""


class SignalLog:
    def __init__(self, path: str | os.PathLike):
        self.path = path

    def add(self, **signal: object) -> None:
        """Store one Signal; keys are the column names, Outcome and scored-at left null."""
        columns = ", ".join(signal)
        marks = ", ".join(f":{k}" for k in signal)
        with transaction(self.path, SCHEMA) as conn:
            conn.execute(f"INSERT INTO signals ({columns}) VALUES ({marks})", signal)

    def score(self, ticker: str, interval: str, history: pd.DataFrame, scored_at: str) -> None:
        """Give pending buy/sell Signals for the ticker and Interval an Outcome from the bars after their own.

        A Signal whose bar isn't in history, or without enough later bars or a default plan, stays pending.
        """
        with transaction(self.path, SCHEMA) as conn:
            pending = conn.execute(
                "SELECT id, data_date, verdict, long_stop, long_target, short_stop, short_target FROM signals "
                "WHERE ticker = ? AND interval = ? AND outcome IS NULL AND verdict IN ('buy', 'sell')",
                (ticker, interval),
            ).fetchall()
            for id_, data_date, verdict, long_stop, long_target, short_stop, short_target in pending:
                bar = pd.Timestamp(data_date)
                if bar not in history.index:
                    continue
                later = cast("pd.DataFrame", history[history.index > bar])
                if verdict == "buy":
                    outcome = _outcome(later, long_stop, long_target, long=True)
                else:
                    outcome = _outcome(later, short_stop, short_target, long=False)
                if outcome is not None:
                    conn.execute(
                        "UPDATE signals SET outcome = ?, scored_at = ? WHERE id = ?", (outcome, scored_at, id_)
                    )

    def hit_rate(self, ticker: str, interval: str, prompt_version: int) -> float | None:
        """Targets / (Targets + Stops) over buy/sell Signals of this prompt version; None under HIT_RATE_MIN_SCORED."""
        with transaction(self.path, SCHEMA) as conn:
            targets, stops = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE outcome = 'target'), COUNT(*) FILTER (WHERE outcome = 'stop') "
                "FROM signals WHERE ticker = ? AND interval = ? AND prompt_version = ? AND verdict IN ('buy', 'sell')",
                (ticker, interval, prompt_version),
            ).fetchone()
        if targets + stops < HIT_RATE_MIN_SCORED:
            return None
        return targets / (targets + stops)

    def latest(self, ticker: str, interval: str) -> dict | None:
        """The most recent Signal for the ticker and Interval as the classifier sees it, or None."""
        with transaction(self.path, SCHEMA) as conn:
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
