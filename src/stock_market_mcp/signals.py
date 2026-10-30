"""Signal log: one row per fresh classifier answer in the server database (ADR-0003)."""

import os

from stock_market_mcp.db import transaction

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
