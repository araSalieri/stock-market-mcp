"""Portfolio storage: one row per Position in the server database (ADR-0003, ADR-0004)."""

import os

from stock_market_mcp.db import transaction

SCHEMA = """CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    direction TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL
)"""


class PortfolioStore:
    def __init__(self, path: str | os.PathLike):
        self.path = path

    def load(self) -> dict[str, dict]:
        """Every Position as {ticker: {direction, quantity, avg_cost}}, by ticker."""
        with transaction(self.path, SCHEMA) as conn:
            rows = conn.execute("SELECT ticker, direction, quantity, avg_cost FROM positions ORDER BY ticker")
            return {t: {"direction": d, "quantity": q, "avg_cost": c} for t, d, q, c in rows}

    def set(self, ticker: str, direction: str, quantity: float, avg_cost: float) -> None:
        """Replace the ticker's Position."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute("INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?)", (ticker, direction, quantity, avg_cost))

    def delete(self, ticker: str) -> None:
        """Remove the ticker's Position; one not held is fine."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
