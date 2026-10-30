"""Watchlist storage: one row per ticker in the server database (ADR-0002, ADR-0003)."""

import os

from stock_market_mcp.db import transaction

# Keeps one Scan within what free Yahoo tolerates (ADR-0002).
WATCHLIST_MAX_TICKERS = 15

SCHEMA = """CREATE TABLE IF NOT EXISTS watchlist_tickers (
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    PRIMARY KEY (name, position)
)"""


class WatchlistStore:
    def __init__(self, path: str | os.PathLike):
        self.path = path

    def load(self) -> dict[str, list[str]]:
        lists: dict[str, list[str]] = {}
        with transaction(self.path, SCHEMA) as conn:
            rows = conn.execute("SELECT name, ticker FROM watchlist_tickers ORDER BY name, position")
            for name, ticker in rows:
                lists.setdefault(name, []).append(ticker)
        return lists

    def set(self, name: str, tickers: list[str]) -> None:
        """Replace one Watchlist in one transaction; an empty list deletes it."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute("DELETE FROM watchlist_tickers WHERE name = ?", (name,))
            conn.executemany(
                "INSERT INTO watchlist_tickers VALUES (?, ?, ?)",
                [(name, i, t) for i, t in enumerate(tickers)],
            )
