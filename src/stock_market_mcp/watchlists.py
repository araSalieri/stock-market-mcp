"""Watchlist storage: one row per ticker in the server database (ADR-0002, ADR-0003)."""

import os
import re

from stock_market_mcp.db import transaction
from stock_market_mcp.errors import InputError

# Keeps one Scan within what free Yahoo tolerates (ADR-0002).
WATCHLIST_MAX_TICKERS = 15

SCHEMA = """CREATE TABLE IF NOT EXISTS watchlist_tickers (
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    PRIMARY KEY (name, position)
)"""


def normalise_tickers(tickers: list[str]) -> list[str]:
    """Upper-cased and de-duplicated in order; one Scan or Watchlist takes at most WATCHLIST_MAX_TICKERS."""
    names = list(dict.fromkeys(t.upper() for t in tickers))
    if len(names) > WATCHLIST_MAX_TICKERS:
        raise InputError(f"at most {WATCHLIST_MAX_TICKERS} tickers, got {len(names)}")
    return names


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
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):
            raise InputError("Watchlist name must be 1-32 letters, digits, '-' or '_'")
        tickers = normalise_tickers(tickers)
        with transaction(self.path, SCHEMA) as conn:
            conn.execute("DELETE FROM watchlist_tickers WHERE name = ?", (name,))
            conn.executemany(
                "INSERT INTO watchlist_tickers VALUES (?, ?, ?)",
                [(name, i, t) for i, t in enumerate(tickers)],
            )
