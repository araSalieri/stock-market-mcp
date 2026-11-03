"""Portfolio storage: one row per Position in the server database (ADR-0003, ADR-0004)."""

import math
import os
from dataclasses import dataclass

from stock_market_mcp.bars import finite
from stock_market_mcp.db import transaction
from stock_market_mcp.errors import InputError
from stock_market_mcp.watchlists import WATCHLIST_MAX_TICKERS

# direction stays so older files keep working; new rows write long.
SCHEMA = """CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    direction TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL
)"""


@dataclass(frozen=True)
class Position:
    quantity: float
    avg_cost: float

    def unrealised_pnl(self, price: float) -> dict:
        """Unrealised P&L at price, as an amount and a percent of cost; null when non-finite."""
        pnl = (price - self.avg_cost) * self.quantity
        return {
            "unrealised_pnl": finite(pnl),
            "unrealised_pnl_pct": finite(pnl / (self.avg_cost * self.quantity) * 100),
        }

    def classifier_view(self, pnl: dict) -> dict:
        """The Position as the classifier sees it: quantity stays out, so size doesn't sway the Verdict or bust the cache (ADR-0005)."""
        return {
            "avg_cost": self.avg_cost,
            "unrealised_pnl_pct": pnl["unrealised_pnl_pct"],
        }

    def opposes_losing(self, verdict: str, pnl: dict) -> bool:
        """Whether the Verdict is sell while the Unrealised P&L loses; break-even and null don't lose."""
        amount = pnl["unrealised_pnl"]
        return verdict == "sell" and amount is not None and amount < 0


class PortfolioStore:
    def __init__(self, path: str | os.PathLike):
        self.path = path

    def load(self) -> dict[str, Position]:
        """Every Position by ticker; a short row from before long-only fails loudly, naming its ticker."""
        with transaction(self.path, SCHEMA) as conn:
            rows = conn.execute("SELECT ticker, direction, quantity, avg_cost FROM positions ORDER BY ticker").fetchall()
        shorts = [t for t, d, _, _ in rows if d != "long"]
        if shorts:
            raise InputError(f"short Positions {', '.join(shorts)}; clear each with set_position quantity 0")
        return {t: Position(q, c) for t, _, q, c in rows}

    def set(self, ticker: str, quantity: float, avg_cost: float) -> None:
        """Replace the ticker's Position; as many Positions as a Watchlist has tickers."""
        if not (math.isfinite(quantity) and quantity >= 0):
            raise InputError(f"quantity must be a finite number of at least 0, got {quantity}")
        # Separate from the check above, whose text set_position replies with; the tool deletes on 0 instead.
        if quantity == 0:
            raise InputError("quantity 0 is no Position; delete it instead")
        if not (math.isfinite(avg_cost) and avg_cost > 0):
            raise InputError(f"avg_cost must be a finite number above 0, got {avg_cost}")
        with transaction(self.path, SCHEMA) as conn:
            held = conn.execute("SELECT count(*), sum(ticker = ?) FROM positions", (ticker,)).fetchone()
            if not held[1] and held[0] >= WATCHLIST_MAX_TICKERS:
                raise InputError(f"at most {WATCHLIST_MAX_TICKERS} Positions; delete one before adding {ticker}")
            conn.execute("INSERT OR REPLACE INTO positions VALUES (?, 'long', ?, ?)", (ticker, quantity, avg_cost))

    def delete(self, ticker: str) -> None:
        """Remove the ticker's Position; one not held is fine."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
