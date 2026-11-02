"""Portfolio storage: one row per Position in the server database (ADR-0003, ADR-0004)."""

import math
import os
from dataclasses import dataclass

from stock_market_mcp.db import transaction

SCHEMA = """CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,
    direction TEXT NOT NULL,
    quantity REAL NOT NULL,
    avg_cost REAL NOT NULL
)"""


def _finite(value: float) -> float | None:
    """JSON has no NaN; a non-finite amount is null."""
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class Position:
    direction: str
    quantity: float
    avg_cost: float

    def unrealised_pnl(self, price: float) -> dict:
        """Unrealised P&L at price, as an amount and a percent of cost; null when non-finite."""
        sign = 1 if self.direction == "long" else -1
        pnl = sign * (price - self.avg_cost) * self.quantity
        return {
            "unrealised_pnl": _finite(pnl),
            "unrealised_pnl_pct": _finite(pnl / (self.avg_cost * self.quantity) * 100),
        }

    def classifier_view(self, price: float) -> dict:
        """The Position as the classifier sees it: quantity stays out, so size doesn't sway the Verdict or bust the cache (ADR-0005)."""
        return {
            "direction": self.direction,
            "avg_cost": self.avg_cost,
            "unrealised_pnl_pct": self.unrealised_pnl(price)["unrealised_pnl_pct"],
        }

    def opposes_losing(self, verdict: str, price: float) -> bool:
        """Whether the Verdict points against this Position while it loses at price; break-even and null P&L don't lose."""
        pnl = self.unrealised_pnl(price)["unrealised_pnl"]
        return verdict == {"long": "sell", "short": "buy"}[self.direction] and pnl is not None and pnl < 0


class PortfolioStore:
    def __init__(self, path: str | os.PathLike):
        self.path = path

    def load(self) -> dict[str, Position]:
        """Every Position by ticker."""
        with transaction(self.path, SCHEMA) as conn:
            rows = conn.execute("SELECT ticker, direction, quantity, avg_cost FROM positions ORDER BY ticker")
            return {t: Position(d, q, c) for t, d, q, c in rows}

    def set(self, ticker: str, direction: str, quantity: float, avg_cost: float) -> None:
        """Replace the ticker's Position."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute("INSERT OR REPLACE INTO positions VALUES (?, ?, ?, ?)", (ticker, direction, quantity, avg_cost))

    def delete(self, ticker: str) -> None:
        """Remove the ticker's Position; one not held is fine."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute("DELETE FROM positions WHERE ticker = ?", (ticker,))
