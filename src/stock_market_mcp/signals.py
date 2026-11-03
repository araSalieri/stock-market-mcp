"""Signal log: one row per fresh classifier answer in the server database (ADR-0003), and the Trade Plan maths, default plan included."""

import math
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

import pandas as pd

from stock_market_mcp.bars import EMA_LENGTH, finite, last_value
from stock_market_mcp.classifier import PROMPT_VERSION
from stock_market_mcp.db import transaction
from stock_market_mcp.errors import InputError

PLAN_ATR_MULTIPLE = 2.0
PLAN_REWARD_RATIO = 2.0
# Hit Rate a default Trade Plan must beat to break even: a Target gains PLAN_REWARD_RATIO times what a Stop loses.
BREAK_EVEN_HIT_RATE = 1 / (1 + PLAN_REWARD_RATIO)

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
# Confidence bands are 1 / BANDS_PER_CONFIDENCE wide: 0.1.
BANDS_PER_CONFIDENCE = 10
# Confidence band as an integer (6 for 0.6-0.69); the epsilon keeps 0.6 in band 6 despite float error.
BAND = f"CAST(confidence * {BANDS_PER_CONFIDENCE} + 1e-9 AS INTEGER)"

# Outcome is null while pending; plan prices are the default Trade Plans, null when one has non-positive prices.
# short_stop and short_target are the default plan's distances mirrored above and below the Price, which score a cut.
# cut_loss: 1 when the Verdict was cut on a losing Position at any Confidence, 0 when held otherwise, null when not held.
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

# Re-entry Lock per ticker and Interval: the confident cut's Price and data date; lifted stays 1 until the next such cut.
LOCK_SCHEMA = """CREATE TABLE IF NOT EXISTS reentry_locks (
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    cut_price REAL NOT NULL,
    cut_date TEXT NOT NULL,
    lifted INTEGER NOT NULL,
    PRIMARY KEY (ticker, interval)
)"""


class SignalLog:
    def __init__(self, path: str | os.PathLike):
        self.path = path

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A transaction on the signals table, adding cut_loss to a file from before it existed (null on old rows)."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute(LOCK_SCHEMA)
            # shortcut: two servers upgrading one file at once can fail one call on a duplicate column; retry succeeds.
            if "cut_loss" not in {column[1] for column in conn.execute("PRAGMA table_info(signals)")}:
                conn.execute("ALTER TABLE signals ADD COLUMN cut_loss INTEGER")
            yield conn

    def record(
        self,
        ticker: str,
        interval: str,
        signal: dict,
        history: pd.DataFrame,
        price: float,
        cut_loss: bool | None,
        lock: bool = False,
    ) -> None:
        """Store one pending Signal made on history's last bar, with both default Trade Plans at price.

        lock=True also starts a Re-entry Lock at price in the same transaction, replacing any earlier one.
        """
        data_date = cast("pd.Timestamp", history.index[-1]).isoformat()
        long_stop, long_target = _default_plan(history, price, long=True)
        short_stop, short_target = _default_plan(history, price, long=False)
        with self._open() as conn:
            conn.execute(
                "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, "
                "long_stop, long_target, short_stop, short_target, cut_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    interval,
                    PROMPT_VERSION,
                    data_date,
                    signal["verdict"],
                    signal["confidence"],
                    price,
                    long_stop,
                    long_target,
                    short_stop,
                    short_target,
                    cut_loss,
                ),
            )
            if lock:
                conn.execute(
                    "INSERT OR REPLACE INTO reentry_locks VALUES (?, ?, ?, ?, 0)", (ticker, interval, price, data_date)
                )

    def reentry_lock(self, ticker: str, interval: str, history: pd.DataFrame) -> dict | None:
        """The Re-entry Lock on the ticker and Interval as {cut_price, cut_date, ema}, or None when there's none.

        A lock lifts, for good, once a bar of history after the cut closes above both the EMA (EMA_LENGTH bars) and the cut's Price.
        """
        with self._open() as conn:
            row = conn.execute(
                "SELECT cut_price, cut_date FROM reentry_locks WHERE ticker = ? AND interval = ? AND lifted = 0",
                (ticker, interval),
            ).fetchone()
            if row is None:
                return None
            cut_price, cut_date = row
            later = cast("pd.DataFrame", history[history.index > pd.Timestamp(cut_date)])
            if ((later["Close"] > later[f"EMA_{EMA_LENGTH}"]) & (later["Close"] > cut_price)).any():
                conn.execute(
                    "UPDATE reentry_locks SET lifted = 1 WHERE ticker = ? AND interval = ?", (ticker, interval)
                )
                return None
        return {"cut_price": cut_price, "cut_date": cut_date, "ema": finite(float(history[f"EMA_{EMA_LENGTH}"].iloc[-1]))}

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
                else:  # cut, and sell from older prompt versions
                    outcome = _outcome(later, short_stop, short_target, long=False)
                if outcome is not None:
                    conn.execute(
                        "UPDATE signals SET outcome = ?, scored_at = ? WHERE id = ?", (outcome, scored_at, id_)
                    )

    def hit_rate(self, ticker: str, interval: str, verdict: str) -> float | None:
        """Hit Rate of the current prompt version's Signals of the verdict's kind; None under HIT_RATE_MIN_SCORED.

        Buy/cut: Targets / (Targets + Stops). Hold: Quiet / scored holds.
        """
        hold = verdict == "hold"
        hits, misses = ("quiet", "missed") if hold else ("target", "stop")
        with self._open() as conn:
            right, wrong = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
                "FROM signals WHERE ticker = ? AND interval = ? AND prompt_version = ? "
                f"AND verdict {'=' if hold else '!='} 'hold'",
                (hits, misses, ticker, interval, PROMPT_VERSION),
            ).fetchone()
        return _rate(right, wrong)

    def min_confidence(self) -> float | None:
        """Lowest 0.1 Confidence band of the current prompt version whose buy Hit Rate is above BREAK_EVEN_HIT_RATE.

        Only bands with BAND_MIN_SCORED Targets plus Stops count: MIN_CONFIDENCE while none has that many,
        None when some do but none is above BREAK_EVEN_HIT_RATE.
        """
        with self._open() as conn:
            bands = conn.execute(
                f"SELECT {BAND} AS band, "
                "COUNT(*) FILTER (WHERE outcome = 'target'), COUNT(*) FILTER (WHERE outcome = 'stop') "
                "FROM signals WHERE prompt_version = ? AND verdict = 'buy' GROUP BY band ORDER BY band",
                (PROMPT_VERSION,),
            ).fetchall()
        rates = [(band, _rate(targets, stops, BAND_MIN_SCORED)) for band, targets, stops in bands]
        scored = [(band, rate) for band, rate in rates if rate is not None]
        for band, rate in scored:
            if rate > BREAK_EVEN_HIT_RATE:
                return band / BANDS_PER_CONFIDENCE
        return None if scored else MIN_CONFIDENCE

    def track_record(self, ticker: str | None, interval: str | None) -> list[dict]:
        """Outcome counts and Hit Rates of the current prompt version's Signals per 0.1 Confidence band and Verdict,
        optionally for one ticker and/or Interval, bands with Signals only, ascending.

        Leaves out Signals score never scores: buy without a long default plan, cut without a mirrored one,
        hold without both.
        """
        with self._open() as conn:
            rows = conn.execute(
                f"SELECT {BAND} AS band, verdict, "
                "COUNT(*) FILTER (WHERE outcome = 'target'), COUNT(*) FILTER (WHERE outcome = 'stop'), "
                "COUNT(*) FILTER (WHERE outcome = 'expired'), COUNT(*) FILTER (WHERE outcome = 'quiet'), "
                "COUNT(*) FILTER (WHERE outcome = 'missed'), COUNT(*) FILTER (WHERE outcome IS NULL) "
                "FROM signals WHERE prompt_version = ? AND (? IS NULL OR ticker = ?) AND (? IS NULL OR interval = ?) "
                # Plan prices are null together, so the Target stands for the whole plan.
                "AND CASE verdict WHEN 'buy' THEN long_target IS NOT NULL WHEN 'cut' THEN short_target IS NOT NULL "
                "ELSE long_target IS NOT NULL AND short_target IS NOT NULL END "
                "GROUP BY band, verdict ORDER BY band",
                (PROMPT_VERSION, ticker, ticker, interval, interval),
            ).fetchall()
        bands: dict[int, dict] = {}
        for band, verdict, *counts in rows:
            entry = bands.setdefault(
                band, {"band": band / BANDS_PER_CONFIDENCE, **{v: _counts(v) for v in ("buy", "cut", "hold")}}
            )
            entry[verdict] = _counts(verdict, *counts)
        for entry in bands.values():
            entry["buy_hit_rate"] = _rate(entry["buy"]["target"], entry["buy"]["stop"], BAND_MIN_SCORED)
        return list(bands.values())

    def cut_loss_hit_rate(self) -> float | None:
        """Targets / (Targets + Stops) over the current prompt version's Signals stored with cut_loss = 1, all tickers and
        Intervals; None under CUT_LOSS_MIN_SCORED."""
        with self._open() as conn:
            targets, stops = conn.execute(
                "SELECT COUNT(*) FILTER (WHERE outcome = 'target'), COUNT(*) FILTER (WHERE outcome = 'stop') "
                "FROM signals WHERE prompt_version = ? AND cut_loss = 1",
                (PROMPT_VERSION,),
            ).fetchone()
        return _rate(targets, stops, CUT_LOSS_MIN_SCORED)

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


def _rate(hits: int, misses: int, min_scored: int = HIT_RATE_MIN_SCORED) -> float | None:
    """hits / (hits + misses); None under min_scored."""
    if hits + misses < min_scored:
        return None
    return hits / (hits + misses)


def _counts(
    verdict: str, target: int = 0, stop: int = 0, expired: int = 0, quiet: int = 0, missed: int = 0, pending: int = 0
) -> dict:
    """One Verdict's Outcome counts and Hit Rate in a Track Record band."""
    if verdict == "hold":
        return {"quiet": quiet, "missed": missed, "pending": pending, "hit_rate": _rate(quiet, missed)}
    return {"target": target, "stop": stop, "expired": expired, "pending": pending, "hit_rate": _rate(target, stop)}


def trade_plan(
    history: pd.DataFrame, entry: float, atr_multiple: float, reward_ratio: float, *, long: bool = True
) -> dict:
    """Stop and Target around entry (the Price) from the last ATR(14) of history; long=False mirrors them, to score a cut."""
    atr = last_value("atr", history)
    if not math.isfinite(atr):
        raise InputError(f"latest row has a non-finite ATR (atr={atr})")
    stop = entry - (1 if long else -1) * atr_multiple * atr
    # Same line both ways: mirrored Target = Entry - ratio * (Stop - Entry).
    target = entry + reward_ratio * (entry - stop)
    if stop <= 0 or target <= 0:
        raise InputError(
            f"plan has non-positive prices (stop={stop}, target={target}); "
            "ATR is too large relative to the last close"
        )
    return {
        "entry": entry,
        "stop": stop,
        "target": target,
        "atr": atr,
        "atr_multiple": atr_multiple,
        "reward_ratio": reward_ratio,
    }


def _default_plan(history: pd.DataFrame, entry: float, *, long: bool) -> tuple[float | None, float | None]:
    """Stop and Target of the default Trade Plan stored with a Signal, or mirrored; nulls when that plan can't be made."""
    try:
        plan = trade_plan(history, entry, PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO, long=long)
    except InputError:
        return None, None
    return plan["stop"], plan["target"]


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
