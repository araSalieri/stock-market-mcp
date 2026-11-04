"""Signal log: one row per fresh classifier answer in the server database (ADR-0003), and the Trade Plan maths, default plan included."""

import math
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import cast

import pandas as pd

from stock_market_mcp.bars import BAR_LENGTH, EMA_LENGTH, Interval, finite, last_value
from stock_market_mcp.prompt import PROMPT_VERSION, PreviousSignal
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

# Re-entry Lock per ticker and Interval: the confident cut's Price, data date and Buyback Price (its mirrored default
# Target, null when that plan can't be made); lifted stays 1 until the next such cut.
# lifted_date is the lifting bar's date and lifted_by buyback or re_entry; both null on rows lifted before they existed.
LOCK_SCHEMA = """CREATE TABLE IF NOT EXISTS reentry_locks (
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    cut_price REAL NOT NULL,
    cut_date TEXT NOT NULL,
    buyback_price REAL,
    lifted INTEGER NOT NULL,
    lifted_date TEXT,
    lifted_by TEXT,
    PRIMARY KEY (ticker, interval)
)"""
# Columns added to tables after their first release, as table -> (column, type); missing ones are added on open.
_ADDED_COLUMNS = {
    "signals": (("cut_loss", "INTEGER"),),
    "reentry_locks": (("lifted_date", "TEXT"), ("lifted_by", "TEXT"), ("buyback_price", "REAL")),
}


class SignalLog:
    """get_signal calls before_classifier then after_classifier once per Signal, cached or fresh; between them they keep these guarantees:

    - Pending Signals are scored before any Hit Rate is read, so a Hit Rate never misses an Outcome the current bars give.
    - The Minimum Confidence is read per call, so a cached Signal gets the current one, and is applied once to the Trade Plan, the lock and cut_loss.
    - The Re-entry Lock is checked on cached Signals too, so a cached buy can't slip past a lock or miss a lift.
    - The Trade Plan is built before anything is written, and the reads, the lift, the record and a new lock share one transaction, so an error leaves nothing half-written.
    - Only a fresh Signal is recorded, so one classifier answer is one row; a lock can't lift on its own bar.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = path

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A transaction on the signals and reentry_locks tables, adding columns a file from before them lacks (null on old
        rows, except an unlifted lock's buyback_price)."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute(LOCK_SCHEMA)
            for table, columns in _ADDED_COLUMNS.items():
                have = {column[1] for column in conn.execute(f"PRAGMA table_info({table})")}
                for name, type_ in columns:
                    # shortcut: two servers upgrading one file at once can fail one call on a duplicate column; retry succeeds.
                    if name not in have:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {type_}")
                        if name == "buyback_price":
                            _backfill_buyback_prices(conn)
            yield conn

    def before_classifier(
        self, ticker: str, interval: str, history: pd.DataFrame, scored_at: str
    ) -> PreviousSignal | None:
        """Score pending Signals for the ticker and Interval, then return the previous Signal's Verdict, Confidence and Outcome (None while pending), or None."""
        with self._open() as conn:
            _score(conn, ticker, interval, history, scored_at)
            row = conn.execute(
                "SELECT verdict, confidence, outcome FROM signals WHERE ticker = ? AND interval = ? "
                "ORDER BY id DESC LIMIT 1",
                (ticker, interval),
            ).fetchone()
        return None if row is None else PreviousSignal(*row)

    def after_classifier(
        self,
        ticker: str,
        interval: Interval,
        history: pd.DataFrame,
        price: float,
        signal: dict,
        *,
        now: datetime,
        fresh: bool,
        losing: bool | None,
        atr_multiple: float,
        reward_ratio: float,
    ) -> dict:
        """The reply fields for a Signal made on history's last bar: hit_rate, min_confidence, trade_plan, re_entry_lock once
        a confident cut has made one (locked or lifted), and cut_loss and cut_loss_hit_rate when held (losing not None).

        A fresh Signal is recorded with both default Trade Plans at price and its cut_loss: a cut on a losing Position at
        any Confidence, null when not held. A confident cut also starts a Re-entry Lock at price, replacing any earlier one.
        """
        verdict = signal["verdict"]
        cut_loss = None if losing is None else verdict == "cut" and losing
        with self._open() as conn:
            min_confidence = _min_confidence(conn)
            hit_rate = _hit_rate(conn, ticker, interval, verdict)
            confident = min_confidence is not None and signal["confidence"] >= min_confidence
            lock = _reentry_lock(conn, ticker, interval, history, now)
            buys = confident and verdict == "buy" and (lock is None or lock["state"] == "lifted")
            plan = _trade_plan(history, price, atr_multiple, reward_ratio) if buys else None
            if fresh:
                _record(conn, ticker, interval, signal, history, price, cut_loss)
                if confident and verdict == "cut":
                    _, buyback_price = _default_plan(history, price, long=False)
                    conn.execute(
                        "INSERT OR REPLACE INTO reentry_locks (ticker, interval, cut_price, cut_date, buyback_price, lifted) "
                        "VALUES (?, ?, ?, ?, ?, 0)",
                        (ticker, interval, price, signal["data_date"], buyback_price),
                    )
                    lock = _lock_view(price, signal["data_date"], buyback_price, history)
            fields = {"hit_rate": hit_rate, "min_confidence": min_confidence, "trade_plan": plan}
            if lock is not None:
                fields["re_entry_lock"] = lock
            if cut_loss is not None:
                fields |= {"cut_loss": confident and cut_loss, "cut_loss_hit_rate": _cut_loss_hit_rate(conn)}
        return fields

    def track_record(self, ticker: str | None, interval: str | None) -> dict:
        """The Track Record's bands, with the server-wide Minimum Confidence and Cut Loss Hit Rate get_signal reports."""
        with self._open() as conn:
            return {
                "bands": _bands(conn, ticker, interval),
                "min_confidence": _min_confidence(conn),
                "cut_loss_hit_rate": _cut_loss_hit_rate(conn),
            }


def _backfill_buyback_prices(conn: sqlite3.Connection) -> None:
    """Give unlifted locks from before buyback_price their cut Signal's mirrored default Target."""
    conn.execute(
        "UPDATE reentry_locks SET buyback_price = (SELECT short_target FROM signals s "
        "WHERE s.ticker = reentry_locks.ticker AND s.interval = reentry_locks.interval "
        "AND s.data_date = reentry_locks.cut_date AND s.verdict = 'cut' ORDER BY s.id DESC LIMIT 1) "
        "WHERE lifted = 0"
    )


def _record(
    conn: sqlite3.Connection,
    ticker: str,
    interval: str,
    signal: dict,
    history: pd.DataFrame,
    price: float,
    cut_loss: bool | None,
) -> None:
    """Store one pending Signal at its data_date, with both default Trade Plans at price."""
    long_stop, long_target = _default_plan(history, price, long=True)
    short_stop, short_target = _default_plan(history, price, long=False)
    conn.execute(
        "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, "
        "long_stop, long_target, short_stop, short_target, cut_loss) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            ticker,
            interval,
            PROMPT_VERSION,
            signal["data_date"],
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


def _reentry_lock(
    conn: sqlite3.Connection, ticker: str, interval: Interval, history: pd.DataFrame, now: datetime
) -> dict | None:
    """The Re-entry Lock on the ticker and Interval, or None when there's none; writes the lift when history shows one."""
    row = conn.execute(
        "SELECT cut_price, cut_date, buyback_price, lifted, lifted_date, lifted_by FROM reentry_locks "
        "WHERE ticker = ? AND interval = ?",
        (ticker, interval),
    ).fetchone()
    if row is None:
        return None
    cut_price, cut_date, buyback_price, lifted, lifted_date, lifted_by = row
    if not lifted:
        lift = find_lift(history, interval, now, cut_price, cut_date, buyback_price)
        if lift is None:
            return _lock_view(cut_price, cut_date, buyback_price, history)
        lifted_date, lifted_by = lift
        conn.execute(
            "UPDATE reentry_locks SET lifted = 1, lifted_date = ?, lifted_by = ? WHERE ticker = ? AND interval = ?",
            (lifted_date, lifted_by, ticker, interval),
        )
    return {
        "state": "lifted",
        "cut_price": cut_price,
        "cut_date": cut_date,
        "buyback_price": buyback_price,
        "lifted_date": lifted_date,
        "lifted_by": lifted_by,
    }


def find_lift(
    history: pd.DataFrame,
    interval: Interval,
    now: datetime,
    cut_price: float,
    cut_date: str,
    buyback_price: float | None,
) -> tuple[str, str] | None:
    """The first bar of history after the cut that lifts a Re-entry Lock, as (its date, "buyback" or "re_entry"), or None.

    Any bar, forming included, lifts by buyback when its low is at or below buyback_price; a completed bar lifts by
    re-entry when it closes above both its EMA (EMA_LENGTH bars) and cut_price. A bar doing both lifts by buyback.
    """
    index = cast("pd.DatetimeIndex", history.index)
    # tz_convert(None) gives naive UTC, for bars without a timezone.
    completed = index + BAR_LENGTH[interval] <= pd.Timestamp(now).tz_convert(index.tz)
    later = pd.Series(index > pd.Timestamp(cut_date), index=index)
    # A null Buyback Price (older or plan-less locks) never lifts by buyback.
    buyback = later & (history["Low"] <= (-math.inf if buyback_price is None else buyback_price))
    re_entry = later & completed & (history["Close"] > history[f"EMA_{EMA_LENGTH}"]) & (history["Close"] > cut_price)
    lifting = cast("pd.DataFrame", history[buyback | re_entry])
    if lifting.empty:
        return None
    date = cast("pd.Timestamp", lifting.index[0])
    return date.isoformat(), "buyback" if buyback[date] else "re_entry"


def _lock_view(cut_price: float, cut_date: str, buyback_price: float | None, history: pd.DataFrame) -> dict:
    """A locked Re-entry Lock as the reply shows it, with the Re-entry Price from the current EMA."""
    ema = finite(float(history[f"EMA_{EMA_LENGTH}"].iloc[-1]))
    return {
        "state": "locked",
        "cut_price": cut_price,
        "cut_date": cut_date,
        "buyback_price": buyback_price,
        "ema": ema,
        "re_entry_price": None if ema is None else max(cut_price, ema),
    }


def _score(conn: sqlite3.Connection, ticker: str, interval: str, history: pd.DataFrame, scored_at: str) -> None:
    """Give pending Signals for the ticker and Interval an Outcome from the bars after their own.

    A Signal whose bar isn't in history, or without enough later bars or a default plan, stays pending.
    """
    pending = conn.execute(
        f"SELECT id, data_date, verdict, {', '.join(_PLAN_COLUMNS)} FROM signals "
        "WHERE ticker = ? AND interval = ? AND outcome IS NULL",
        (ticker, interval),
    ).fetchall()
    for id_, data_date, verdict, *prices in pending:
        bar = pd.Timestamp(data_date)
        if bar not in history.index:
            continue
        later = cast("pd.DataFrame", history[history.index > bar])
        rule = _RULES[verdict]
        plan: dict[str, float | None] = dict(zip(_PLAN_COLUMNS, prices, strict=True))
        outcome = rule.outcome(later, *(plan[column] for column in rule.plan))
        if outcome is not None:
            conn.execute("UPDATE signals SET outcome = ?, scored_at = ? WHERE id = ?", (outcome, scored_at, id_))


def _hit_rate(conn: sqlite3.Connection, ticker: str, interval: str, verdict: str) -> float | None:
    """Hit Rate of the current prompt version's Signals with this verdict; None under HIT_RATE_MIN_SCORED."""
    rule = _RULES[verdict]
    right, wrong = conn.execute(
        "SELECT COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
        "FROM signals WHERE ticker = ? AND interval = ? AND prompt_version = ? AND verdict = ?",
        (rule.hit, rule.miss, ticker, interval, PROMPT_VERSION, verdict),
    ).fetchone()
    return _rate(right, wrong)


def _min_confidence(conn: sqlite3.Connection) -> float | None:
    """Lowest Confidence band (1 / BANDS_PER_CONFIDENCE wide) of the current prompt version whose buy Hit Rate is above BREAK_EVEN_HIT_RATE.

    Only bands with BAND_MIN_SCORED Targets plus Stops count: MIN_CONFIDENCE while none has that many,
    None when some do but none is above BREAK_EVEN_HIT_RATE.
    """
    buy = _RULES["buy"]
    bands = conn.execute(
        f"SELECT {BAND} AS band, COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
        "FROM signals WHERE prompt_version = ? AND verdict = 'buy' GROUP BY band ORDER BY band",
        (buy.hit, buy.miss, PROMPT_VERSION),
    ).fetchall()
    rates = [(band, _rate(targets, stops, BAND_MIN_SCORED)) for band, targets, stops in bands]
    scored = [(band, rate) for band, rate in rates if rate is not None]
    for band, rate in scored:
        if rate > BREAK_EVEN_HIT_RATE:
            return band / BANDS_PER_CONFIDENCE
    return None if scored else MIN_CONFIDENCE


def _bands(conn: sqlite3.Connection, ticker: str | None, interval: str | None) -> list[dict]:
    """Outcome counts and Hit Rates of the current prompt version's Signals per Confidence band (1 / BANDS_PER_CONFIDENCE wide) and Verdict,
    optionally for one ticker and/or Interval, bands with Signals only, ascending.

    Leaves out Signals that can never be scored: those missing a price their Outcome rule is judged by.
    """
    scorable = " OR ".join(
        f"(verdict = '{verdict}' AND {' AND '.join(f'{column} IS NOT NULL' for column in _RULES[verdict].plan)})"
        for verdict in _TRACKED
    )
    rows = conn.execute(
        f"SELECT {BAND} AS band, verdict, COALESCE(outcome, 'pending'), COUNT(*) FROM signals "
        "WHERE prompt_version = ? AND (? IS NULL OR ticker = ?) AND (? IS NULL OR interval = ?) "
        f"AND ({scorable}) GROUP BY band, verdict, outcome ORDER BY band",
        (PROMPT_VERSION, ticker, ticker, interval, interval),
    ).fetchall()
    tallies: dict[int, dict[str, dict[str, int]]] = {}
    for band, verdict, outcome, count in rows:
        tallies.setdefault(band, {v: {} for v in _TRACKED})[verdict][outcome] = count
    buy = _RULES["buy"]
    return [
        {
            "band": band / BANDS_PER_CONFIDENCE,
            **{verdict: _counts(verdict, tally[verdict]) for verdict in _TRACKED},
            "buy_hit_rate": _rate(tally["buy"].get(buy.hit, 0), tally["buy"].get(buy.miss, 0), BAND_MIN_SCORED),
        }
        for band, tally in tallies.items()
    ]


def _cut_loss_hit_rate(conn: sqlite3.Connection) -> float | None:
    """Targets / (Targets + Stops) over the current prompt version's Signals stored with cut_loss = 1, all tickers and
    Intervals; None under CUT_LOSS_MIN_SCORED."""
    cut = _RULES["cut"]
    targets, stops = conn.execute(
        "SELECT COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
        "FROM signals WHERE prompt_version = ? AND cut_loss = 1",
        (cut.hit, cut.miss, PROMPT_VERSION),
    ).fetchone()
    return _rate(targets, stops, CUT_LOSS_MIN_SCORED)


def _rate(hits: int, misses: int, min_scored: int = HIT_RATE_MIN_SCORED) -> float | None:
    """hits / (hits + misses); None under min_scored."""
    if hits + misses < min_scored:
        return None
    return hits / (hits + misses)


def _counts(verdict: str, tally: dict[str, int]) -> dict:
    """One Verdict's Outcome counts and Hit Rate in a Track Record band."""
    rule = _RULES[verdict]
    counts = {outcome: tally.get(outcome, 0) for outcome in (rule.hit, rule.miss, *rule.other, "pending")}
    return {**counts, "hit_rate": _rate(counts[rule.hit], counts[rule.miss])}


def _trade_plan(
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
        plan = _trade_plan(history, entry, PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO, long=long)
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


@dataclass(frozen=True)
class _Rule:
    """How one Verdict is scored: the stored prices passed to outcome, and which Outcomes are hits, misses or neither."""

    plan: tuple[str, ...]
    outcome: Callable[..., str | None]
    hit: str
    miss: str
    other: tuple[str, ...] = ()


# long_* is the default Trade Plan, short_* the same distances mirrored, which a cut is judged by.
_PLAN_COLUMNS = ("long_stop", "long_target", "short_stop", "short_target")
_CUT = _Rule(("short_stop", "short_target"), partial(_outcome, long=False), "target", "stop", ("expired",))
_RULES = {
    "buy": _Rule(("long_stop", "long_target"), partial(_outcome, long=True), "target", "stop", ("expired",)),
    "cut": _CUT,
    "hold": _Rule(("long_target", "short_target"), _hold_outcome, "quiet", "missed"),
    # Older prompt versions' sell exited the long, as cut does now.
    "sell": _CUT,
}
# Verdicts the current prompt version gives, each reported in a Track Record band.
_TRACKED = ("buy", "cut", "hold")
