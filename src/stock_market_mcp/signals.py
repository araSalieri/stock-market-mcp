"""Signal log: one row per fresh classifier answer in the server database (ADR-0003), and the Trade Plan maths, default plan included."""

import math
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import NamedTuple, cast

import pandas as pd

from stock_market_mcp.bars import BAR_LENGTH, EMA_LENGTH, Interval, finite, last_value
from stock_market_mcp.prompt import PROMPT_VERSION, PreviousSignal
from stock_market_mcp.db import transaction
from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PricedPosition

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
# Targets plus Stops among one arming path's tagged buys needed before its Re-entry Hit Rate is reported.
RE_ENTRY_MIN_SCORED = 30
# Minimum Confidence until a band has BAND_MIN_SCORED.
MIN_CONFIDENCE = 0.6
# Confidence bands are 1 / BANDS_PER_CONFIDENCE wide: 0.1.
BANDS_PER_CONFIDENCE = 10
# Confidence band as an integer (6 for 0.6-0.69); the epsilon keeps 0.6 in band 6 despite float error.
BAND = f"CAST(confidence * {BANDS_PER_CONFIDENCE} + 1e-9 AS INTEGER)"

# Outcome is null while pending; plan prices are the default Trade Plans, null when one has non-positive prices.
# cut_loss: 1 when the Verdict was cut on a losing Position at any Confidence, 0 when held otherwise, null when not held.
# re_entry: the arming path (buyback or re_entry) on the buy that lifted a Re-entry Lock; else null.
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
    cut_loss INTEGER,
    re_entry TEXT
)"""

# Re-entry Lock per ticker and Interval: the confident cut's Price, data date and Buyback Price (its mirrored default
# Target, null when that plan can't be made). armed_by (buyback or re_entry) and armed_date (the arming bar's date) are
# set by the first bar after the cut that arms it; lifted is set by the first fresh confident buy while armed, and stays 1
# until the next such cut. lifted_date is the lifting buy's data date and lifted_by its armed_by; locks lifted before
# arming existed have the lifting bar's date and path, or nulls on rows lifted before those existed.
# cut_quantity is the held Position's quantity at the cut; null when not held or on rows from before it existed.
LOCK_SCHEMA = """CREATE TABLE IF NOT EXISTS reentry_locks (
    ticker TEXT NOT NULL,
    interval TEXT NOT NULL,
    cut_price REAL NOT NULL,
    cut_date TEXT NOT NULL,
    buyback_price REAL,
    lifted INTEGER NOT NULL,
    lifted_date TEXT,
    lifted_by TEXT,
    cut_quantity REAL,
    armed_by TEXT,
    armed_date TEXT,
    PRIMARY KEY (ticker, interval)
)"""
# Columns added to tables after their first release, as table -> (column, type); missing ones are added on open.
_ADDED_COLUMNS = {
    "signals": (("cut_loss", "INTEGER"), ("re_entry", "TEXT")),
    "reentry_locks": (("lifted_date", "TEXT"), ("lifted_by", "TEXT"), ("buyback_price", "REAL"), ("cut_quantity", "REAL"),
        ("armed_by", "TEXT"), ("armed_date", "TEXT"),
    ),
}


class SignalLog:
    """get_signal calls before_classifier then after_classifier once per Signal, cached or fresh; between them they keep these guarantees:

    - Pending Signals are scored before any Hit Rate is read, so a Hit Rate never misses an Outcome the current bars give.
    - The Minimum Confidence is read per call, so a cached Signal gets the current one, and is applied once to the Trade Plan, the lock and cut_loss.
    - The Re-entry Lock is checked for arming on cached Signals too, so a cached buy can't slip past a locked lock or miss an armed one's Trade Plan.
    - The Trade Plan is built before anything is written, and the reads, the arming, the record, the lift and a new lock share one transaction, so an error leaves nothing half-written.
    - Only a fresh Signal is recorded, so one classifier answer is one row; only a fresh buy lifts, so each lift has its recorded buy.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = path

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A transaction on the signals and reentry_locks tables, adding columns a file from before them lacks (null on old
        rows, except an unlifted lock's buyback_price, and first-buy tags from before the armed state are cleared)."""
        with transaction(self.path, SCHEMA) as conn:
            conn.execute(LOCK_SCHEMA)
            added = set()
            for table, columns in _ADDED_COLUMNS.items():
                have = {column[1] for column in conn.execute(f"PRAGMA table_info({table})")}
                for name, type_ in columns:
                    # shortcut: two servers upgrading one file at once can fail one call on a duplicate column; retry succeeds.
                    if name not in have:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {type_}")
                        added.add(name)
            if "buyback_price" in added:
                _backfill_buyback_prices(conn)
            if "armed_by" in added:
                # Tags from the old price-only lift, at any Confidence, aren't buys that lifted an armed lock.
                conn.execute("UPDATE signals SET re_entry = NULL")
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
        held: PricedPosition | None,
        atr_multiple: float,
        reward_ratio: float,
    ) -> dict:
        """The reply fields for a Signal made on history's last bar: hit_rate, min_confidence, trade_plan, re_entry_lock and
        re_entry_hit_rate once a confident cut has made a lock (locked, armed or lifted), and cut_loss and cut_loss_hit_rate when
        held.

        A fresh Signal is recorded with both default Trade Plans at price and its cut_loss: a cut on a losing Position at
        any Confidence, null when not held. A fresh confident buy on an armed lock lifts it and is tagged with the arming
        path. A confident cut also starts a Re-entry Lock at price, replacing any earlier one.
        """
        verdict = signal["verdict"]
        cut_loss = None if held is None else verdict == "cut" and held.losing
        with self._open() as conn:
            min_confidence = _min_confidence(conn)
            hit_rate = _hit_rate(conn, ticker, interval, verdict)
            confident = min_confidence is not None and signal["confidence"] >= min_confidence
            lock = _reentry_lock(conn, ticker, interval, lambda: history, now, held=held is not None)
            gets_plan = confident and verdict == "buy" and (lock is None or lock["state"] != "locked")
            plan = _trade_plan(history, price, atr_multiple, reward_ratio) if gets_plan else None
            if fresh:
                # Only a recorded buy lifts, so the Re-entry Hit Rate counts each lifting buy.
                lifts = gets_plan and lock is not None and lock["state"] == "armed"
                re_entry = lock["armed_by"] if lock is not None and lifts else None
                _record(conn, ticker, interval, signal, history, price, cut_loss, re_entry)
                if lifts:
                    conn.execute(
                        "UPDATE reentry_locks SET lifted = 1, lifted_date = ?, lifted_by = armed_by "
                        "WHERE ticker = ? AND interval = ?",
                        (signal["data_date"], ticker, interval),
                    )
                    lock = _reentry_lock(conn, ticker, interval, lambda: history, now, held=held is not None)
                if confident and verdict == "cut":
                    _, buyback_price = _default_plan(history, price, long=False)
                    quantity = None if held is None else held.quantity
                    cut = _Cut(price, signal["data_date"], buyback_price, quantity)
                    conn.execute(
                        "INSERT OR REPLACE INTO reentry_locks "
                        "(ticker, interval, cut_price, cut_date, buyback_price, cut_quantity, lifted) "
                        "VALUES (?, ?, ?, ?, ?, ?, 0)",
                        (ticker, interval, *cut),
                    )
                    # Still held at the cut, so no Buyback Quantity.
                    lock = _lock_view("locked", cut, held=True, **_re_entry_fields(cut, history))
            fields = {"hit_rate": hit_rate, "min_confidence": min_confidence, "trade_plan": plan}
            if lock is not None:
                fields |= {"re_entry_lock": lock, "re_entry_hit_rate": _re_entry_hit_rate(conn)}
            if cut_loss is not None:
                fields |= {"cut_loss": confident and cut_loss, "cut_loss_hit_rate": _cut_loss_hit_rate(conn)}
        return fields

    def scan_locks(
        self, ticker: str, history: Callable[[Interval], pd.DataFrame], now: datetime, *, held: bool
    ) -> list[dict]:
        """Every Re-entry Lock on the ticker, one per Interval with its Interval, as get_signal shows it; writes any arming
        history(interval) shows, and never lifts (no Verdict). A locked lock whose history or arming check fails shows its
        stored state with stale and the error."""
        with self._open() as conn:
            rows = conn.execute(
                f"SELECT interval, {_CUT_COLUMNS} FROM reentry_locks WHERE ticker = ? ORDER BY interval", (ticker,)
            ).fetchall()
            locks = []
            for interval, *cut in rows:
                try:
                    lock = _reentry_lock(conn, ticker, interval, partial(history, interval), now, held=held)
                except Exception as exc:
                    lock = _lock_view("locked", _Cut(*cut), held=held, stale=True, error=str(exc))
                locks.append({"interval": interval, **cast("dict", lock)})
        return locks

    def track_record(self, ticker: str | None, interval: str | None) -> dict:
        """The Track Record's bands, with the server-wide Minimum Confidence, Cut Loss Hit Rate and Re-entry Hit Rate
        get_signal reports."""
        with self._open() as conn:
            return {
                "bands": _bands(conn, ticker, interval),
                "min_confidence": _min_confidence(conn),
                "cut_loss_hit_rate": _cut_loss_hit_rate(conn),
                "re_entry_hit_rate": _re_entry_hit_rate(conn),
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
    re_entry: str | None,
) -> None:
    """Store one pending Signal at its data_date, with both default Trade Plans at price."""
    long_stop, long_target = _default_plan(history, price, long=True)
    short_stop, short_target = _default_plan(history, price, long=False)
    conn.execute(
        "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, "
        "long_stop, long_target, short_stop, short_target, cut_loss, re_entry) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
            re_entry,
        ),
    )


def _reentry_lock(
    conn: sqlite3.Connection,
    ticker: str,
    interval: Interval,
    history: Callable[[], pd.DataFrame],
    now: datetime,
    *,
    held: bool,
) -> dict | None:
    """The Re-entry Lock on the ticker and Interval, or None when there's none; writes the arming when history() shows
    a bar that arms it. It never lifts: only a fresh confident buy does, in after_classifier.

    history is only called for a locked lock, so an armed or lifted one needs no fetch."""
    row = conn.execute(
        f"SELECT {_CUT_COLUMNS}, lifted, lifted_date, lifted_by, armed_by, armed_date FROM reentry_locks "
        "WHERE ticker = ? AND interval = ?",
        (ticker, interval),
    ).fetchone()
    if row is None:
        return None
    cut = _Cut(*row[:4])
    lifted, lifted_date, lifted_by, armed_by, armed_date = row[4:]
    if lifted:
        return _lock_view("lifted", cut, held=held, lifted_date=lifted_date, lifted_by=lifted_by)
    if armed_by is None:
        bars = history()
        arming = _find_arming(bars, interval, now, cut)
        if arming is None:
            return _lock_view("locked", cut, held=held, **_re_entry_fields(cut, bars))
        armed_date, armed_by = arming
        conn.execute(
            "UPDATE reentry_locks SET armed_date = ?, armed_by = ? WHERE ticker = ? AND interval = ?",
            (armed_date, armed_by, ticker, interval),
        )
    return _lock_view("armed", cut, held=held, armed_by=armed_by, armed_date=armed_date)


class _Cut(NamedTuple):
    """The confident cut a Re-entry Lock starts from; buyback_price and quantity are null when unknown."""

    price: float
    date: str
    buyback_price: float | None
    quantity: float | None


# A Re-entry Lock's _Cut fields, in order.
_CUT_COLUMNS = "cut_price, cut_date, buyback_price, cut_quantity"


def _lock_view(state: str, cut: _Cut, *, held: bool, **extra: object) -> dict:
    """A Re-entry Lock as the reply shows it, with extra after the cut's fields. While not held it adds the Buyback
    Quantity: whole shares the cut's proceeds buy at the Buyback Price, fees left out; absent while held (buying more
    then is averaging down) or when the cut quantity or Buyback Price is unknown."""
    view = {"state": state, "cut_price": cut.price, "cut_date": cut.date, "buyback_price": cut.buyback_price, **extra}
    if not held and cut.quantity is not None and cut.buyback_price is not None:
        view["buyback_quantity"] = math.floor(cut.quantity * cut.price / cut.buyback_price)
    return view


def _re_entry_fields(cut: _Cut, history: pd.DataFrame) -> dict:
    """A locked Re-entry Lock's current EMA and Re-entry Price."""
    ema = finite(float(history[f"EMA_{EMA_LENGTH}"].iloc[-1]))
    return {"ema": ema, "re_entry_price": None if ema is None else max(cut.price, ema)}


def _find_arming(history: pd.DataFrame, interval: Interval, now: datetime, cut: _Cut) -> tuple[str, str] | None:
    """The first bar of history after the cut that arms a Re-entry Lock, as (its date, "buyback" or "re_entry"), or None.

    Any bar, forming included, arms by buyback when its low is at or below the cut's Buyback Price; a completed bar
    arms by re-entry when it closes above both its EMA (EMA_LENGTH bars) and the cut's Price. A bar doing both arms by
    buyback.
    """
    index = cast("pd.DatetimeIndex", history.index)
    # tz_convert(None) gives naive UTC, for bars without a timezone.
    completed = index + BAR_LENGTH[interval] <= pd.Timestamp(now).tz_convert(index.tz)
    later = pd.Series(index > pd.Timestamp(cut.date), index=index)
    # A null Buyback Price (older or plan-less locks) never arms by buyback.
    buyback = later & (history["Low"] <= (-math.inf if cut.buyback_price is None else cut.buyback_price))
    re_entry = later & completed & (history["Close"] > history[f"EMA_{EMA_LENGTH}"]) & (history["Close"] > cut.price)
    arming = cast("pd.DataFrame", history[buyback | re_entry])
    if arming.empty:
        return None
    date = cast("pd.Timestamp", arming.index[0])
    return date.isoformat(), "buyback" if buyback[date] else "re_entry"


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


def _re_entry_hit_rate(conn: sqlite3.Connection) -> dict[str, float | None]:
    """Targets / (Targets + Stops) over the current prompt version's tagged buys per arming path, all tickers and
    Intervals; each None under RE_ENTRY_MIN_SCORED."""
    buy = _RULES["buy"]
    rates = {}
    for path in ("buyback", "re_entry"):
        targets, stops = conn.execute(
            "SELECT COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
            "FROM signals WHERE prompt_version = ? AND verdict = 'buy' AND re_entry = ?",
            (buy.hit, buy.miss, PROMPT_VERSION, path),
        ).fetchone()
        rates[path] = _rate(targets, stops, RE_ENTRY_MIN_SCORED)
    return rates


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
