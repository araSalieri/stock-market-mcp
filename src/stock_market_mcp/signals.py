"""Signal log: one row per fresh classifier answer in the server database (ADR-0003)."""

import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

import pandas as pd

from stock_market_mcp.bars import Interval
from stock_market_mcp.prompt import PROMPT_VERSION, PreviousSignal
from stock_market_mcp.db import transaction
from stock_market_mcp.verdicts import PLAN_COLUMNS, RULES, TRACKED, default_plans

# Scored Signals of one kind (Targets plus Stops, or scored holds) needed before a Hit Rate is reported.
HIT_RATE_MIN_SCORED = 10
# Targets plus Stops a Confidence band needs before its Hit Rate can set the Minimum Confidence.
BAND_MIN_SCORED = 30
# Targets plus Stops among Cut Loss Signals needed before the Cut Loss Hit Rate is reported.
CUT_LOSS_MIN_SCORED = 30
# Minimum Confidence until a band has BAND_MIN_SCORED.
MIN_CONFIDENCE = 0.4
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

# Columns added to signals after its first release, as (column, type); missing ones are added on open.
# backfilled: 1 on a Backfilled Signal (ADR-0013), null on a live one.
_ADDED_COLUMNS = (("cut_loss", "INTEGER"), ("backfilled", "INTEGER"))


@dataclass(frozen=True)
class Measures:
    """What the Signal log has measured for one Signal: dip_min_confidence only when not held, cut_* only when held, else None."""

    hit_rate: float | None
    min_confidence: float | None
    dip_min_confidence: float | None
    cut_min_confidence: float | None
    cut_loss_hit_rate: float | None


class SignalLog:
    """Scores, records and reports what it has measured. The Trade Plan is built before the record, so a failure writes nothing."""

    def __init__(self, path: str | os.PathLike):
        self.path = path

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        """A transaction on the signals table, adding columns a file from before them lacks (null on old rows) and
        dropping the Re-entry Lock's table and column a file from before ADR-0009 has."""
        with transaction(self.path, SCHEMA) as conn:
            have = {column[1] for column in conn.execute("PRAGMA table_info(signals)")}
            for name, type_ in _ADDED_COLUMNS:
                # shortcut: two servers upgrading one file at once can fail one call on a duplicate column; retry succeeds.
                if name not in have:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {name} {type_}")
            if "re_entry" in have:
                conn.execute("ALTER TABLE signals DROP COLUMN re_entry")
            conn.execute("DROP TABLE IF EXISTS reentry_locks")
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

    def measures(self, ticker: str, interval: Interval, verdict: str, held: bool) -> Measures:
        """The Hit Rate and Minimum Confidences a get_signal reply reports for a Signal with this verdict."""
        with self._open() as conn:
            return Measures(
                hit_rate=_hit_rate(conn, ticker, interval, verdict),
                min_confidence=_min_confidence(conn, "buy", interval),
                dip_min_confidence=None if held else _min_confidence(conn, "dip", interval),
                cut_min_confidence=_min_confidence(conn, "cut", interval) if held else None,
                cut_loss_hit_rate=_cut_loss_hit_rate(conn) if held else None,
            )

    def record(
        self,
        ticker: str,
        interval: Interval,
        history: pd.DataFrame,
        price: float,
        signal: dict,
        cut_loss: bool | None,
        *,
        backfilled: bool = False,
    ) -> None:
        """Store one pending Signal made on history's last bar, live or Backfilled (ADR-0013), with the default Trade Plan
        at price and its mirror, except that its rule's picked levels (null for none) are stored over their plan columns.
        cut_loss: a cut on a losing Position at any Confidence, null when not held."""
        levels = signal["levels"]
        plan = default_plans(history, price) | {level.column: levels[level.key] for level in RULES[signal["verdict"]].levels}
        with self._open() as conn:
            conn.execute(
                "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, "
                "long_stop, long_target, short_stop, short_target, cut_loss, backfilled) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ticker,
                    interval,
                    PROMPT_VERSION,
                    signal["data_date"],
                    signal["verdict"],
                    signal["confidence"],
                    price,
                    *(plan[column] for column in PLAN_COLUMNS),
                    cut_loss,
                    backfilled or None,
                ),
            )

    def data_dates(self, ticker: str, interval: str) -> set[str]:
        """The data_date of every Signal, live or backfilled, for the ticker and Interval under the current Prompt Version."""
        with self._open() as conn:
            rows = conn.execute(
                "SELECT data_date FROM signals WHERE ticker = ? AND interval = ? AND prompt_version = ?",
                (ticker, interval, PROMPT_VERSION),
            )
            return {row[0] for row in rows}

    def score(self, ticker: str, interval: str, history: pd.DataFrame, scored_at: str) -> None:
        """Score pending Signals for the ticker and Interval against history."""
        with self._open() as conn:
            _score(conn, ticker, interval, history, scored_at)

    def track_record(self, ticker: str | None, interval: str | None) -> dict:
        """The Track Record's bands, with the server-wide Minimum Confidence and Cut Loss Hit Rate get_signal reports, and
        the Interval's Cut Minimum Confidence (None without one) and the Dip Minimum Confidence (all Intervals)."""
        with self._open() as conn:
            return {
                "bands": _bands(conn, ticker, interval),
                "min_confidence": _min_confidence(conn, "buy", interval),
                "cut_loss_hit_rate": _cut_loss_hit_rate(conn),
                "cut_min_confidence": _min_confidence(conn, "cut", interval) if interval else None,
                "dip_min_confidence": _min_confidence(conn, "dip", interval),
            }


def _score(conn: sqlite3.Connection, ticker: str, interval: str, history: pd.DataFrame, scored_at: str) -> None:
    """Give pending Signals for the ticker and Interval an Outcome from the bars after their own.

    A Signal whose bar isn't in history, or without enough later bars or a default plan, stays pending.
    """
    pending = conn.execute(
        f"SELECT id, data_date, verdict, {', '.join(PLAN_COLUMNS)} FROM signals "
        "WHERE ticker = ? AND interval = ? AND outcome IS NULL",
        (ticker, interval),
    ).fetchall()
    for id_, data_date, verdict, *prices in pending:
        bar = pd.Timestamp(data_date)
        if bar not in history.index:
            continue
        later = cast("pd.DataFrame", history[history.index > bar])
        rule = RULES[verdict]
        plan: dict[str, float | None] = dict(zip(PLAN_COLUMNS, prices, strict=True))
        outcome = rule.outcome(later, *(plan[column] for column in rule.plan))
        if outcome is not None:
            conn.execute("UPDATE signals SET outcome = ?, scored_at = ? WHERE id = ?", (outcome, scored_at, id_))


def _hit_rate(conn: sqlite3.Connection, ticker: str, interval: str, verdict: str) -> float | None:
    """Hit Rate of the current prompt version's Signals with this verdict; None under HIT_RATE_MIN_SCORED."""
    rule = RULES[verdict]
    right, wrong = conn.execute(
        "SELECT COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
        "FROM signals WHERE ticker = ? AND interval = ? AND prompt_version = ? AND verdict = ?",
        (rule.hit, rule.miss, ticker, interval, PROMPT_VERSION, verdict),
    ).fetchone()
    return _rate(right, wrong)


def _min_confidence(conn: sqlite3.Connection, verdict: str, interval: str | None) -> float | None:
    """Lowest Confidence band of the current prompt version's Signals with this verdict whose Hit Rate is above its
    rule's break-even (see _lowest_band, _break_evens), on interval when the rule's scope is per Interval, else
    server-wide: the Minimum Confidence, Dip Minimum Confidence or Cut Minimum Confidence. None for a rule with no Minimum Confidence."""
    rule = RULES[verdict]
    if rule.scope is None or rule.break_even is None:
        return None
    if rule.scope == "server":
        interval = None
    fixed = rule.break_even
    if fixed == "picked":
        break_evens = _break_evens(conn, verdict, None, interval)
        # A band with scored Signals always has a break-even, and only those reach break_even.
        return _lowest_band(_band_counts(conn, verdict, interval), lambda band: cast("float", break_evens[band]))
    return _lowest_band(_band_counts(conn, verdict, interval), lambda _: fixed)


def _band_counts(conn: sqlite3.Connection, verdict: str, interval: str | None) -> list[tuple[int, int, int]]:
    """(band, hits, misses) per Confidence band of the current prompt version's Signals with this verdict, optionally on
    one Interval, ascending."""
    rule = RULES[verdict]
    return conn.execute(
        f"SELECT {BAND} AS band, COUNT(*) FILTER (WHERE outcome = ?), COUNT(*) FILTER (WHERE outcome = ?) "
        "FROM signals WHERE prompt_version = ? AND verdict = ? AND (? IS NULL OR interval = ?) GROUP BY band ORDER BY band",
        (rule.hit, rule.miss, PROMPT_VERSION, verdict, interval, interval),
    ).fetchall()


def _lowest_band(bands: list[tuple[int, int, int]], break_even: Callable[[int], float]) -> float | None:
    """Lowest band (1 / BANDS_PER_CONFIDENCE wide) whose Hit Rate is above its break_even.

    Only bands with BAND_MIN_SCORED hits plus misses count: MIN_CONFIDENCE while none has that many,
    None when some do but none is above its break_even.
    """
    scored = [(band, rate) for band, hits, misses in bands if (rate := _rate(hits, misses, BAND_MIN_SCORED)) is not None]
    for band, rate in scored:
        if rate > break_even(band):
            return band / BANDS_PER_CONFIDENCE
    return None if scored else MIN_CONFIDENCE


def _break_evens(
    conn: sqlite3.Connection, verdict: str, ticker: str | None, interval: str | None
) -> dict[int, float | None]:
    """Break-even per Confidence band of the current prompt version's Signals with this verdict (one whose rule's plan
    is its picked Stop and Target: cut's Cut Break-even, dip's Dip Break-even), optionally for one ticker and/or Interval, over their
    hits and misses; None in a band without any."""
    rule = RULES[verdict]
    stop, target = rule.plan
    return dict(
        conn.execute(
            f"SELECT {BAND} AS band, AVG(({stop} - price) / ({stop} - {target})) "
            "FILTER (WHERE outcome IN (?, ?)) FROM signals WHERE prompt_version = ? AND verdict = ? "
            "AND (? IS NULL OR ticker = ?) AND (? IS NULL OR interval = ?) GROUP BY band",
            (rule.hit, rule.miss, PROMPT_VERSION, verdict, ticker, ticker, interval, interval),
        ).fetchall()
    )


def _bands(conn: sqlite3.Connection, ticker: str | None, interval: str | None) -> list[dict]:
    """Outcome counts and Hit Rates of the current prompt version's Signals per Confidence band (1 / BANDS_PER_CONFIDENCE wide) and Verdict,
    optionally for one ticker and/or Interval, bands with Signals only, ascending, each with only the Verdicts it has;
    a band's cut and dip also have their Cut or Dip Break-even over the same filter, None without scored ones.

    Leaves out Signals that can never be scored: those missing a price their Outcome rule is judged by.
    """
    scorable = " OR ".join(
        f"(verdict = '{verdict}' AND {' AND '.join(f'{column} IS NOT NULL' for column in RULES[verdict].plan)})"
        for verdict in TRACKED
    )
    rows = conn.execute(
        f"SELECT {BAND} AS band, verdict, COALESCE(outcome, 'pending'), COUNT(*) FROM signals "
        "WHERE prompt_version = ? AND (? IS NULL OR ticker = ?) AND (? IS NULL OR interval = ?) "
        f"AND ({scorable}) GROUP BY band, verdict, outcome ORDER BY band",
        (PROMPT_VERSION, ticker, ticker, interval, interval),
    ).fetchall()
    tallies: dict[int, dict[str, dict[str, int]]] = {}
    for band, verdict, outcome, count in rows:
        tallies.setdefault(band, {v: {} for v in TRACKED})[verdict][outcome] = count
    buy = RULES["buy"]
    break_evens = {
        verdict: _break_evens(conn, verdict, ticker, interval) for verdict in TRACKED if RULES[verdict].break_even == "picked"
    }
    return [
        {
            "band": band / BANDS_PER_CONFIDENCE,
            **{
                verdict: _counts(verdict, tally[verdict])
                | ({"break_even": break_evens[verdict][band]} if verdict in break_evens else {})
                for verdict in TRACKED
                if tally[verdict]
            },
            "buy_hit_rate": _rate(tally["buy"].get(buy.hit, 0), tally["buy"].get(buy.miss, 0), BAND_MIN_SCORED),
        }
        for band, tally in tallies.items()
    ]


def _cut_loss_hit_rate(conn: sqlite3.Connection) -> float | None:
    """Targets / (Targets + Stops) over the current prompt version's Signals stored with cut_loss = 1, all tickers and
    Intervals; None under CUT_LOSS_MIN_SCORED."""
    cut = RULES["cut"]
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
    rule = RULES[verdict]
    counts = {outcome: tally.get(outcome, 0) for outcome in (rule.hit, rule.miss, *rule.other, "pending")}
    return {**counts, "hit_rate": _rate(counts[rule.hit], counts[rule.miss])}
