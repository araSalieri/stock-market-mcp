"""Tests for the Re-entry Lock: SignalMaker locks a ticker on an Interval after a confident cut, until a later bar's low
touches the Buyback Price or a later completed bar closes above the Re-entry Price (the higher of the 10-bar EMA and the
cut's Price), whichever comes first.

Drives SignalMaker with fakes, not the MCP surface, to step history bar by bar."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pandas as pd
import pytest
from fakes import FakeClassifier, FakeFetcher

from stock_market_mcp.portfolio import PortfolioStore


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test"}


def bars(closes: list[float], window: int = 60) -> pd.DataFrame:
    """Daily bars from 2024-01-01 with High/Low one unit from Close, cut to the last `window` like a history fetch."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", name="Date")
    close = pd.Series(closes, index=idx, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000})
    return df.iloc[-window:]


FLAT = [100.0] * 60  # EMA 100 and ATR 2; a cut here is at Price 100 on 2024-02-29, Buyback Price 100 - 2 * 2 * 2 = 92


def _calls(signal_maker, tmp_path, steps, clocks=None, clear_at=None):
    """One make() per (history, signal[, interval]) step on a held AAPL, 5m by default; ttl 0, so every call is fresh.

    clocks (optional) is the server's clock at each step; without it, the real clock, so every bar is completed.
    clear_at (optional) is the step from which AAPL is no longer held."""
    PortfolioStore(tmp_path / "smm.db").set("AAPL", 10, 110.0)
    fetcher = FakeFetcher(history=[step[0] for step in steps])
    classifier = FakeClassifier(replies=[step[1] for step in steps])
    clock = [datetime.now(UTC)]
    maker = signal_maker(fetcher, classifier, db_path=tmp_path / "smm.db", ttl=0, now=lambda: clock[0])
    replies = []
    for i, step in enumerate(steps):
        if clocks is not None:
            clock[0] = clocks[i]
        if i == clear_at:
            PortfolioStore(tmp_path / "smm.db").delete("AAPL")
        replies.append(maker.make("aapl", *step[2:]))
    return replies


def _lifted(reply) -> bool:
    return reply["re_entry_lock"]["state"] == "lifted"


def test_confident_cut_locks_out_the_next_buy(signal_maker, tmp_path):
    cut, buy = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut")), (bars(FLAT), _signal("buy"))])

    assert buy["trade_plan"] is None
    assert buy["re_entry_lock"] == {
        "state": "locked",
        "cut_price": 100.0,
        "cut_date": "2024-02-29T00:00:00",
        "buyback_price": 92.0,
        "ema": 100.0,
        "re_entry_price": 100.0,
    }
    assert cut["re_entry_lock"] == buy["re_entry_lock"]


def test_re_entry_price_is_the_higher_of_the_cut_price_and_the_current_ema(signal_maker, tmp_path):
    # Cut at 100 after a fall from 120, so the EMA is still above the cut Price; then it falls below it.
    falling = [120.0] * 59 + [100.0]
    cut, buy = _calls(
        signal_maker, tmp_path, [(bars(falling), _signal("cut")), (bars(falling + [90.0] * 30), _signal("buy"))]
    )

    assert cut["re_entry_lock"]["re_entry_price"] == cut["re_entry_lock"]["ema"] > 100.0
    assert buy["re_entry_lock"]["ema"] < 100.0
    assert buy["re_entry_lock"]["re_entry_price"] == 100.0


def test_buy_without_a_lock_has_no_lock_key(signal_maker, tmp_path):
    [buy] = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("buy"))])

    assert buy["trade_plan"] is not None
    assert "re_entry_lock" not in buy


def test_weak_cut_does_not_lock(signal_maker, tmp_path):
    # No scored Signals, so the Minimum Confidence is the 0.6 default.
    _, buy = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut", 0.59)), (bars(FLAT), _signal("buy"))])

    assert buy["trade_plan"] is not None
    assert "re_entry_lock" not in buy


def test_lock_is_per_interval(signal_maker, tmp_path):
    _, other, same = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut"), "5m"), (bars(FLAT), _signal("buy"), "1d"), (bars(FLAT), _signal("buy"), "5m")],
    )

    assert other["trade_plan"] is not None
    assert "re_entry_lock" not in other
    assert same["trade_plan"] is None


@pytest.mark.parametrize(
    ("before", "after", "lifted"),
    [
        (FLAT, [101.0], True),  # above the EMA and the cut Price
        (FLAT, [94.0] * 20 + [97.0], False),  # above the EMA only, lows above the Buyback Price
        ([120.0] * 59 + [100.0], [105.0], False),  # above the cut Price only
    ],
)
def test_close_above_both_the_ema_and_the_cut_price_lifts_the_lock(signal_maker, tmp_path, before, after, lifted):
    _, buy = _calls(signal_maker, tmp_path, [(bars(before), _signal("cut")), (bars(before + after), _signal("buy"))])

    assert (buy["trade_plan"] is not None) is lifted
    assert _lifted(buy) is lifted


def test_forming_bar_closing_above_the_re_entry_price_does_not_lift_until_completed(signal_maker, tmp_path):
    # The 101 bar starts 2024-03-01 and completes a day later.
    lifting = bars(FLAT + [101.0])
    _, forming, completed = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut"), "1d"), (lifting, _signal("buy"), "1d"), (lifting, _signal("buy"), "1d")],
        clocks=[datetime(2024, 3, 1, 12, tzinfo=UTC), datetime(2024, 3, 1, 12, tzinfo=UTC), datetime(2024, 3, 2, tzinfo=UTC)],
    )

    assert forming["trade_plan"] is None
    assert forming["re_entry_lock"]["state"] == "locked"
    assert completed["trade_plan"] is not None
    assert completed["re_entry_lock"] == {
        "state": "lifted",
        "cut_price": 100.0,
        "cut_date": "2024-02-29T00:00:00",
        "buyback_price": 92.0,
        "lifted_date": "2024-03-01T00:00:00",
        "lifted_by": "re_entry",
    }


def test_buy_after_a_lift_still_needs_the_minimum_confidence(signal_maker, tmp_path):
    lifted = FLAT + [101.0]
    _, weak = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut")), (bars(lifted), _signal("buy", 0.59))])

    assert _lifted(weak)
    assert weak["trade_plan"] is None


def test_lifted_lock_stays_lifted_after_a_dip_and_after_the_lifting_bar_leaves_the_window(signal_maker, tmp_path):
    lifted = FLAT + [101.0]
    *_, buy = _calls(
        signal_maker,
        tmp_path,
        [
            (bars(FLAT), _signal("cut")),
            (bars(lifted), _signal("buy")),
            (bars(lifted + [80.0] * 70), _signal("buy")),  # 70 dip bars push the 101 bar out of the 60-bar window
        ],
    )

    assert buy["trade_plan"] is not None
    assert buy["re_entry_lock"]["state"] == "lifted"
    assert buy["re_entry_lock"]["lifted_date"] == "2024-03-01T00:00:00"


def test_next_confident_cut_locks_again_at_its_price(signal_maker, tmp_path):
    lifted = FLAT + [101.0, 110.0]
    *_, buy = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(lifted), _signal("buy")), (bars(lifted), _signal("cut"))]
        + [(bars(lifted), _signal("buy"))],
    )

    assert buy["trade_plan"] is None
    assert buy["re_entry_lock"]["state"] == "locked"
    assert buy["re_entry_lock"]["cut_price"] == 110.0
    assert buy["re_entry_lock"]["cut_date"] == "2024-03-02T00:00:00"


@pytest.mark.parametrize(("buy_bars", "lifted"), [(FLAT, False), (FLAT + [101.0], True)])
def test_cached_buy_respects_the_lock_and_the_lift(signal_maker, tmp_path, buy_bars, lifted):
    # Unheld buy, held cut on the flat bars, unheld buy again: the third call is the cached first Signal.
    db = tmp_path / "smm.db"
    fetcher = FakeFetcher(history=[bars(buy_bars), bars(FLAT), bars(buy_bars)])
    classifier = FakeClassifier(replies=[_signal("buy"), _signal("cut")])
    maker = signal_maker(fetcher, classifier, db_path=db)
    first = maker.make("aapl", "1d")
    PortfolioStore(db).set("AAPL", 10, 110.0)
    maker.make("aapl", "1d")
    PortfolioStore(db).delete("AAPL")
    cached = maker.make("aapl", "1d")

    assert len(classifier.prompts) == 2  # the lock changes neither the prompt nor the Signal cache key
    assert first["trade_plan"] is not None
    assert (cached["trade_plan"] is not None) is lifted
    assert _lifted(cached) is lifted


def test_lock_rows_from_an_older_database_keep_working(signal_maker, tmp_path):
    # An older file can only be built by hand: no tool writes the old lock table.
    db = tmp_path / "smm.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "CREATE TABLE reentry_locks (ticker TEXT NOT NULL, interval TEXT NOT NULL, cut_price REAL NOT NULL, "
            "cut_date TEXT NOT NULL, lifted INTEGER NOT NULL, PRIMARY KEY (ticker, interval))"
        )
        conn.execute("INSERT INTO reentry_locks VALUES ('AAPL', '1d', 100.0, '2024-02-29T00:00:00', 0)")
        conn.execute("INSERT INTO reentry_locks VALUES ('AAPL', '5m', 100.0, '2024-02-29T00:00:00', 1)")
        conn.execute(
            "CREATE TABLE signals (id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, interval TEXT NOT NULL, "
            "prompt_version INTEGER NOT NULL, data_date TEXT NOT NULL, verdict TEXT NOT NULL, confidence REAL NOT NULL, "
            "price REAL NOT NULL, long_stop REAL, long_target REAL, short_stop REAL, short_target REAL, outcome TEXT, "
            "scored_at TEXT)"
        )
        # The 1d lock's cut Signal, with its mirrored default Target as the short Target.
        conn.execute(
            "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, "
            "long_stop, long_target, short_stop, short_target) "
            "VALUES ('AAPL', '1d', 1, '2024-02-29T00:00:00', 'cut', 0.7, 100.0, 96.0, 108.0, 104.0, 92.0)"
        )
    # Not held, so only the missing cut quantity keeps buyback_quantity out.
    locked, old_lifted, lifting = _calls(
        signal_maker,
        tmp_path,
        [
            (bars(FLAT), _signal("buy"), "1d"),
            (bars(FLAT), _signal("buy"), "5m"),
            (bars(FLAT + [101.0]), _signal("buy"), "1d"),
        ],
        clear_at=0,
    )

    assert locked["trade_plan"] is None
    assert locked["re_entry_lock"]["state"] == "locked"
    assert locked["re_entry_lock"]["buyback_price"] == 92.0
    assert "buyback_quantity" not in locked["re_entry_lock"]
    assert old_lifted["trade_plan"] is not None
    assert old_lifted["re_entry_lock"] == {
        "state": "lifted",
        "cut_price": 100.0,
        "cut_date": "2024-02-29T00:00:00",
        "buyback_price": None,
        "lifted_date": None,
        "lifted_by": None,
    }
    assert lifting["re_entry_lock"]["lifted_by"] == "re_entry"


@pytest.mark.parametrize(("close", "lifted"), [(93.0, True), (93.5, False)])  # lows 92 and 92.5
def test_low_at_or_below_the_buyback_price_lifts_the_lock_on_a_forming_bar(signal_maker, tmp_path, close, lifted):
    # The bar starts 2024-03-01 and is still forming at noon.
    noon = datetime(2024, 3, 1, 12, tzinfo=UTC)
    _, buy = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut"), "1d"), (bars(FLAT + [close]), _signal("buy"), "1d")],
        clocks=[noon, noon],
    )

    assert (buy["trade_plan"] is not None) is lifted
    if lifted:
        assert buy["re_entry_lock"] == {
            "state": "lifted",
            "cut_price": 100.0,
            "cut_date": "2024-02-29T00:00:00",
            "buyback_price": 92.0,
            "lifted_date": "2024-03-01T00:00:00",
            "lifted_by": "buyback",
        }
    else:
        assert buy["re_entry_lock"]["state"] == "locked"


def test_bar_touching_the_buyback_price_and_closing_above_the_re_entry_price_lifts_by_buyback(signal_maker, tmp_path):
    both = bars(FLAT + [101.0])
    both.loc[both.index[-1], "Low"] = 92.0
    _, buy = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut")), (both, _signal("buy"))])

    assert buy["re_entry_lock"]["lifted_by"] == "buyback"


def test_buy_below_the_minimum_confidence_after_a_buyback_lift_gets_no_trade_plan(signal_maker, tmp_path):
    _, weak = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut")), (bars(FLAT + [93.0]), _signal("buy", 0.59))])

    assert weak["re_entry_lock"]["lifted_by"] == "buyback"
    assert weak["trade_plan"] is None


def test_next_confident_cut_after_a_buyback_lift_gives_a_fresh_buyback_price(signal_maker, tmp_path):
    # 60 bars at 80 fill the window: ATR 2, so a cut at 80 has Buyback Price 72.
    lower = FLAT + [93.0] + [80.0] * 60
    _, lifted, cut = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(FLAT + [93.0]), _signal("buy")), (bars(lower), _signal("cut"))],
    )

    assert lifted["re_entry_lock"]["lifted_by"] == "buyback"
    assert cut["re_entry_lock"]["state"] == "locked"
    assert cut["re_entry_lock"]["buyback_price"] == 72.0


def test_cleared_position_gets_the_buyback_quantity_rounded_down_locked_or_lifted(signal_maker, tmp_path):
    # 10 shares cut at 100 free 1000, which buys 10.87 shares at the Buyback Price 92.
    _, locked, lifted = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(FLAT), _signal("buy")), (bars(FLAT + [93.0]), _signal("buy"))],
        clear_at=1,
    )

    assert locked["re_entry_lock"]["state"] == "locked"
    assert locked["re_entry_lock"]["buyback_quantity"] == 10
    assert lifted["re_entry_lock"]["state"] == "lifted"
    assert lifted["re_entry_lock"]["buyback_quantity"] == 10


def test_no_buyback_quantity_while_the_ticker_is_held(signal_maker, tmp_path):
    cut, locked, lifted = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(FLAT), _signal("buy")), (bars(FLAT + [93.0]), _signal("buy"))],
    )

    assert all("buyback_quantity" not in reply["re_entry_lock"] for reply in (cut, locked, lifted))
