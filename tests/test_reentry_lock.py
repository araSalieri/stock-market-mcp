"""Tests for the Re-entry Lock: SignalMaker locks a ticker on an Interval after a confident cut. A later bar's low
touching the Buyback Price or a later completed bar closing above the Re-entry Price (the higher of the 10-bar EMA and
the cut's Price), whichever comes first, arms it; the first fresh buy at or above the Minimum Confidence then lifts it.

Drives SignalMaker with fakes, not the MCP surface, to step history bar by bar."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pandas as pd
import pytest
from fakes import FakeClassifier, FakeFetcher, seed_signals

from stock_market_mcp.portfolio import PortfolioStore
from stock_market_mcp.prompt import PROMPT_VERSION


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
    assert "re_entry_hit_rate" not in buy


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
    # A cached buy is never recorded, so it can't lift: a confident one on an armed lock gets a Trade Plan only.
    assert (cached["trade_plan"] is not None) is lifted
    assert cached["re_entry_lock"]["state"] == ("armed" if lifted else "locked")


def test_lock_rows_from_an_older_database_keep_working(signal_maker, tmp_path):
    # An older file can only be built by hand: no tool writes the old lock table. This one is from before the Buyback
    # Price and the armed state, with first-buy tags from the old price-only lift.
    db = tmp_path / "smm.db"
    with closing(sqlite3.connect(db)) as conn, conn:
        conn.execute(
            "CREATE TABLE reentry_locks (ticker TEXT NOT NULL, interval TEXT NOT NULL, cut_price REAL NOT NULL, "
            "cut_date TEXT NOT NULL, lifted INTEGER NOT NULL, lifted_date TEXT, lifted_by TEXT, "
            "PRIMARY KEY (ticker, interval))"
        )
        conn.execute("INSERT INTO reentry_locks VALUES ('AAPL', '1d', 100.0, '2024-02-29T00:00:00', 0, NULL, NULL)")
        conn.execute(
            "INSERT INTO reentry_locks VALUES ('AAPL', '5m', 100.0, '2024-02-29T00:00:00', 1, '2024-03-01T00:00:00', "
            "'buyback')"
        )
        conn.execute(
            "CREATE TABLE signals (id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, interval TEXT NOT NULL, "
            "prompt_version INTEGER NOT NULL, data_date TEXT NOT NULL, verdict TEXT NOT NULL, confidence REAL NOT NULL, "
            "price REAL NOT NULL, long_stop REAL, long_target REAL, short_stop REAL, short_target REAL, outcome TEXT, "
            "scored_at TEXT, cut_loss INTEGER, re_entry TEXT)"
        )
        # The 1d lock's cut Signal, with its mirrored default Target as the short Target.
        conn.execute(
            "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, "
            "long_stop, long_target, short_stop, short_target) "
            "VALUES ('AAPL', '1d', ?, '2024-02-29T00:00:00', 'cut', 0.7, 100.0, 96.0, 108.0, 104.0, 92.0)",
            (PROMPT_VERSION,),
        )
        # 30 scored first-buy tags under the old rule, at a band that keeps the Minimum Confidence at 0.1.
        conn.executemany(
            "INSERT INTO signals (ticker, interval, prompt_version, data_date, verdict, confidence, price, outcome, "
            "re_entry) VALUES ('MSFT', '1d', ?, '2023-01-01T00:00:00', 'buy', 0.1, 100.0, 'target', 're_entry')",
            [(PROMPT_VERSION,)] * 30,
        )
    # Not held, so only the missing cut quantity keeps buyback_quantity out.
    locked, old_lifted, armed, lifting = _calls(
        signal_maker,
        tmp_path,
        [
            (bars(FLAT), _signal("buy"), "1d"),
            (bars(FLAT), _signal("buy"), "5m"),
            (bars(FLAT + [101.0]), _signal("hold"), "1d"),  # arms by re-entry
            (bars(FLAT + [101.0]), _signal("buy"), "1d"),  # lifts: tagged
        ],
        clear_at=0,
    )
    # 29 more tags, so scoring the lifting buy as a Target completes 30 only if a later open keeps its tag.
    seed_signals(db, ["target"] * 29, confidence=0.1, ticker="MSFT", re_entry="re_entry")
    (later,) = _calls(signal_maker, tmp_path, [(bars(FLAT + [101.0, 300.0]), _signal("hold"), "1d")])

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
        "lifted_date": "2024-03-01T00:00:00",
        "lifted_by": "buyback",
    }
    assert armed["re_entry_lock"] == {
        "state": "armed",
        "cut_price": 100.0,
        "cut_date": "2024-02-29T00:00:00",
        "buyback_price": 92.0,
        "armed_by": "re_entry",
        "armed_date": "2024-03-01T00:00:00",
    }
    assert lifting["re_entry_lock"]["lifted_by"] == "re_entry"
    assert lifting["re_entry_hit_rate"] == {"buyback": None, "re_entry": None}
    assert later["re_entry_hit_rate"] == {"buyback": None, "re_entry": 1.0}


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


def test_buy_below_the_minimum_confidence_after_a_buyback_touch_leaves_the_lock_armed(signal_maker, tmp_path):
    _, weak = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut")), (bars(FLAT + [93.0]), _signal("buy", 0.59))])

    assert weak["re_entry_lock"]["state"] == "armed"
    assert weak["re_entry_lock"]["armed_by"] == "buyback"
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


def test_cleared_position_gets_the_buyback_quantity_rounded_down_locked_armed_or_lifted(signal_maker, tmp_path):
    # 10 shares cut at 100 free 1000, which buys 10.87 shares at the Buyback Price 92.
    dip = bars(FLAT + [93.0])
    _, locked, armed, lifted = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(FLAT), _signal("buy")), (dip, _signal("hold")), (dip, _signal("buy"))],
        clear_at=1,
    )

    assert [reply["re_entry_lock"]["state"] for reply in (locked, armed, lifted)] == ["locked", "armed", "lifted"]
    assert all(reply["re_entry_lock"]["buyback_quantity"] == 10 for reply in (locked, armed, lifted))


def test_no_buyback_quantity_while_the_ticker_is_held(signal_maker, tmp_path):
    dip = bars(FLAT + [93.0])
    replies = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(FLAT), _signal("buy")), (dip, _signal("hold")), (dip, _signal("buy"))],
    )

    assert [reply["re_entry_lock"]["state"] for reply in replies] == ["locked", "locked", "armed", "lifted"]
    assert all("buyback_quantity" not in reply["re_entry_lock"] for reply in replies)


def test_only_the_buy_lifting_each_lock_is_tagged_with_its_path_for_the_re_entry_hit_rate(signal_maker, tmp_path):
    # 29 scored buys per path elsewhere, at a band that keeps the Minimum Confidence at 0.1, so each tagged buy here
    # completes 30 when scored.
    for path in ("buyback", "re_entry"):
        seed_signals(tmp_path / "smm.db", ["target"] * 15 + ["stop"] * 14, confidence=0.1, ticker="MSFT", re_entry=path)
    dip = FLAT + [93.0]  # low 92 arms the cut at 100 by buyback
    jump = dip + [200.0]  # closes above the EMA and the 93 cut: arms by re-entry, and scores the buys at 93 as Targets
    *_, last = _calls(
        signal_maker,
        tmp_path,
        [
            (bars(FLAT), _signal("cut")),
            (bars(dip), _signal("buy", 0.05)),  # armed, below the Minimum Confidence: not tagged
            (bars(dip), _signal("buy")),  # confident, lifts by buyback: tagged
            (bars(dip), _signal("buy")),  # after the lift: not tagged
            (bars(dip), _signal("cut")),  # the next confident cut starts over
            (bars(jump), _signal("buy")),  # lifts by re-entry: tagged
            (bars(jump), _signal("buy")),
            (bars(jump + [300.0]), _signal("buy")),  # scores the buys at 200 as Targets
        ],
    )

    assert last["re_entry_hit_rate"] == {"buyback": pytest.approx(16 / 30), "re_entry": pytest.approx(16 / 30)}


ARMED = {
    "state": "armed",
    "cut_price": 100.0,
    "cut_date": "2024-02-29T00:00:00",
    "buyback_price": 92.0,
    "armed_by": "re_entry",
    "armed_date": "2024-03-01T00:00:00",
}


@pytest.mark.parametrize("signal", [_signal("hold"), _signal("buy", 0.59)], ids=["hold", "weak buy"])
def test_price_arms_the_lock_and_only_a_confident_buy_lifts_it(signal_maker, tmp_path, signal):
    _, waiting = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut")), (bars(FLAT + [101.0]), signal)])

    assert waiting["trade_plan"] is None
    assert waiting["re_entry_lock"] == ARMED


def test_first_arming_stays_after_later_armings_and_the_price_moving_back(signal_maker, tmp_path):
    armed = FLAT + [101.0]
    *_, later, back = _calls(
        signal_maker,
        tmp_path,
        [
            (bars(FLAT), _signal("cut")),
            (bars(armed), _signal("hold")),  # arms by re-entry on 2024-03-01
            (bars(armed + [93.0]), _signal("hold")),  # low 92 touches the Buyback Price later
            (bars(FLAT), _signal("buy", 0.59)),  # bars that arm by neither path
        ],
    )

    assert later["re_entry_lock"] == back["re_entry_lock"] == ARMED


def test_confident_cut_on_an_armed_lock_locks_again_at_its_price(signal_maker, tmp_path):
    armed = FLAT + [101.0, 110.0]
    *_, buy = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(armed), _signal("hold")), (bars(armed), _signal("cut"))]
        + [(bars(armed), _signal("buy"))],
    )

    assert buy["trade_plan"] is None
    assert buy["re_entry_lock"]["state"] == "locked"
    assert buy["re_entry_lock"]["cut_price"] == 110.0


def test_no_lock_lifts_while_the_minimum_confidence_is_null(signal_maker, tmp_path):
    _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut"))])
    # 30 scored buys elsewhere, all Stops, so no band beats break-even.
    seed_signals(tmp_path / "smm.db", ["stop"] * 30, confidence=0.95, ticker="MSFT")
    [buy] = _calls(signal_maker, tmp_path, [(bars(FLAT + [101.0]), _signal("buy", 0.99))])

    assert buy["min_confidence"] is None
    assert buy["trade_plan"] is None
    assert buy["re_entry_lock"] == ARMED


def test_confident_buy_on_an_armed_lock_lifts_it_on_the_buys_data_date(signal_maker, tmp_path):
    armed = FLAT + [101.0]
    *_, buy, after = _calls(
        signal_maker,
        tmp_path,
        [
            (bars(FLAT), _signal("cut")),
            (bars(armed), _signal("hold")),  # arms by re-entry on 2024-03-01
            (bars(armed + [102.0]), _signal("buy")),
            (bars(armed + [102.0]), _signal("hold")),
        ],
    )

    lifted = {
        "state": "lifted",
        "cut_price": 100.0,
        "cut_date": "2024-02-29T00:00:00",
        "buyback_price": 92.0,
        "lifted_date": "2024-03-02T00:00:00",
        "lifted_by": "re_entry",
    }
    assert buy["trade_plan"] is not None
    assert buy["re_entry_lock"] == after["re_entry_lock"] == lifted
