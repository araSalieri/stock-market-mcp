"""Tests for the Re-entry Lock (#93): SignalMaker locks a ticker on an Interval after a confident cut, until a later
bar closes above both the 10-bar EMA and the cut's Price.

Drives SignalMaker with fakes, not the MCP surface: spec #89's Testing Decisions ask for it, to step history bar by bar."""

import pandas as pd
import pytest
from fakes import FakeClassifier, FakeFetcher

from stock_market_mcp.portfolio import PortfolioStore


def _signal(verdict, confidence=0.7):
    return {"verdict": verdict, "confidence": confidence, "rationale": "test", "data_date": "2024-02-29T00:00:00"}


def bars(closes: list[float], window: int = 60) -> pd.DataFrame:
    """Daily bars from 2024-01-01 with High/Low one unit from Close, cut to the last `window` like a history fetch."""
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="D", name="Date")
    close = pd.Series(closes, index=idx, dtype=float)
    df = pd.DataFrame({"Open": close, "High": close + 1.0, "Low": close - 1.0, "Close": close, "Volume": 1000})
    return df.iloc[-window:]


FLAT = [100.0] * 60  # EMA 100; a cut here is at Price 100 on 2024-02-29


def _calls(signal_maker, tmp_path, steps):
    """One make() per (history, signal[, interval]) step on a held AAPL, 1d by default; ttl 0, so every call is fresh."""
    PortfolioStore(tmp_path / "smm.db").set("AAPL", 10, 110.0)
    fetcher = FakeFetcher(history=[step[0] for step in steps])
    classifier = FakeClassifier(replies=[step[1] for step in steps])
    maker = signal_maker(fetcher, classifier, db_path=tmp_path / "smm.db", ttl=0)
    return [maker.make("aapl", *step[2:]) for step in steps]


def test_confident_cut_locks_out_the_next_buy(signal_maker, tmp_path):
    cut, buy = _calls(signal_maker, tmp_path, [(bars(FLAT), _signal("cut")), (bars(FLAT), _signal("buy"))])

    assert buy["trade_plan"] is None
    assert buy["re_entry_lock"] == {"cut_price": 100.0, "cut_date": "2024-02-29T00:00:00", "ema": 100.0}
    assert cut["re_entry_lock"] == buy["re_entry_lock"]


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
        (FLAT, [90.0] * 20 + [95.0], False),  # above the EMA only
        ([120.0] * 59 + [100.0], [105.0], False),  # above the cut Price only
    ],
)
def test_close_above_both_the_ema_and_the_cut_price_lifts_the_lock(signal_maker, tmp_path, before, after, lifted):
    _, buy = _calls(signal_maker, tmp_path, [(bars(before), _signal("cut")), (bars(before + after), _signal("buy"))])

    assert (buy["trade_plan"] is not None) is lifted
    assert ("re_entry_lock" not in buy) is lifted


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
    assert "re_entry_lock" not in buy


def test_next_confident_cut_locks_again_at_its_price(signal_maker, tmp_path):
    lifted = FLAT + [101.0, 110.0]
    *_, buy = _calls(
        signal_maker,
        tmp_path,
        [(bars(FLAT), _signal("cut")), (bars(lifted), _signal("buy")), (bars(lifted), _signal("cut"))]
        + [(bars(lifted), _signal("buy"))],
    )

    assert buy["trade_plan"] is None
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
    assert ("re_entry_lock" not in cached) is lifted
