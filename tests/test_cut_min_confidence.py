"""Tests for the Cut Minimum Confidence: set per Interval from cut Hit Rates against each band's Cut Break-even (ADR-0011)."""

from fakes import FakeClassifier, FakeFetcher, flat_history, seed_signals

from stock_market_mcp.portfolio import PortfolioStore


def _signal(verdict="cut", confidence=0.7):
    # Flat close 100 with High 101 and Low 99: the 20-bar low (99) and high (101) are the only levels off the Price.
    return {
        "verdict": verdict,
        "confidence": confidence,
        "rationale": "test",
        "picks": {
            "buyback": {"choice": "low_20", "confidence": 0.8},
            "cut_stop": {"choice": "high_20", "confidence": 0.6},
        },
    }


def _reply(signal_maker, tmp_path, *seeds, signal=None, avg_cost=110.0) -> dict:
    """1d reply for AAPL held 50 @ avg_cost at a flat close of 100, after seeding scored cuts at a Price of 100.

    Each seed is seed_signals kwargs plus targets, stops and expired counts; seeded cuts default to a Cut Stop of
    104 and a Buyback Price of 92, a Cut Break-even of 4 / (4 + 8) = 1/3.
    """
    db = tmp_path / "smm.db"
    for seed in seeds:
        seed = {"verdict": "cut", **seed}
        outcomes = ["target"] * seed.pop("targets", 0) + ["stop"] * seed.pop("stops", 0)
        seed_signals(db, outcomes + ["expired"] * seed.pop("expired", 0), **seed)
    PortfolioStore(db).set("AAPL", 50, avg_cost)
    maker = signal_maker(FakeFetcher(history=flat_history()), FakeClassifier(signal or _signal()), db_path=db)
    return maker.make("aapl", "1d")


def test_no_band_with_30_scored_cuts_gives_0_6(signal_maker, tmp_path):
    seeds = [{"confidence": 0.9, "stops": 29, "expired": 10}, {"confidence": 0.5, "targets": 29}]
    reply = _reply(signal_maker, tmp_path, *seeds)

    assert reply["position"]["cut_min_confidence"] == 0.6


LEVEL_KEYS = {"buyback_price", "buyback_quantity", "buyback_confidence", "cut_stop", "cut_stop_confidence"}


def test_null_when_no_band_beats_its_cut_break_even(signal_maker, tmp_path):
    # Cut Stop 8 above and Buyback Price 2 below: break-even 0.8, so 20 of 30 (above a fixed 1/3 or 1/2) isn't enough.
    seeds = [{"confidence": 0.9, "targets": 20, "stops": 10, "short": (108.0, 98.0)}]
    reply = _reply(signal_maker, tmp_path, *seeds, signal=_signal("cut", 1.0))

    assert reply["position"]["cut_min_confidence"] is None
    assert reply["verdict"] == "cut"
    assert reply["position"]["cut_loss"] is False
    assert not LEVEL_KEYS & reply.keys()


def test_lowest_band_beating_its_cut_break_even_sets_it(signal_maker, tmp_path):
    seeds = [
        # 0.3 band: break-even averages 1/3 and 0.8 over its 30 scored cuts (0.567); 16 of 30 is under it.
        {"confidence": 0.3, "targets": 15},
        {"confidence": 0.3, "targets": 1, "stops": 14, "short": (108.0, 98.0)},
        {"confidence": 0.3, "expired": 30, "short": (101.0, 50.0)},  # Expired cuts don't move break-even
        {"confidence": 0.4, "targets": 9, "stops": 21},  # under 1/3
        {"confidence": 0.5, "targets": 12, "stops": 18},  # 40%: under 50% but above 1/3
        {"confidence": 0.8, "targets": 30},
    ]
    reply = _reply(signal_maker, tmp_path, *seeds)

    assert reply["position"]["cut_min_confidence"] == 0.5


def test_only_cuts_on_its_interval_count_and_winning_positions_do(signal_maker, tmp_path):
    seeds = [
        {"confidence": 0.5, "targets": 30, "interval": "1h"},
        {"confidence": 0.5, "targets": 30, "verdict": "buy"},
        {"confidence": 0.7, "targets": 20, "stops": 10, "cut_loss": 0, "ticker": "MSFT"},
    ]
    reply = _reply(signal_maker, tmp_path, *seeds)

    assert reply["position"]["cut_min_confidence"] == 0.7


def test_cut_below_it_on_a_losing_position_is_not_a_cut_loss(signal_maker, tmp_path):
    seeds = [{"confidence": 0.7, "targets": 20, "stops": 10}]
    reply = _reply(signal_maker, tmp_path, *seeds, signal=_signal("cut", 0.69))

    assert reply["verdict"] == "cut"
    assert reply["position"]["cut_loss"] is False
    assert not LEVEL_KEYS & reply.keys()


def test_cut_at_it_on_a_losing_position_is_a_cut_loss_with_levels(signal_maker, tmp_path):
    seeds = [{"confidence": 0.7, "targets": 20, "stops": 10}, {"confidence": 0.9, "targets": 0, "stops": 30}]
    reply = _reply(signal_maker, tmp_path, *seeds, signal=_signal("cut", 0.7))

    assert reply["position"]["cut_loss"] is True
    # 50 shares at 100 buy back 5000 / 99 = 50.5 shares at 99, rounded down.
    assert {key: reply[key] for key in LEVEL_KEYS} == {
        "buyback_price": 99.0,
        "buyback_quantity": 50,
        "buyback_confidence": 0.8,
        "cut_stop": 101.0,
        "cut_stop_confidence": 0.6,
    }


def test_buy_minimum_confidence_and_trade_plan_ignore_cut_outcomes(signal_maker, tmp_path):
    # The cuts give a null Cut Minimum Confidence; the buy keeps the 0.6 default and its Trade Plan.
    seeds = [{"confidence": 0.6, "stops": 30}]
    reply = _reply(signal_maker, tmp_path, *seeds, signal=_signal("buy", 0.6))

    assert reply["position"]["cut_min_confidence"] is None
    assert reply["min_confidence"] == 0.6
    assert reply["trade_plan"] is not None
