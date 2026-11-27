"""Verdict rules: one rule per Verdict, saying how its levels are read, stored, scored and given a Minimum Confidence, and what a confident one adds to the get_signal reply, plus the Trade Plan maths."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal, NamedTuple

import pandas as pd

from stock_market_mcp.bars import last_value
from stock_market_mcp.errors import InputError
from stock_market_mcp.portfolio import PricedPosition

PLAN_ATR_MULTIPLE = 2.0
PLAN_REWARD_RATIO = 2.0
# Hit Rate a default Trade Plan must beat to break even: a Target gains PLAN_REWARD_RATIO times what a Stop loses.
BREAK_EVEN_HIT_RATE = 1 / (1 + PLAN_REWARD_RATIO)

# Bars after a Signal's own bar within which it gets an Outcome.
OUTCOME_BARS = 20

# The stored plan columns: long_* hold the default Trade Plan, short_* the same distances mirrored, unless a rule's
# levels are stored in them instead (ADR-0014).
PLAN_COLUMNS = ("long_stop", "long_target", "short_stop", "short_target")


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


def default_plans(history: pd.DataFrame, entry: float) -> dict[str, float | None]:
    """The default Trade Plan and its mirror at entry by plan column; nulls for one that can't be made."""
    plans = {}
    for long in (True, False):
        try:
            plan = _trade_plan(history, entry, PLAN_ATR_MULTIPLE, PLAN_REWARD_RATIO, long=long)
            prices = plan["stop"], plan["target"]
        except InputError:
            prices = None, None
        side = "long" if long else "short"
        plans |= {f"{side}_stop": prices[0], f"{side}_target": prices[1]}
    return plans


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


def _buy_additions(
    history: pd.DataFrame, price: float, levels: dict, held: PricedPosition | None, atr_multiple: float, reward_ratio: float
) -> dict:
    """A confident buy's Trade Plan from ATR."""
    return {"trade_plan": _trade_plan(history, price, atr_multiple, reward_ratio)}


def _dip_additions(
    history: pd.DataFrame, price: float, levels: dict, held: PricedPosition | None, atr_multiple: float, reward_ratio: float
) -> dict:
    """A confident dip's Trade Plan at the Price from its picked Stop and Target, with their Reward Ratio; none without both."""
    stop, target = levels["stop"], levels["target"]
    if stop is None or target is None:
        return {"trade_plan": None}
    return {"trade_plan": {"entry": price, "stop": stop, "target": target, "reward_ratio": (target - price) / (price - stop)}}


def _cut_additions(
    history: pd.DataFrame, price: float, levels: dict, held: PricedPosition, atr_multiple: float, reward_ratio: float
) -> dict:
    """A confident cut's picked levels with the Buyback Quantity the Position's proceeds buy at the Buyback Price."""
    buyback = levels["buyback_price"]
    quantity = None if buyback is None else math.floor(held.quantity * price / buyback)
    return {**levels, "buyback_quantity": quantity}


def _no_additions(
    history: pd.DataFrame, price: float, levels: dict, held: PricedPosition | None, atr_multiple: float, reward_ratio: float
) -> dict:
    return {}


class Level(NamedTuple):
    """A level question the classifier picks a price for, the key the pick goes under, and the plan column it is stored in."""

    question: str
    key: str
    column: str


@dataclass(frozen=True)
class Rule:
    """How one Verdict is read, stored, scored and given a Minimum Confidence.

    break_even "picked" is one per Confidence band from the stored picked levels (the Dip and Cut Break-even). scope:
    whether the Minimum Confidence counts Signals server-wide or per Interval. min_confidence: the Measures field holding
    it. None for no Minimum Confidence.
    """

    plan: tuple[str, str]
    outcome: Callable[..., str | None]
    hit: str
    miss: str
    other: tuple[str, ...]
    levels: tuple[Level, ...]
    break_even: float | Literal["picked"] | None
    scope: Literal["server", "interval"] | None
    min_confidence: Literal["min_confidence", "dip_min_confidence", "cut_min_confidence"] | None
    additions: Callable[..., dict]
    exits: bool = False


_CUT = Rule(
    plan=("short_stop", "short_target"),
    outcome=partial(_outcome, long=False),
    hit="target",
    miss="stop",
    other=("expired",),
    levels=(Level("buyback", "buyback_price", "short_target"), Level("cut_stop", "cut_stop", "short_stop")),
    break_even="picked",
    # Per Interval, so one Interval's record doesn't silence another (ADR-0011).
    scope="interval",
    min_confidence="cut_min_confidence",
    additions=_cut_additions,
    exits=True,
)
RULES = {
    "buy": Rule(
        plan=("long_stop", "long_target"),
        outcome=partial(_outcome, long=True),
        hit="target",
        miss="stop",
        other=("expired",),
        levels=(),
        break_even=BREAK_EVEN_HIT_RATE,
        scope="server",
        min_confidence="min_confidence",
        additions=_buy_additions,
    ),
    "dip": Rule(
        plan=("long_stop", "long_target"),
        outcome=partial(_outcome, long=True),
        hit="target",
        miss="stop",
        other=("expired",),
        levels=(Level("dip_stop", "stop", "long_stop"), Level("dip_target", "target", "long_target")),
        break_even="picked",
        # Across all Intervals, since one alone would take long to fill (ADR-0012).
        scope="server",
        min_confidence="dip_min_confidence",
        additions=_dip_additions,
    ),
    "cut": _CUT,
    "hold": Rule(
        plan=("long_target", "short_target"),
        outcome=_hold_outcome,
        hit="quiet",
        miss="missed",
        other=(),
        levels=(),
        break_even=None,
        scope=None,
        min_confidence=None,
        additions=_no_additions,
    ),
    # Older prompt versions' sell exited the long, as cut does now.
    "sell": _CUT,
}
# Verdicts the current prompt version gives, each reported in a Track Record band.
TRACKED = ("buy", "dip", "cut", "hold")
