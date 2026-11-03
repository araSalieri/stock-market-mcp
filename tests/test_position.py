"""Table tests on Position: Unrealised P&L, the classifier view and the Cut Loss check, on plain values."""

import math

import pytest

from stock_market_mcp.portfolio import Position


@pytest.mark.parametrize(
    ("price", "pnl", "pct"),
    [
        (100.0, 100.0, 100 / 9),  # 10 @ 90 on a cost of 900
        (80.0, -100.0, -100 / 9),
    ],
)
def test_unrealised_pnl_is_an_amount_and_a_percent_of_cost(price, pnl, pct):
    position = Position(10, 90.0)

    assert position.unrealised_pnl(price) == {
        "unrealised_pnl": pytest.approx(pnl),
        "unrealised_pnl_pct": pytest.approx(pct),
    }


@pytest.mark.parametrize("price", [math.nan, math.inf])
def test_non_finite_unrealised_pnl_is_null(price):
    assert Position(10, 90.0).unrealised_pnl(price) == {"unrealised_pnl": None, "unrealised_pnl_pct": None}


def test_classifier_view_leaves_quantity_out():
    position = Position(10, 90.0)

    assert position.classifier_view(position.unrealised_pnl(100.0)) == {
        "avg_cost": 90.0,
        "unrealised_pnl_pct": pytest.approx(100 / 9),
    }


@pytest.mark.parametrize(
    ("verdict", "avg_cost", "expected"),
    [
        ("cut", 110.0, True),  # losing
        ("cut", 90.0, False),  # profit
        ("cut", 100.0, False),  # break-even
        ("buy", 110.0, False),
        ("hold", 110.0, False),
    ],
)
def test_cuts_losing_at_a_price_of_100(verdict, avg_cost, expected):
    position = Position(10, avg_cost)

    assert position.cuts_losing(verdict, position.unrealised_pnl(100.0)) is expected


def test_a_non_finite_unrealised_pnl_is_not_losing():
    position = Position(10, 110.0)

    assert position.cuts_losing("cut", position.unrealised_pnl(math.nan)) is False
