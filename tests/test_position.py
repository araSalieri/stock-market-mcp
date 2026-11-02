"""Table tests on Position (#78): Unrealised P&L, the classifier view and the Cut Loss check, on plain values."""

import math

import pytest

from stock_market_mcp.portfolio import Position


@pytest.mark.parametrize(
    ("direction", "price", "pnl", "pct"),
    [
        ("long", 100.0, 100.0, 100 / 9),  # 10 @ 90 on a cost of 900
        ("long", 80.0, -100.0, -100 / 9),
        ("short", 100.0, -100.0, -100 / 9),
        ("short", 80.0, 100.0, 100 / 9),
    ],
)
def test_unrealised_pnl_is_an_amount_and_a_percent_of_cost(direction, price, pnl, pct):
    position = Position(direction, 10, 90.0)

    assert position.unrealised_pnl(price) == {
        "unrealised_pnl": pytest.approx(pnl),
        "unrealised_pnl_pct": pytest.approx(pct),
    }


@pytest.mark.parametrize("price", [math.nan, math.inf])
def test_non_finite_unrealised_pnl_is_null(price):
    assert Position("long", 10, 90.0).unrealised_pnl(price) == {"unrealised_pnl": None, "unrealised_pnl_pct": None}


def test_classifier_view_leaves_quantity_out():
    position = Position("short", 10, 90.0)

    assert position.classifier_view(position.unrealised_pnl(100.0)) == {
        "direction": "short",
        "avg_cost": 90.0,
        "unrealised_pnl_pct": pytest.approx(-100 / 9),
    }


@pytest.mark.parametrize(
    ("direction", "verdict", "avg_cost", "expected"),
    [
        ("long", "sell", 110.0, True),  # losing, opposing
        ("short", "buy", 90.0, True),
        ("long", "sell", 90.0, False),  # profit
        ("short", "buy", 110.0, False),
        ("long", "sell", 100.0, False),  # break-even
        ("long", "buy", 110.0, False),  # agreeing
        ("short", "sell", 90.0, False),
        ("long", "hold", 110.0, False),
        ("short", "hold", 90.0, False),
    ],
)
def test_opposes_losing_at_a_price_of_100(direction, verdict, avg_cost, expected):
    position = Position(direction, 10, avg_cost)

    assert position.opposes_losing(verdict, position.unrealised_pnl(100.0)) is expected


def test_a_non_finite_unrealised_pnl_is_not_losing():
    position = Position("long", 10, 110.0)

    assert position.opposes_losing("sell", position.unrealised_pnl(math.nan)) is False
