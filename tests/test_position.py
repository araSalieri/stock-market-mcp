"""Table tests on pricing a Position: Unrealised P&L and losing, on plain values."""

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
    priced = Position(10, 90.0).priced(price)

    assert (priced.quantity, priced.avg_cost) == (10, 90.0)
    assert priced.unrealised_pnl == pytest.approx(pnl)
    assert priced.unrealised_pnl_pct == pytest.approx(pct)


@pytest.mark.parametrize("price", [math.nan, math.inf])
def test_non_finite_unrealised_pnl_is_null(price):
    priced = Position(10, 90.0).priced(price)

    assert (priced.unrealised_pnl, priced.unrealised_pnl_pct) == (None, None)


@pytest.mark.parametrize(
    ("avg_cost", "expected"),
    [
        (110.0, True),  # below cost
        (90.0, False),  # profit
        (100.0, False),  # break-even
    ],
)
def test_losing_at_a_price_of_100(avg_cost, expected):
    assert Position(10, avg_cost).priced(100.0).losing is expected


def test_a_non_finite_unrealised_pnl_is_not_losing():
    assert Position(10, 110.0).priced(math.nan).losing is False
