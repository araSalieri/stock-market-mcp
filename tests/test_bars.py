"""Tests for the bars module: Interval period rules and the indicators joined onto history."""

import pytest
from fakes import FakeFetcher, flat_history, intraday_history

from stock_market_mcp.bars import history, last_value
from stock_market_mcp.errors import InputError


@pytest.mark.parametrize(
    ("interval", "period"), [("1m", "5d"), ("5m", "1mo"), ("15m", "1mo"), ("1h", "6mo"), ("1d", "1y")]
)
def test_an_intervals_default_period_is_fetched_for_the_upper_cased_ticker(interval, period):
    fetcher = FakeFetcher(history=intraday_history())

    history(fetcher, "bbca.jk", interval, None)

    assert fetcher.history_calls == [("BBCA.JK", period, interval, False)]


@pytest.mark.parametrize(
    ("interval", "period", "limit"),
    [("1m", "1mo", "7 days"), ("5m", "3mo", "60 days"), ("15m", "1y", "60 days"), ("1h", "5y", "730 days")],
)
def test_a_period_past_yahoos_limit_is_rejected_naming_it_before_any_fetch(interval, period, limit):
    fetcher = FakeFetcher(history=intraday_history())

    with pytest.raises(InputError, match=f"longer than Yahoo keeps for {interval} bars \\({limit}\\)"):
        history(fetcher, "AAPL", interval, period)

    assert fetcher.history_calls == []


@pytest.mark.parametrize(("interval", "period"), [("5m", "banana"), ("1h", "1week")])
def test_an_unknown_period_is_rejected_as_not_valid(interval, period):
    fetcher = FakeFetcher(history=intraday_history())

    with pytest.raises(InputError, match=f"not a valid period for {interval} bars") as exc:
        history(fetcher, "AAPL", interval, period)

    assert "longer than" not in str(exc.value)
    assert fetcher.history_calls == []


@pytest.mark.parametrize(("interval", "period"), [("5m", "5d"), ("1d", "max")])
def test_a_period_that_fits_the_interval_is_fetched(interval, period):
    fetcher = FakeFetcher(history=intraday_history())

    history(fetcher, "AAPL", interval, period)

    assert fetcher.history_calls == [("AAPL", period, interval, False)]


def test_too_few_rows_for_an_indicator_is_rejected():
    # 5 rows is below pandas-ta's default lookbacks, so indicators come back None.
    with pytest.raises(InputError, match="not enough history rows for indicator: "):
        history(FakeFetcher(history=flat_history(rows=5)), "AAPL", "1d", None)


def test_every_indicator_is_joined_onto_the_bars():
    df = history(FakeFetcher(history=flat_history()), "AAPL", "1d", None)

    assert {"SMA_10", "EMA_10", "RSI_14", "ATRr_14", "ADX_14", "VWAP_D"} <= set(df.columns)
    assert any(c.startswith("MACD_") for c in df.columns)
    assert any(c.startswith("BBL_") for c in df.columns)


def test_last_value_is_the_indicators_latest_reading():
    # flat_history's True Range is 2 every row, so ATR(14) is 2.
    assert last_value("atr", flat_history()) == pytest.approx(2.0)
