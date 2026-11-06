"""Unit tests for the prompt module: what the classifier reads, literals as truth."""

import json

import pandas as pd
import pytest

from stock_market_mcp.portfolio import Position
from stock_market_mcp.prompt import DEFAULT_MAX_STATE_CHARS, signal_prompt


def prompt_frame(rows: int = 3) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", name="Date")
    return pd.DataFrame(
        {
            "Open": [10.0] * rows,
            "High": [10.2] * rows,
            "Low": [9.8] * rows,
            "Close": [10.0, 10.5, 11.0],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )


def test_prompt_payload_has_ticker_period_and_iso_dates():
    prompt = signal_prompt("AAPL", "1mo", prompt_frame()).state

    payload = json.loads(prompt)
    assert payload["ticker"] == "AAPL"
    assert payload["period"] == "1mo"
    assert payload["rows"][0]["Date"] == "2024-01-01T00:00:00"
    assert payload["rows"][2]["Date"] == "2024-01-01T02:00:00"


@pytest.mark.parametrize(
    ("interval", "freq", "last_date"),
    [("1d", "D", "2024-03-12T00:00:00-04:00"), ("5m", "5min", "2024-03-12T09:40:00-04:00")],
)
def test_prompt_dates_keep_the_exchanges_offset_on_every_interval(interval, freq, last_date):
    start = "2024-03-10" if interval == "1d" else "2024-03-12 09:30"
    idx = pd.date_range(start, periods=3, freq=freq, tz="America/New_York", name="Date")
    df = prompt_frame().set_axis(idx)

    payload = json.loads(signal_prompt("AAPL", "1mo", df, interval=interval).state)

    assert payload["rows"][-1]["Date"] == last_date


def test_prompt_serializes_nan_indicator_values_as_null():
    df = prompt_frame().assign(SMA_10=[float("nan"), float("nan"), 10.166666666666666])

    payload = json.loads(signal_prompt("AAPL", "1mo", df).state)

    assert payload["rows"][0]["SMA_10"] is None
    assert payload["rows"][2]["SMA_10"] == pytest.approx(10.166666666666666)


def test_prompt_rows_are_compact_strings_not_nested_frames():
    df = prompt_frame().assign(SMA_10=[1.0, 2.0, 3.0])

    prompt = signal_prompt("AAPL", "1mo", df).state

    assert prompt.count('{"Date"') == 3  # one flat JSON object per row


def wide_frame(rows: int) -> pd.DataFrame:
    """Frame wide enough that each row's serialized form is non-trivial."""
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", name="Date")
    return pd.DataFrame({"Close": [100.0 + i for i in range(rows)]}, index=idx)


def test_prompt_truncates_oldest_rows_to_the_budget():
    # typesafe/jev-1.13 rejects states over ~32k input tokens with a 400
    # max_tokens_exceeded, so the builder bounds the payload: the MOST RECENT
    # rows are kept and the oldest are dropped – recency is the point of a
    # timing signal.
    prompt = signal_prompt("AAPL", "1y", wide_frame(500), max_chars=8000).state

    assert len(prompt) <= 8000
    payload = json.loads(prompt)
    assert len(payload["rows"]) < 500
    # Newest row survives; the oldest rows are the ones dropped.
    assert payload["rows"][-1]["Close"] == pytest.approx(599.0)
    assert payload["rows"][0]["Close"] > 100.0


def test_prompt_keeps_every_row_within_the_budget():
    df = wide_frame(20)

    payload = json.loads(signal_prompt("AAPL", "1mo", df, max_chars=1_000_000).state)

    assert len(payload["rows"]) == 20


def test_prompt_budget_default_and_env_override(monkeypatch):
    # Default budget sits under the discovered ~32k-token model cap; the env
    # override is read at call time like the classifier key and model id.
    small = signal_prompt("AAPL", "1y", wide_frame(500)).state
    assert len(small) <= DEFAULT_MAX_STATE_CHARS

    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "500")
    bounded = signal_prompt("AAPL", "1y", wide_frame(500)).state
    assert len(bounded) <= 500

    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "")
    assert len(signal_prompt("AAPL", "1y", wide_frame(500)).state) <= DEFAULT_MAX_STATE_CHARS  # empty env counts as unset


def five_minute_frame(rows: int) -> pd.DataFrame:
    """About a month of 5m bars with OHLCV plus the full indicator column set."""
    idx = pd.date_range("2024-01-02 09:30", periods=rows, freq="5min", tz="America/New_York", name="Datetime")
    base = [100.0 + i * 0.01 for i in range(rows)]
    cols = ["Open", "High", "Low", "Close", "SMA_20", "EMA_20", "RSI_14", "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9", "BBL_20_2.0", "BBM_20_2.0", "BBU_20_2.0", "ATRr_14", "OBV", "VWAP_D"]
    return pd.DataFrame({c: base for c in cols} | {"Volume": list(range(rows))}, index=idx)


def test_prompt_for_a_month_of_5m_bars_builds_fast():
    # Called directly: timing over MCP would measure the server round trip, not the builder.
    import time

    start = time.perf_counter()
    prompt = signal_prompt("AAPL", "1mo", five_minute_frame(1600), interval="5m").state
    elapsed = time.perf_counter() - start

    assert len(prompt) <= DEFAULT_MAX_STATE_CHARS
    assert elapsed < 1.0  # loose bound for CI; the old row-at-a-time loop took seconds


@pytest.mark.parametrize("max_chars", [100, 500, 2_000, 8_000])
def test_prompt_keeps_the_longest_run_of_newest_rows_that_fits(max_chars):
    # Called directly: no tool argument sets the character budget.
    df = wide_frame(300)

    payload = json.loads(signal_prompt("AAPL", "1y", df, max_chars=max_chars).state)
    kept = len(payload["rows"])

    assert kept >= 1
    assert payload["rows"][-1]["Close"] == pytest.approx(399.0)
    # One more older row would push the prompt over the budget.
    longer = json.loads(signal_prompt("AAPL", "1y", df, max_chars=1_000_000).state)
    longer["rows"] = longer["rows"][-(kept + 1):]
    assert len(json.dumps(longer)) > max_chars or kept == 300
    if kept > 1:
        assert len(json.dumps(payload)) <= max_chars


def test_prompt_for_an_empty_frame_has_no_rows():
    payload = json.loads(signal_prompt("AAPL", "1mo", prompt_frame().iloc[0:0]).state)

    assert payload == {"ticker": "AAPL", "period": "1mo", "interval": "1d", "position": None, "rows": []}


def test_the_position_the_classifier_sees_leaves_quantity_out():
    prompt = signal_prompt("AAPL", "1mo", prompt_frame(), position=Position(10, 90.0).priced(100.0))

    assert list(prompt.questions["verdict"]["criteria"]) == ["buy", "hold", "cut"]
    assert json.loads(prompt.state)["position"] == {"avg_cost": 90.0, "unrealised_pnl_pct": pytest.approx(100 / 9)}
