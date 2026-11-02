"""Unit tests for the pure prompt builder: serialized payload shape, literals as truth."""

import json

import pandas as pd
import pytest

from stock_market_mcp.prompt import DEFAULT_MAX_STATE_CHARS, build_signal_prompt


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
    prompt = build_signal_prompt("AAPL", "1mo", prompt_frame())

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

    payload = json.loads(build_signal_prompt("AAPL", "1mo", df, interval=interval))

    assert payload["rows"][-1]["Date"] == last_date


def test_prompt_serializes_nan_indicator_values_as_null():
    df = prompt_frame().assign(SMA_10=[float("nan"), float("nan"), 10.166666666666666])

    payload = json.loads(build_signal_prompt("AAPL", "1mo", df))

    assert payload["rows"][0]["SMA_10"] is None
    assert payload["rows"][2]["SMA_10"] == pytest.approx(10.166666666666666)


def test_prompt_rows_are_compact_strings_not_nested_frames():
    df = prompt_frame().assign(SMA_10=[1.0, 2.0, 3.0])

    prompt = build_signal_prompt("AAPL", "1mo", df)

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
    prompt = build_signal_prompt("AAPL", "1y", wide_frame(500), max_chars=8000)

    assert len(prompt) <= 8000
    payload = json.loads(prompt)
    assert len(payload["rows"]) < 500
    # Newest row survives; the oldest rows are the ones dropped.
    assert payload["rows"][-1]["Close"] == pytest.approx(599.0)
    assert payload["rows"][0]["Close"] > 100.0


def test_prompt_keeps_every_row_within_the_budget():
    df = wide_frame(20)

    payload = json.loads(build_signal_prompt("AAPL", "1mo", df, max_chars=1_000_000))

    assert len(payload["rows"]) == 20


def test_prompt_budget_default_and_env_override(monkeypatch):
    # Default budget sits under the discovered ~32k-token model cap; the env
    # override is read at call time like the classifier key and model id.
    small = build_signal_prompt("AAPL", "1y", wide_frame(500))
    assert len(small) <= DEFAULT_MAX_STATE_CHARS

    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "500")
    bounded = build_signal_prompt("AAPL", "1y", wide_frame(500))
    assert len(bounded) <= 500

    monkeypatch.setenv("SMM_CLASSIFIER_MAX_STATE_CHARS", "")
    assert len(build_signal_prompt("AAPL", "1y", wide_frame(500))) <= DEFAULT_MAX_STATE_CHARS  # empty env counts as unset
