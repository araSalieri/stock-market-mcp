"""Unit tests for the pure prompt builder: serialized payload shape, literals as truth."""

import json

import pandas as pd
import pytest

from stock_market_mcp.prompt import build_signal_prompt


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
    assert payload["rows"][0]["Date"] == "2024-01-01T00:00:00.000"
    assert payload["rows"][2]["Date"] == "2024-01-01T02:00:00.000"


def test_prompt_serializes_nan_indicator_values_as_null():
    df = prompt_frame().assign(SMA_10=[float("nan"), float("nan"), 10.166666666666666])

    payload = json.loads(build_signal_prompt("AAPL", "1mo", df))

    assert payload["rows"][0]["SMA_10"] is None
    assert payload["rows"][2]["SMA_10"] == pytest.approx(10.166666666666666)


def test_prompt_rows_are_compact_strings_not_nested_frames():
    df = prompt_frame().assign(SMA_10=[1.0, 2.0, 3.0])

    prompt = build_signal_prompt("AAPL", "1mo", df)

    assert prompt.count('{"Date"') == 3  # one flat JSON object per row
