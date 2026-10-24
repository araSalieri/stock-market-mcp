"""Shared test fakes importable by any test module."""

import pandas as pd


def fake_history(rows: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", name="Date")
    close = pd.Series([10.0 + i * 0.5 + (i % 3) for i in range(rows)], index=idx)
    return pd.DataFrame(
        {
            "Open": close - 0.1,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": [1000 + i for i in range(rows)],
        }
    )
