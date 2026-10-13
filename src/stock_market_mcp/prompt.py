"""Pure prompt builder for get_signal: history + indicators to a compact classifier prompt.

No model, no I/O — serialization only, so NaN handling and date formats are
unit-tested with literals as truth.
"""

import json

import pandas as pd


def build_signal_prompt(ticker: str, period: str, df: pd.DataFrame) -> str:
    """Serialize OHLCV history plus joined indicator columns for the classifier.

    The frame must already carry the indicator columns the tool computed; rows
    are emitted exactly the way get_history serves them: ISO date strings and
    null for NaN (warmup rows of later lookbacks).
    """
    payload = df.reset_index().to_json(orient="records", date_format="iso")
    if payload is None:
        raise ValueError("could not serialize history rows")
    return json.dumps({"ticker": ticker, "period": period, "rows": json.loads(payload)})
