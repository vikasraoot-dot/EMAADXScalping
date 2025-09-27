import pandas as pd
import pytest

from EMAMerged.src.filters import long_ok, explain_long_gate

def _make_df():
    return pd.DataFrame({
        "ema_fast": [9.5, 10.5],
        "ema_slow": [10.0, 10.0],
        "adx": [25, 28],
        "volume": [1000, 1200],
    }, index=pd.date_range("2025-01-01", periods=2, freq="5min"))

def test_long_ok_and_explain_accept_series_not_dataframe():
    df = _make_df()
    last_row = df.iloc[-1]

    # Should accept Series
    ok = long_ok(last_row, {})
    reasons = explain_long_gate(last_row, {})
    assert isinstance(ok, (bool, int)), "long_ok should return bool/int when passed Series"
    assert isinstance(reasons, dict), "explain_long_gate should return dict when passed Series"

    # Passing DataFrame should either raise AssertionError or return same result as Series
    with pytest.raises(AssertionError):
        long_ok(df, {})

    with pytest.raises(AssertionError):
        explain_long_gate(df, {})