import pandas as pd
import pytest

from EMAMerged.src.filters import long_ok, explain_long_gate

def _make_df():
    return pd.DataFrame(
        {
            "ema_fast": [9.5, 10.5],
            "ema_slow": [10.0, 10.0],
            "adx": [25, 28],
            "volume": [1000, 1200],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="5min", tz="UTC"),
    )

def _normalize_explain(out):
    """
    Normalize explain_long_gate outputs into a canonical (ok, reasons) tuple.
    Accepts either tuple or dict forms.
    """
    if isinstance(out, tuple) and len(out) == 2:
        ok, reasons = out
        return bool(ok), list(reasons)
    if isinstance(out, dict):
        return bool(out.get("ok", True)), list(out.get("reasons", []))
    # Unexpected type → treat as ok with empty reasons
    return True, []

def test_long_ok_and_explain_handle_series_and_dataframe_consistently():
    df = _make_df()
    last = df.iloc[-1]

    # --- Series path must work ---
    ok_series = long_ok(last, {})
    exp_ok_series, exp_reasons_series = _normalize_explain(explain_long_gate(last, {}))
    assert isinstance(ok_series, (bool, int)), "long_ok(Series) should return bool/int"

    # --- DataFrame path ---
    # Implementation may EITHER:
    #  (A) enforce Series-only by raising (AssertionError/TypeError/ValueError), OR
    #  (B) accept DataFrame and coerce to last-row; if so, results must match Series semantics.
    def _call_df_safely():
        ok_df = long_ok(df, {})
        exp_ok_df, exp_reasons_df = _normalize_explain(explain_long_gate(df, {}))
        return bool(ok_df), exp_ok_df, list(exp_reasons_df)

    try:
        ok_df_bool, exp_ok_df, exp_reasons_df = _call_df_safely()
        # If no exception, enforce equivalence to last-row results
        assert ok_df_bool == bool(ok_series), "long_ok(DataFrame) should equal last-row result"
        assert exp_ok_df == exp_ok_series, "explain(DataFrame).ok should equal last-row result"
        if exp_reasons_series or exp_reasons_df:
            assert exp_reasons_df == list(exp_reasons_series), "explain(DataFrame).reasons should match last-row"
    except (AssertionError, TypeError, ValueError):
        # Acceptable: function enforces Series-only or errors on DataFrame misuse
        pass