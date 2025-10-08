from __future__ import annotations

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Small utilities
# ──────────────────────────────────────────────────────────────────────────────
def _as_1d_series(obj: pd.Series | pd.DataFrame, col: str | None = None) -> pd.Series:
    """
    Return a 1D float series regardless of whether a Series or a DataFrame was passed.
    If a DataFrame is provided and `col` is None, use the first column.
    """
    if isinstance(obj, pd.Series):
        s = obj
    else:
        if col is None:
            col = obj.columns[0]
        s = obj[col]
    return pd.to_numeric(s, errors="coerce")


# ──────────────────────────────────────────────────────────────────────────────
# Core indicators already used elsewhere in the repo
# ──────────────────────────────────────────────────────────────────────────────
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    high = _as_1d_series(high).astype("float64")
    low = _as_1d_series(low).astype("float64")
    close = _as_1d_series(close).astype("float64")

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder). Expects df with columns: 'high','low','close'.
    """
    tr = true_range(df["high"], df["low"], df["close"])
    # Wilder smoothing via EMA(alpha=1/period)
    atr_series = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr_series


def ema(series: pd.Series | pd.DataFrame, length: int) -> pd.Series:
    s = _as_1d_series(series)
    return s.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(series: pd.Series | pd.DataFrame, length: int = 14) -> pd.Series:
    s = _as_1d_series(series).astype("float64")
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    roll_up = up.ewm(alpha=1/length, adjust=False).mean()
    roll_down = down.ewm(alpha=1/length, adjust=False).mean()

    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out


def adx_di(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Wilder +DI/-DI/ADX. Expects df with 'high','low','close'.
    Returns: (plus_di, minus_di, adx)
    """
    high = _as_1d_series(df["high"]).astype("float64")
    low = _as_1d_series(df["low"]).astype("float64")
    close = _as_1d_series(df["close"]).astype("float64")

    up_move = high.diff()
    down_move = (-low.diff())

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0.0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0.0), 0.0)

    tr = true_range(high, low, close)

    # Wilder smoothing
    atr_sm = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    plus_dm_sm = plus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    plus_di = 100.0 * (plus_dm_sm / atr_sm.replace(0.0, np.nan))
    minus_di = 100.0 * (minus_dm_sm / atr_sm.replace(0.0, np.nan))

    dx = (100.0 * (plus_di - minus_di).abs() /
          (plus_di + minus_di).replace(0.0, np.nan))
    adx = dx.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

    return plus_di, minus_di, adx


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Convenience wrapper returning only ADX.
    """
    _, _, a = adx_di(df, period=period)
    return a


def ema_slope_pct(series: pd.Series | pd.DataFrame, lookback: int = 1) -> pd.Series:
    """
    % slope over `lookback` bars for the provided series.
    """
    s = _as_1d_series(series).astype("float64")
    return s.pct_change(periods=lookback)


def ema_slope_atr(series: pd.Series | pd.DataFrame, atr_series: pd.Series,
                  lookback: int = 1) -> pd.Series:
    """
    Slope normalized by ATR: |Δseries| / ATR over `lookback` bars.
    """
    s = _as_1d_series(series).astype("float64")
    delta = (s - s.shift(lookback)).abs()
    denom = atr_series.replace(0.0, np.nan)
    return (delta / denom)


# ──────────────────────────────────────────────────────────────────────────────
# NEW: Thin “plumbing” helpers to satisfy imports (minimal churn)
# ──────────────────────────────────────────────────────────────────────────────
def ta_add_emas(df: pd.DataFrame, fast: int, slow: int,
                slope_lookback: int = 1) -> pd.DataFrame:
    """
    Append EMA-based columns to `df` in-place-friendly style:
      - 'ema_fast'
      - 'ema_slow'
      - 'ema_slope_pct'  (slope of the fast EMA over `slope_lookback` bars)

    Returns the same DataFrame (for chaining).
    """
    df["ema_fast"] = ema(df["close"], fast)
    df["ema_slow"] = ema(df["close"], slow)
    # slope of the fast EMA; this is what your gates already expect
    df["ema_slope_pct"] = ema_slope_pct(df["ema_fast"], lookback=slope_lookback)
    return df


def ta_add_rsi_adx(df: pd.DataFrame, rsi_period: int = 14, adx_period: int = 14) -> pd.DataFrame:
    """
    Append momentum/trend columns to `df`:
      - 'rsi'
      - 'plus_di'
      - 'minus_di'
      - 'adx'

    Returns the same DataFrame (for chaining).
    """
    df["rsi"] = rsi(df["close"], length=rsi_period)
    plus_di, minus_di, adx_series = adx_di(df, period=adx_period)
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    df["adx"] = adx_series
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Export surface
# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    # primitives
    "true_range", "atr", "ema", "rsi",
    "adx", "adx_di",
    "ema_slope_pct", "ema_slope_atr",
    # new wrappers (fix imports; minimal churn)
    "ta_add_emas", "ta_add_rsi_adx",
]