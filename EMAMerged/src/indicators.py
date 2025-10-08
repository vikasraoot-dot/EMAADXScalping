# === EMAMerged/src/indicators.py ===
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Primitive indicators (Series in, Series/DataFrame out)
# ---------------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Exponential moving average (EMA) with standard Wilder-ish smoothing.
    NaNs are preserved at the head until enough data accumulates.
    """
    if series is None or len(series) == 0:
        return pd.Series(dtype='float64')
    return pd.Series(series, copy=False).ewm(span=int(period), adjust=False, min_periods=1).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder RSI (0..100).
    """
    close = pd.Series(close, copy=False)
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(up).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1/period, adjust=False).mean()

    rs = roll_up / (roll_down.replace(0.0, np.nan))
    rsi_vals = 100 - (100 / (1 + rs))
    # fill starting NaNs gracefully
    return pd.Series(rsi_vals).fillna(method='bfill')


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range using Wilder smoothing.
    Requires columns: high, low, close
    """
    if not {'high','low','close'}.issubset(df.columns):
        return pd.Series(dtype='float64')

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr_series


def di_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute +DI, -DI, and ADX (Wilder).
    Requires columns: high, low, close
    """
    if not {'high','low','close'}.issubset(df.columns):
        return pd.DataFrame(index=df.index, data={'plus_di': np.nan, 'minus_di': np.nan, 'adx': np.nan})

    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)

    up_move = high.diff()
    down_move = (-low.diff())

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([
        (high - low),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_series = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series.replace(0.0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr_series.replace(0.0, np.nan)

    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan) ) * 100
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean()

    out = pd.DataFrame(index=df.index)
    out['plus_di'] = plus_di
    out['minus_di'] = minus_di
    out['adx'] = adx
    return out


def ema_slope_pct(series: pd.Series, period: int) -> pd.Series:
    """
    Percent slope of EMA(period): (ema - ema.shift(1)) / ema.shift(1)
    """
    e = ema(series, period)
    return (e - e.shift(1)) / e.shift(1)


# ---------------------------------------------------------------------------
# DataFrame helpers that add columns in-place and return the same df
# (Used by live loop and filters to compute once per symbol)
# ---------------------------------------------------------------------------

def ta_add_emas(df: pd.DataFrame, fast: int, slow: int, col: str = 'close') -> pd.DataFrame:
    """
    Adds: ema_fast, ema_slow, ema_slope_pct
    """
    if col not in df.columns:
        return df
    ef = ema(df[col], fast)
    es = ema(df[col], slow)
    df['ema_fast'] = ef
    df['ema_slow'] = es
    # slope of fast ema
    df['ema_slope_pct'] = (ef - ef.shift(1)) / ef.shift(1)
    return df


def ta_add_rsi(df: pd.DataFrame, length: int = 14, col: str = 'close') -> pd.DataFrame:
    """
    Adds: rsi
    """
    if col not in df.columns:
        return df
    df['rsi'] = rsi(df[col], period=length)
    return df


def ta_add_adx_di(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds: adx, plus_di, minus_di
    """
    out = di_adx(df, period=period)
    for c in ('adx','plus_di','minus_di'):
        df[c] = out[c]
    return df


def ta_add_vol_dollar(df: pd.DataFrame, window: int = 20, min_periods: int = 7) -> pd.DataFrame:
    """
    Adds: dollar_vol_avg = rolling mean of (close * volume)
    """
    if not {'close','volume'}.issubset(df.columns):
        return df
    dv = (df['close'].astype(float) * df['volume'].astype(float)).rolling(window=window, min_periods=min_periods).mean()
    df['dollar_vol_avg'] = dv.fillna(0.0)
    return df


__all__ = [
    'ema','rsi','atr','di_adx','ema_slope_pct',
    'ta_add_emas','ta_add_rsi','ta_add_adx_di','ta_add_vol_dollar',
]