# === EMAMerged/src/indicators.py ===
from __future__ import annotations
import numpy as np
import pandas as pd

# -------- helpers --------
def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("expected DataFrame")
    if not {"high", "low", "close"}.issubset(df.columns):
        missing = {"high", "low", "close"} - set(df.columns)
        raise KeyError(f"DataFrame is missing columns: {sorted(missing)}")
    if df.index.tz is None:
        # keep it simple: assume UTC if tz-naive
        df = df.set_index(pd.DatetimeIndex(df.index, tz="UTC"))
    return df

# -------- EMAs --------
def ta_add_emas(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    fast = max(1, int(fast))
    slow = max(1, int(slow))
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False, min_periods=fast).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
    return df

# -------- RSI (Wilder) --------
def ta_add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    df = _ensure_df(df).copy()
    length = max(1, int(length))
    delta = df["close"].diff()

    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    # Wilder's RMA via EWM(alpha=1/n)
    avg_gain = gain.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # fix FutureWarning: don't use fillna(method=...)
    df["rsi"] = rsi.bfill().ffill()
    return df

# -------- ADX / +DI / -DI (Wilder) --------
def ta_add_adx_di(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Adds '+DI', '-DI', and 'ADX' columns (plus lowercase aliases for safety).
    """
    df = _ensure_df(df).copy()
    period = max(1, int(period))

    high = df["high"]
    low = df["low"]
    close = df["close"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # Wilder smoothing via EWM(alpha=1/period)
    atr = tr.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    pos_dm_sm = pd.Series(pos_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    neg_dm_sm = pd.Series(neg_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

    pdi = 100.0 * (pos_dm_sm / atr.replace(0.0, np.nan))
    mdi = 100.0 * (neg_dm_sm / atr.replace(0.0, np.nan))

    dx = 100.0 * ( (pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan) )
    adx = dx.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

    # Fill mild leading NaNs without deprecated fillna(method=...)
    pdi = pdi.bfill().ffill()
    mdi = mdi.bfill().ffill()
    adx = adx.bfill().ffill()

    # Expose EXACT column names the live loop expects
    df["+DI"] = pdi.astype(float)
    df["-DI"] = mdi.astype(float)
    df["ADX"] = adx.astype(float)

    # Also expose lowercase aliases (harmless, improves compatibility)
    df["pdi"] = df["+DI"]
    df["mdi"] = df["-DI"]
    df["adx"] = df["ADX"]

    return df

# -------- Dollar-volume SMA (utility) --------
def ta_add_vol_dollar(df: pd.DataFrame, window: int = 10, min_periods: int = None) -> pd.DataFrame:
    """
    Adds 'dollar_vol_avg' = SMA(close*volume, window).
    """
    df = _ensure_df(df).copy()
    window = max(1, int(window))
    if min_periods is None:
        min_periods = max(1, window // 2)
    dv = (df["close"] * df["volume"]).rolling(window=window, min_periods=min_periods).mean()
    df["dollar_vol_avg"] = dv.ffill().fillna(0.0)
    return df