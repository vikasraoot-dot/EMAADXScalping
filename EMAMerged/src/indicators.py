import numpy as np
import pandas as pd

def _as_1d_series(x: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            col = "close" if "close" in x.columns else x.columns[0]
            x = x[col]
        else:
            x = x.iloc[:, 0]
    return pd.to_numeric(pd.Series(x), errors="coerce")

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
    return out.fillna(0.0)

def atr(df_or_high, low: pd.Series | None = None, close: pd.Series | None = None, period: int = 14) -> pd.Series:
    if isinstance(df_or_high, pd.DataFrame):
        h = _as_1d_series(df_or_high["high"]).astype("float64")
        l = _as_1d_series(df_or_high["low"]).astype("float64")
        c = _as_1d_series(df_or_high["close"]).astype("float64")
    else:
        h = _as_1d_series(df_or_high).astype("float64")
        l = _as_1d_series(low).astype("float64")
        c = _as_1d_series(close).astype("float64")
    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    idx = df.index
    if idx.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    eastern_idx = df.index.tz_convert("US/Eastern")
    dates = pd.to_datetime(eastern_idx.date)
    high = _as_1d_series(df["high"]).astype("float64")
    low = _as_1d_series(df["low"]).astype("float64")
    close = _as_1d_series(df["close"]).astype("float64")
    vol = _as_1d_series(df["volume"]).fillna(0).astype("float64")
    typical = (high + low + close) / 3.0
    grouped = pd.DataFrame({"tpv": typical * vol, "vol": vol, "date": dates}, index=df.index)
    ctpv = grouped.groupby("date")["tpv"].cumsum()
    cvol = grouped.groupby("date")["vol"].cumsum().replace({0: np.nan})
    out = ctpv / cvol
    out.index = idx
    return out

def adx_di(high: pd.Series | pd.DataFrame, low: pd.Series | None = None,
           close: pd.Series | None = None, length: int = 14):
    if isinstance(high, pd.DataFrame):
        df = high
        h = _as_1d_series(df["high"]).astype("float64")
        l = _as_1d_series(df["low"]).astype("float64")
        c = _as_1d_series(df["close"]).astype("float64")
    else:
        h = _as_1d_series(high).astype("float64")
        l = _as_1d_series(low).astype("float64")
        c = _as_1d_series(close).astype("float64")

    up_move   = h.diff()
    down_move = (-l.diff())
    plus_dm  = (up_move.where((up_move > down_move) & (up_move > 0), 0.0)).abs()
    minus_dm = (down_move.where((down_move > up_move) & (down_move > 0), 0.0)).abs()

    pc = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

    plus_di  = (100.0 * plus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean()  / atr_).replace([np.inf, -np.inf], np.nan)
    minus_di = (100.0 * minus_dm.ewm(alpha=1/length, adjust=False, min_periods=length).mean() / atr_).replace([np.inf, -np.inf], np.nan)

    dx  = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx = dx.ewm(alpha=1/length, adjust=False, min_periods=length).mean().fillna(0.0)

    return plus_di.fillna(0.0), minus_di.fillna(0.0), adx.fillna(0.0)

def ema_slope_pct(ema_series: pd.Series, lookback: int = 3) -> pd.Series:
    s = _as_1d_series(ema_series).astype("float64")
    prev = s.shift(lookback)
    pct = (s - prev) / prev.replace(0, np.nan)
    return pct.fillna(0.0)

def ema_slope_atr(ema_series: pd.Series, atr_series: pd.Series, lookback: int = 3) -> pd.Series:
    s = _as_1d_series(ema_series).astype("float64")
    a = _as_1d_series(atr_series).astype("float64")
    prev = s.shift(lookback)
    return ((s - prev) / a.replace(0, np.nan)).fillna(0.0)
