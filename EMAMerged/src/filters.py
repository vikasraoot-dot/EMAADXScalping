from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Optional imports from your indicators module (use if available)
# ──────────────────────────────────────────────────────────────────────────────
_E_HAS = {"ema": False, "ema_slope_pct": False, "rsi": False, "adx": False, "di_adx": False}

try:
    from .indicators import ema as _ext_ema
    _E_HAS["ema"] = True
except Exception:
    _ext_ema = None

try:
    from .indicators import ema_slope_pct as _ext_ema_slope_pct
    _E_HAS["ema_slope_pct"] = True
except Exception:
    _ext_ema_slope_pct = None

# ADX may be exposed in one of two shapes in various snapshots:
# (1) adx(df, period) -> Series
# (2) di_adx(df, period) -> (+DI, -DI, ADX)
try:
    from .indicators import adx as _ext_adx
    _E_HAS["adx"] = True
except Exception:
    _ext_adx = None

try:
    from .indicators import di_adx as _ext_di_adx
    _E_HAS["di_adx"] = True
except Exception:
    _ext_di_adx = None

try:
    from .indicators import rsi as _ext_rsi
    _E_HAS["rsi"] = True
except Exception:
    _ext_rsi = None


# ──────────────────────────────────────────────────────────────────────────────
# In-module indicator fallbacks (used when imports above are missing)
# ──────────────────────────────────────────────────────────────────────────────
def _fb_ema(x: pd.Series, length: int) -> pd.Series:
    return pd.Series(pd.to_numeric(x, errors="coerce")).ewm(span=max(1, int(length)), adjust=False, min_periods=1).mean()

def _fb_ema_slope_pct(ema_series: pd.Series, lookback: int = 3) -> pd.Series:
    ema_series = pd.to_numeric(ema_series, errors="coerce")
    prev = ema_series.shift(max(1, int(lookback)))
    out = (ema_series - prev) / prev.abs()
    return out.fillna(0.0)

def _fb_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    close = pd.to_numeric(df["close"], errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    n = max(1, int(period))

    # Wilder's RMA
    rma_gain = gain.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rma_loss = loss.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = rma_gain / rma_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50.0)

def _fb_wilder_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Wilder's ADX, computed from df[['high','low','close']].
    Returns a Series aligned to df.index.
    """
    n = max(1, int(period))
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")

    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing (RMA)
    tr_rma = pd.Series(tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_dm_rma = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    minus_dm_rma = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

    plus_di = 100 * (plus_dm_rma / tr_rma.replace(0.0, np.nan))
    minus_di = 100 * (minus_dm_rma / tr_rma.replace(0.0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    return pd.to_numeric(adx, errors="coerce").fillna(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Gate explanation and checks
# ──────────────────────────────────────────────────────────────────────────────
def explain_long_gate(row: pd.Series, cfg: Dict,
                      ema_fast_col: str = "ema_fast",
                      ema_slow_col: str = "ema_slow") -> Tuple[bool, List[str]]:
    """
    Return (ok, reasons) explaining why a row would be blocked by long_ok().
    Uses thresholds from cfg['filters'].
    """
    reasons: List[str] = []
    fcfg = dict(cfg.get("filters", {}))

    # Require fast EMA above slow EMA?
    require_fast = bool(fcfg.get("require_fast_above_slow", False))
    if require_fast:
        ef = float(row.get(ema_fast_col, np.nan))
        es = float(row.get(ema_slow_col, np.nan))
        if not (np.isfinite(ef) and np.isfinite(es) and ef > es):
            reasons.append("ema_fast ≤ ema_slow")

    # ADX
    adx_th = float(fcfg.get("adx_threshold", 25.0))
    adx_val = float(row.get("adx", 0.0))
    if adx_val < adx_th:
        reasons.append(f"ADX {adx_val:.1f} < {adx_th:.1f}")

    # RSI window
    rsi_min = fcfg.get("rsi_min", None)
    rsi_max = fcfg.get("rsi_max", None)
    rsi_val = float(row.get("rsi", 50.0))
    if rsi_min is not None and rsi_val < float(rsi_min):
        reasons.append(f"RSI {rsi_val:.1f} < {float(rsi_min):.1f}")
    if rsi_max is not None and rsi_val > float(rsi_max):
        reasons.append(f"RSI {rsi_val:.1f} > {float(rsi_max):.1f}")

    # EMA slope pct
    slope_th = fcfg.get("slope_threshold_pct")
    if slope_th is not None and ("ema_slope_pct" in row.index):
        slope_val = float(row.get("ema_slope_pct", 0.0))
        if slope_val < float(slope_th):
            reasons.append(f"EMA_slope {slope_val:.5f} < {float(slope_th):.5f}")

    # Price (optional)
    if fcfg.get("min_price") is not None:
        px = float(row.get("close", float("nan")))
        if not np.isnan(px) and px < float(fcfg["min_price"]):
            reasons.append(f"Price {px:.2f} < {float(fcfg['min_price']):.2f}")

    # Liquidity (optional) — only enforce if configured
    if fcfg.get("min_dollar_vol") is not None:
        dv = float(row.get("dollar_vol_avg", 0.0))
        if dv < float(fcfg["min_dollar_vol"]):
            reasons.append(f"DollarVol {dv:.0f} < {float(fcfg['min_dollar_vol']):.0f}")

    ok = (len(reasons) == 0)
    return ok, reasons


def attach_verifiers(df: pd.DataFrame, cfg: Dict,
                     ema_fast_col: str = "ema_fast",
                     ema_slow_col: str = "ema_slow") -> pd.DataFrame:
    """
    Ensures indicator columns needed by long_ok():
      - ema_fast / ema_slow (if missing)
      - ema_slope_pct (pct change over lookback on ema_fast)
      - adx, rsi
      - dollar_vol_avg (only created here if missing and configured)

    Reads periods/thresholds from cfg and cfg['filters'].
    """
    if df is None or df.empty:
        return df

    fcfg = dict(cfg.get("filters", {}))

    # Choose EMA & helpers (external if available, else fallbacks)
    def _EMA(x: pd.Series, length: int) -> pd.Series:
        if _E_HAS["ema"] and _ext_ema is not None:
            try:
                return _ext_ema(x, length)
            except Exception:
                pass
        return _fb_ema(x, length)

    def _EMA_SLOPE_PCT(x: pd.Series, lookback: int) -> pd.Series:
        if _E_HAS["ema_slope_pct"] and _ext_ema_slope_pct is not None:
            try:
                return _ext_ema_slope_pct(x, lookback=lookback)
            except Exception:
                pass
        return _fb_ema_slope_pct(x, lookback=lookback)

    def _RSI(df_: pd.DataFrame, period: int) -> pd.Series:
        if _E_HAS["rsi"] and _ext_rsi is not None:
            try:
                return _ext_rsi(df_, period=period)
            except Exception:
                pass
        return _fb_rsi(df_, period=period)

    def _ADX(df_: pd.DataFrame, period: int) -> pd.Series:
        # Try adx(df, period) -> Series
        if _E_HAS["adx"] and _ext_adx is not None:
            try:
                ser = _ext_adx(df_, period=period)
                return pd.to_numeric(ser, errors="coerce")
            except Exception:
                pass
        # Try di_adx(df, period) -> (+DI, -DI, ADX)
        if _E_HAS["di_adx"] and _ext_di_adx is not None:
            try:
                _, _, adx_ser = _ext_di_adx(df_, period=period)
                return pd.to_numeric(adx_ser, errors="coerce")
            except Exception:
                pass
        # Fallback: local Wilder ADX
        return _fb_wilder_adx(df_, period=period)

    # Ensure EMAs exist for slope computation
    f_len = int(cfg.get("ema_fast", 9))
    s_len = int(cfg.get("ema_slow", 21))
    if ema_fast_col not in df.columns:
        df[ema_fast_col] = _EMA(df["close"], f_len)
    if ema_slow_col not in df.columns:
        df[ema_slow_col] = _EMA(df["close"], s_len)

    # EMA slope pct on ema_fast
    lookback = int(cfg.get("slope_lookback", cfg.get("ema_slope_lookback", 3)))
    df["ema_slope_pct"] = _EMA_SLOPE_PCT(df[ema_fast_col], lookback=lookback)

    # ADX (robust to whichever indicator signatures exist)
    adx_period = int(fcfg.get("adx_period", fcfg.get("adx_length", 14)))
    df["adx"] = _ADX(df, period=adx_period).fillna(0.0)

    # RSI
    rsi_period = int(fcfg.get("rsi_period", cfg.get("rsi_length", 14)))
    df["rsi"] = pd.to_numeric(_RSI(df, period=rsi_period), errors="coerce").bfill().fillna(50.0)

    # Dollar volume: Option-B already computes this upstream; only fill if missing and configured
    if "dollar_vol_avg" not in df.columns:
        if fcfg.get("min_dollar_vol") is not None:
            win = int(fcfg.get("dollar_vol_window", 20))
            minp = int(fcfg.get("dollar_vol_min_periods", max(5, win // 3)))
            dv = (df["close"].astype(float) * df["volume"].astype(float)).rolling(win, min_periods=minp).mean()
            df["dollar_vol_avg"] = dv.fillna(method="ffill").fillna(0.0)
        else:
            df["dollar_vol_avg"] = 0.0

    # Volume SMA (optional info only)
    if "vol_sma_length" in cfg:
        vlen = int(cfg.get("vol_sma_length", 10))
        df["vol_sma"] = df["volume"].astype(float).rolling(vlen, min_periods=1).mean().fillna(0.0)

    return df


def long_ok(row: pd.Series, cfg: Dict,
            ema_fast_col: str = "ema_fast", ema_slow_col: str = "ema_slow") -> bool:
    """
    Main long-entry gate. Thresholds in cfg['filters'].
    """
    fcfg = dict(cfg.get("filters", {}))

    # Require fast > slow?
    require_fast = bool(fcfg.get("require_fast_above_slow", False))
    if require_fast:
        ef = float(row.get(ema_fast_col, 0.0))
        es = float(row.get(ema_slow_col, 0.0))
        if not (ef > es):
            return False

    # ADX
    adx_th = float(fcfg.get("adx_threshold", 25.0))
    if float(row.get("adx", 0.0)) < adx_th:
        return False

    # RSI window
    rsi_min = fcfg.get("rsi_min", None)
    rsi_max = fcfg.get("rsi_max", None)
    rsi_val = float(row.get("rsi", 50.0))
    if rsi_min is not None and rsi_val < float(rsi_min):
        return False
    if rsi_max is not None and rsi_val > float(rsi_max):
        return False

    # EMA slope threshold
    slope_th = fcfg.get("slope_threshold_pct")
    if slope_th is not None:
        slope_val = float(row.get("ema_slope_pct", 0.0))
        if slope_val < float(slope_th):
            return False

    # Price (optional)
    min_price = fcfg.get("min_price", None)
    if min_price is not None and float(row.get("close", 0.0)) < float(min_price):
        return False

    # Liquidity (optional)
    min_dv = fcfg.get("min_dollar_vol", None)
    if min_dv is not None and float(row.get("dollar_vol_avg", 0.0)) < float(min_dv):
        return False

    return True
