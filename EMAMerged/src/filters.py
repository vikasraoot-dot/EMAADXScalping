from __future__ import annotations
from typing import Dict, Any
import pandas as pd
from .indicators import adx_di, ema_slope_pct, ema_slope_atr, rsi as rsi_ind

def attach_verifiers(df: pd.DataFrame, cfg: Dict[str, Any],
                     ema_fast_col: str = "ema_fast", ema_slow_col: str = "ema_slow") -> pd.DataFrame:
    out = df.copy()
    fcfg = cfg.get("filters", {})

    # ADX / DI
    adx_len = int(fcfg.get("adx_len", 14))
    plus_di, minus_di, adx = adx_di(out["high"], out["low"], out["close"], length=adx_len)
    out["plus_di"] = plus_di
    out["minus_di"] = minus_di
    out["adx"] = adx

    # RSI
    rsi_period = int(fcfg.get("rsi_period", cfg.get("rsi_period", 14)))
    out["rsi"] = rsi_ind(out["close"], length=rsi_period)

    # Slopes
    lookback = int(fcfg.get("slope_lookback", 3))
    out["ema_fast_slope_pct"] = ema_slope_pct(out[ema_fast_col], lookback=lookback)
    out["ema_slow_slope_pct"] = ema_slope_pct(out[ema_slow_col], lookback=lookback)
    if "atr" in out.columns:
        out["ema_fast_slope_atr"] = ema_slope_atr(out[ema_fast_col], out["atr"], lookback=lookback)
        out["ema_slow_slope_atr"] = ema_slope_atr(out[ema_slow_col], out["atr"], lookback=lookback)

    return out

def long_ok(row: pd.Series, cfg: Dict[str, Any],
            ema_fast_col: str = "ema_fast", ema_slow_col: str = "ema_slow") -> bool:
    fcfg = cfg.get("filters", {})
    if not bool(fcfg.get("enabled", True)):
        return True

    # ADX/DI
    if float(row.get("adx", 0.0)) < float(fcfg.get("adx_threshold", 22)):
        return False
    if bool(fcfg.get("require_plus_di_over_minus", True)):
        margin = float(fcfg.get("di_margin", 0.0))
        if float(row.get("plus_di", 0.0)) < float(row.get("minus_di", 0.0)) + margin:
            return False

    # RSI
    rsi_min = float(fcfg.get("rsi_min", 50))
    rsi_max = float(fcfg.get("rsi_max", 100))
    r = float(row.get("rsi", 0.0))
    if (r < rsi_min) or (r > rsi_max):
        return False

    # Slope
    mode = str(fcfg.get("slope_mode", "percent")).lower()
    require_ema21_up = bool(fcfg.get("require_ema21_slope_up", True))
    if mode == "atr":
        thr = float(fcfg.get("slope_threshold_atr", 0.15))
        fast_ok = float(row.get("ema_fast_slope_atr", 0.0)) >= thr
        slow_ok = float(row.get("ema_slow_slope_atr", 0.0)) > 0.0 if require_ema21_up else True
    else:
        thr = float(fcfg.get("slope_threshold_pct", 0.0012))
        fast_ok = float(row.get("ema_fast_slope_pct", 0.0)) >= thr
        slow_ok = float(row.get("ema_slow_slope_pct", 0.0)) > 0.0 if require_ema21_up else True
    if not (fast_ok and slow_ok):
        return False

    # Optional separation
    sep_bps = float(fcfg.get("ema_separation_bps", 0.0))
    if sep_bps > 0:
        fast = float(row.get(ema_fast_col, 0.0))
        slow = float(row.get(ema_slow_col, 0.0))
        if slow != 0:
            sep = abs(fast - slow) / abs(slow)
            if sep < (sep_bps / 10000.0):
                return False

    return True
