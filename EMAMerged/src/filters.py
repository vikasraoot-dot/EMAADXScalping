from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


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

    try:
        from .indicators import ema as _ema, ema_slope_pct as _ema_slope_pct, di_adx as _di_adx
        from .indicators import rsi as _rsi
    except Exception:
        from EMAMerged.src.indicators import ema as _ema, ema_slope_pct as _ema_slope_pct, di_adx as _di_adx
        from EMAMerged.src.indicators import rsi as _rsi

    # Ensure EMAs exist for slope computation
    f_len = int(cfg.get("ema_fast", 9))
    s_len = int(cfg.get("ema_slow", 21))
    if ema_fast_col not in df.columns:
        df[ema_fast_col] = _ema(df["close"], f_len)
    if ema_slow_col not in df.columns:
        df[ema_slow_col] = _ema(df["close"], s_len)

    # EMA slope pct on ema_fast
    lookback = int(cfg.get("slope_lookback", cfg.get("ema_slope_lookback", 3)))
    df["ema_slope_pct"] = _ema_slope_pct(df[ema_fast_col], lookback=lookback)

    # ADX
    _, _, adx_series = _di_adx(df, period=int(fcfg.get("adx_period", fcfg.get("adx_length", 14))))
    df["adx"] = pd.to_numeric(adx_series, errors="coerce").fillna(0.0)

    # RSI
    rsi_period = int(fcfg.get("rsi_period", cfg.get("rsi_length", 14)))
    df["rsi"] = pd.to_numeric(_rsi(df, period=rsi_period), errors="coerce").bfill().fillna(50.0)

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
