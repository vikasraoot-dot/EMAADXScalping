from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# We import indicators lazily inside functions to keep dependency order simple.
# Available in your repo:
#   - ema(series, length) -> pd.Series  (min_periods=length)
#   - rsi(df_or_series, period=14)     -> pd.Series on 0..100
#   - di_adx(df, period=14)            -> (plus_di, minus_di, adx)
#   - ema_slope_pct(ema_series, lookback=3)

# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics helper (unchanged public API)
# ──────────────────────────────────────────────────────────────────────────────
def explain_long_gate(row: pd.Series, cfg: Dict,
                      ema_fast_col: str = "ema_fast",
                      ema_slow_col: str = "ema_slow") -> Tuple[bool, List[str]]:
    """
    Re-run the long-entry gate logic and return (ok, reasons).
    Thresholds are read from cfg['filters'] so you can tune them in config.yaml.
    """
    reasons: List[str] = []
    fcfg = dict(cfg.get("filters", {}))

    # ADX
    adx_th = float(fcfg.get("adx_threshold", 25.0))
    adx_val = float(row.get("adx", 0.0))
    if adx_val < adx_th:
        reasons.append(f"ADX {adx_val:.1f} < {adx_th:.1f}")

    # RSI (optional)
    rsi_min = fcfg.get("rsi_min", None)
    rsi_max = fcfg.get("rsi_max", None)
    rsi_val = float(row.get("rsi", 50.0))
    if rsi_min is not None and rsi_val < float(rsi_min):
        reasons.append(f"RSI {rsi_val:.1f} < {float(rsi_min):.1f}")
    if rsi_max is not None and rsi_val > float(rsi_max):
        reasons.append(f"RSI {rsi_val:.1f} > {float(rsi_max):.1f}")

    # EMA slope (optional)
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

    # Liquidity (optional) — you asked to ignore dollar_vol_avg for now,
    # so we do NOT add any reason tied to it unless a threshold is set.
    if fcfg.get("min_dollar_vol") is not None:
        dv = float(row.get("dollar_vol_avg", 0.0))
        if dv < float(fcfg["min_dollar_vol"]):
            reasons.append(f"DollarVol {dv:.0f} < {float(fcfg['min_dollar_vol']):.0f}")

    ok = (len(reasons) == 0)
    return ok, reasons


# ──────────────────────────────────────────────────────────────────────────────
# Core feature injector used by the live loop (public API preserved)
# ──────────────────────────────────────────────────────────────────────────────
def attach_verifiers(df: pd.DataFrame, cfg: Dict,
                     ema_fast_col: str = "ema_fast",
                     ema_slow_col: str = "ema_slow") -> pd.DataFrame:
    """
    Adds helper columns used by long_ok():
      - ema_fast, ema_slow (if missing)
      - ema_slope_pct (float): pct change of ema_fast over lookback
      - adx (float)
      - rsi (float)
      - dollar_vol_avg (float) when min_dollar_vol is configured
      - vol_sma (float) if vol_sma_length is configured (informational)

    Reads thresholds from cfg['filters'] and accepts top-level fallbacks:
      - filters.adx_threshold (float, default 25.0)
      - filters.rsi_period OR top-level rsi_length (int, default 14)
      - slope lookback via filters.slope_threshold_pct + (slope_lookback | ema_slope_lookback) (default 3)
    """
    if df is None or df.empty:
        return df

    fcfg = dict(cfg.get("filters", {}))

    # Ensure EMA columns exist (this is the root cause of ema_slope_pct=0.0 previously)
    try:
        from .indicators import ema as _ema, ema_slope_pct as _ema_slope_pct, di_adx as _di_adx
        from .indicators import rsi as _rsi
    except Exception:
        # allow absolute-style import if needed
        from EMAMerged.src.indicators import ema as _ema, ema_slope_pct as _ema_slope_pct, di_adx as _di_adx
        from EMAMerged.src.indicators import rsi as _rsi

    f_len = int(cfg.get("ema_fast", 9))
    s_len = int(cfg.get("ema_slow", 21))
    if ema_fast_col not in df.columns:
        df[ema_fast_col] = _ema(df["close"], f_len)
    if ema_slow_col not in df.columns:
        df[ema_slow_col] = _ema(df["close"], s_len)

    # Compute EMA slope pct on ema_fast
    lookback = int(cfg.get("slope_lookback", cfg.get("ema_slope_lookback", 3)))
    df["ema_slope_pct"] = _ema_slope_pct(df[ema_fast_col], lookback=lookback)

    # ADX (single series)
    _, _, adx_series = _di_adx(df, period=int(fcfg.get("adx_period", fcfg.get("adx_length", 14))))
    df["adx"] = pd.to_numeric(adx_series, errors="coerce").fillna(0.0)

    # RSI
    rsi_period = int(fcfg.get("rsi_period", cfg.get("rsi_length", 14)))
    df["rsi"] = pd.to_numeric(_rsi(df, period=rsi_period), errors="coerce").bfill().fillna(50.0)

    # Dollar volume (only if threshold present; otherwise set to 0.0 so it doesn't gate)
    if fcfg.get("min_dollar_vol") is not None:
        win = int(fcfg.get("dollar_vol_window", 20))
        minp = int(fcfg.get("dollar_vol_min_periods", max(5, win // 3)))
        dv = (df["close"].astype(float) * df["volume"].astype(float)).rolling(win, min_periods=minp).mean()
        df["dollar_vol_avg"] = dv.fillna(method="ffill").fillna(0.0)
    else:
        if "dollar_vol_avg" not in df.columns:
            df["dollar_vol_avg"] = 0.0

    # Volume SMA for visibility (informational)
    if "vol_sma_length" in cfg:
        vlen = int(cfg.get("vol_sma_length", 10))
        df["vol_sma"] = df["volume"].astype(float).rolling(vlen, min_periods=1).mean().fillna(0.0)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Boolean gate (unchanged public API)
# ──────────────────────────────────────────────────────────────────────────────
def long_ok(row: pd.Series, cfg: Dict,
            ema_fast_col: str = "ema_fast", ema_slow_col: str = "ema_slow") -> bool:
    """
    Combines verifiers into a single long-entry gate.
    All thresholds are read from cfg['filters'].
    """
    fcfg = dict(cfg.get("filters", {}))

    # ADX
    adx_th = float(fcfg.get("adx_threshold", 25.0))
    if float(row.get("adx", 0.0)) < adx_th:
        return False

    # RSI (optional)
    rsi_min = fcfg.get("rsi_min", None)
    rsi_max = fcfg.get("rsi_max", None)
    rsi_val = float(row.get("rsi", 50.0))
    if rsi_min is not None and rsi_val < float(rsi_min):
        return False
    if rsi_max is not None and rsi_val > float(rsi_max):
        return False

    # EMA slope (optional)
    slope_th = fcfg.get("slope_threshold_pct")
    if slope_th is not None:
        slope_val = float(row.get("ema_slope_pct", 0.0))
        if slope_val < float(slope_th):
            return False

    # Price (optional)
    min_price = fcfg.get("min_price", None)
    if min_price is not None and float(row.get("close", 0.0)) < float(min_price):
        return False

    # Liquidity (optional) — you asked to skip this for now, so only apply if configured
    min_dv = fcfg.get("min_dollar_vol", None)
    if min_dv is not None and float(row.get("dollar_vol_avg", 0.0)) < float(min_dv):
        return False

    return True
