from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


# --- Logging helper for diagnostics only ---
def explain_long_gate(row: pd.Series, cfg: Dict,
                      ema_fast_col: str = "ema_fast",
                      ema_slow_col: str = "ema_slow") -> Tuple[bool, List[str]]:
    """
    Re-run the long entry gate logic but return (ok, reasons).
    Reads thresholds from cfg['filters'] (e.g., adx_threshold, rsi_min/max,
    slope_threshold_pct) so you can tune them in config.yaml.
    Reasons may include ADX, RSI, EMA slope, price/liquidity, and MTF bias if enabled.
    """
    reasons: List[str] = []
    fcfg = dict(cfg.get("filters", {}))

    # ADX
    adx_th = float(fcfg.get("adx_threshold", 25.0))
    adx_val = float(row.get("adx", 0.0))
    if adx_val < adx_th:
        reasons.append(f"ADX {adx_val:.1f} < {adx_th:.1f}")

    # RSI
    rsi_val = float(row.get("rsi", 0.0))
    if fcfg.get("rsi_min") is not None and rsi_val < float(fcfg["rsi_min"]):
        reasons.append(f"RSI {rsi_val:.1f} < {float(fcfg['rsi_min']):.1f}")
    if fcfg.get("rsi_max") is not None and rsi_val > float(fcfg["rsi_max"]):
        reasons.append(f"RSI {rsi_val:.1f} > {float(fcfg['rsi_max']):.1f}")

    # EMA slope
    slope_th = fcfg.get("slope_threshold_pct")
    if slope_th is not None and ("ema_slope_pct" in row.index):
        slope_val = float(row.get("ema_slope_pct", 0.0))
        if slope_val < float(slope_th):
            reasons.append(f"EMA_slope {slope_val:.5f} < {float(slope_th):.5f}")

    # Price filter
    px = float(row.get("close", float("nan")))
    if fcfg.get("min_price") is not None and pd.notna(px) and px < float(fcfg["min_price"]):
        reasons.append(f"Price {px:.2f} < {float(fcfg['min_price']):.2f}")

    # Dollar volume filter
    dv = float(row.get("dollar_vol_avg", float("nan")))
    if fcfg.get("min_dollar_vol") is not None and pd.notna(dv) and dv < float(fcfg["min_dollar_vol"]):
        reasons.append(f"DollarVol {dv:.0f} < {float(fcfg['min_dollar_vol']):.0f}")

    # Multi-timeframe bias
    mtf_cfg = fcfg.get("mtf_bias", {})
    if isinstance(mtf_cfg, dict) and mtf_cfg.get("enabled", False):
        if not bool(row.get("htf_bias", False)):
            reasons.append("MTF bias false")

    return (len(reasons) == 0), reasons


def _compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up = high.diff()
    down = -low.diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_components = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)

    tr_n = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_dm_n = pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_n = pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean()

    tr_n = tr_n.replace(0.0, np.nan)

    pdi = 100 * (plus_dm_n / tr_n)
    mdi = 100 * (minus_dm_n / tr_n)

    dx = ((pdi - mdi).abs() / (pdi + mdi).replace(0.0, np.nan)) * 100
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx.bfill().fillna(0.0)


def _ensure_rsi(df: pd.DataFrame, rsi_col: str = "rsi", period: int = 14) -> pd.DataFrame:
    """
    If RSI already exists (from strategy), keep it; otherwise compute a simple RSI(period).
    """
    if rsi_col in df.columns:
        return df

    close = df["close"].astype(float)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    df[rsi_col] = rsi.fillna(method="bfill").fillna(50.0)
    return df


def attach_verifiers(df: pd.DataFrame, cfg: Dict, ema_fast_col: str = "ema_fast", ema_slow_col: str = "ema_slow") -> pd.DataFrame:
    """
    Adds helper columns used by long_ok():
      - adx (float)
      - ema_slope_pct (float): pct change of ema_fast
      - rsi (float): if not already present
      - dollar_vol_avg (float) when min_dollar_vol is configured
      - vol_sma (float) if vol_sma_length is configured (informational)

    Reads thresholds from cfg['filters'] and accepts top-level fallbacks:
      - filters.adx_threshold (float, default 25.0)
      - filters.rsi_period OR top-level rsi_length (int, default 14)
    """
    fcfg = dict(cfg.get("filters", {}))

    # ADX
    adx_period = int(fcfg.get("adx_period", 14))
    if "adx" not in df.columns:
        df["adx"] = _compute_adx(df, period=adx_period)

    # RSI
    rsi_period = int(fcfg.get("rsi_period", cfg.get("rsi_length", 14)))
    df = _ensure_rsi(df, rsi_col="rsi", period=rsi_period)

    # EMA slope pct
    if ema_fast_col in df.columns:
        df["ema_slope_pct"] = df[ema_fast_col].pct_change().fillna(0.0)
    else:
        df["ema_slope_pct"] = 0.0

    # Dollar volume (rolling) if min_dollar_vol threshold is configured
    min_dv = fcfg.get("min_dollar_vol")
    if min_dv is not None:
        win = int(fcfg.get("dollar_vol_window", 20))
        dv = (df["close"].astype(float) * df["volume"].astype(float)).rolling(win).mean()
        df["dollar_vol_avg"] = dv.fillna(0.0)
    else:
        if "dollar_vol_avg" not in df.columns:
            df["dollar_vol_avg"] = 0.0

    # Volume SMA for visibility if requested (no gating unless you add a threshold)
    if "vol_sma_length" in cfg:
        vlen = int(cfg.get("vol_sma_length", 10))
        df["vol_sma"] = df["volume"].astype(float).rolling(vlen).mean().fillna(0.0)

    return df


def long_ok(row: pd.Series, cfg: Dict, ema_fast_col: str = "ema_fast", ema_slow_col: str = "ema_slow") -> bool:
    """
    Combines verifiers into a single long-entry gate.
    All thresholds are read from cfg['filters'] so you can tune via config.yaml.
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
    slope_th = fcfg.get("slope_threshold_pct", None)
    slope_val = float(row.get("ema_slope_pct", 0.0))
    if slope_th is not None and slope_val < float(slope_th):
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
