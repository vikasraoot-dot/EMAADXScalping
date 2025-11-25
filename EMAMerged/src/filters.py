from __future__ import annotations
import math
from typing import Dict, Any
import pandas as pd
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Core indicators (implemented locally to avoid fragile cross-module imports)
# ──────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=1).mean()

def _ema_slope_pct(ema: pd.Series) -> pd.Series:
    prev = ema.shift(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (ema - prev) / prev
    return slope.fillna(0.0)

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    loss = dn.ewm(alpha=1.0/length, adjust=False, min_periods=length).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # pandas deprecates fillna(method="bfill") on Series; use bfill()/ffill()
    return rsi.bfill().fillna(50.0)

def _di_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Wilder's +DI, -DI, ADX. Returns dataframe with +di, -di, adx columns.
    Requires 'high','low','close'.
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr_components = pd.concat([
        (high - low).abs(),
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)

    # Wilder smoothing via EMA with alpha = 1/period (close enough for trading)
    tr_n = pd.Series(tr).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    plus_dm_n = pd.Series(plus_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()
    minus_dm_n = pd.Series(minus_dm, index=df.index).ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        plus_di = 100.0 * (plus_dm_n / tr_n)
        minus_di = 100.0 * (minus_dm_n / tr_n)
        dx = 100.0 * ( (plus_di - minus_di).abs() / (plus_di + minus_di) )

    adx = dx.ewm(alpha=1.0/period, adjust=False, min_periods=period).mean()

    out = pd.DataFrame({
        "+di": plus_di.fillna(0.0),
        "-di": minus_di.fillna(0.0),
        "adx": adx.fillna(0.0),
    }, index=df.index)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Public API used by live loop
# ──────────────────────────────────────────────────────────────────────────────

def attach_verifiers(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Adds columns the gate uses:
      ema_fast, ema_slow, ema_slope_pct, rsi, +di, -di, adx, fresh_cross_up, fresh_cross_down.
    Leaves existing cols untouched; returns a new DataFrame (same index).
    """
    df = df.copy()

    fcfg = dict(cfg.get("filters", {}))
    ema_fast_len = int(cfg.get("ema_fast", 9))
    ema_slow_len = int(cfg.get("ema_slow", 21))
    rsi_len = int(cfg.get("rsi_length", fcfg.get("rsi_period", 14)))
    adx_period = int(fcfg.get("adx_period", cfg.get("adx_period", 14)))

    # EMAs
    df["ema_fast"] = _ema(df["close"], ema_fast_len)
    df["ema_slow"] = _ema(df["close"], ema_slow_len)

    # Slope (% per bar) on fast EMA
    df["ema_slope_pct"] = _ema_slope_pct(df["ema_fast"]).fillna(0.0)

    # RSI
    df["rsi"] = _rsi(df["close"], rsi_len)

    # +DI/-DI/ADX
    di = _di_adx(df, adx_period)
    for col in ["+di", "-di", "adx"]:
        df[col] = di[col]

    # ATR (needed for backtest sizing)
    atr_period = int(cfg.get("atr_period", cfg.get("atr_length", 14)))
    tr = pd.concat([
        (df["high"] - df["low"]),
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"] - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0/atr_period, adjust=False, min_periods=atr_period).mean().fillna(0.0)

    # Relative Volume (RVOL)
    vol_win = int(cfg.get("vol_sma_length", 20))
    vol_avg = df["volume"].rolling(window=vol_win, min_periods=5).mean()
    df["rvol"] = df["volume"] / vol_avg.replace(0, 1)

    # Fresh cross flags (current bar vs previous bar)
    prev_gt = (df["ema_fast"].shift(1) > df["ema_slow"].shift(1))
    now_gt  = (df["ema_fast"] > df["ema_slow"])
    df["fresh_cross_up"] = (~prev_gt) & (now_gt)

    prev_lt = (df["ema_fast"].shift(1) < df["ema_slow"].shift(1))
    now_lt  = (df["ema_fast"] < df["ema_slow"])
    df["fresh_cross_down"] = (~prev_lt) & (now_lt)

    # Optional: keep handy rolling dollar volume if already present upstream
    # (no change to your current pipeline)

    return df


def long_ok(last: pd.Series, cfg: Dict[str, Any]) -> bool:
    ok, _ = explain_long_gate(last, cfg)
    return ok


def explain_long_gate(last: pd.Series, cfg: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Evaluate long gate on a *single* row produced by attach_verifiers(...).
    Returns (ok, reasons[]) where reasons are block explanations when ok=False.
    """
    fcfg = dict(cfg.get("filters", {}))
    reasons: list[str] = []

    # Pull guardrails
    adx_thr = float(fcfg.get("adx_threshold", 23.0))
    rsi_min = float(fcfg.get("rsi_min", 50.0))
    rsi_max = float(fcfg.get("rsi_max", 80.0))
    slope_thr = float(fcfg.get("slope_threshold_pct", 0.0010))
    require_fast_above_slow = bool(fcfg.get("require_fast_above_slow", True))
    require_cross = bool(fcfg.get("require_cross", False))  # stricter entry
    require_di_trend = bool(fcfg.get("require_di_trend", False))  # +DI > -DI
    min_rvol = float(fcfg.get("min_rvol", 0.0))  # New RVOL filter
    min_price = fcfg.get("min_price", None)
    min_dv = fcfg.get("min_dollar_vol", None)
    
    # Time Filter (Lunch Block)
    # Assumes 'last' has a name (timestamp) or we pass it. 
    # For now, we rely on the caller (live loop) to handle session times, 
    # but we can add a specific "Lunch Block" here if the index is a datetime.
    block_lunch = bool(fcfg.get("block_lunch", False))
    is_lunch = False
    if block_lunch and isinstance(last.name, pd.Timestamp):
        # Convert to ET if naive or UTC (assuming standard market hours)
        # Simplified: Just check hour if we know the timezone. 
        # Better: Use the 'hour' from the timestamp directly if it's already localized or UTC.
        # Market Open 9:30 ET. Lunch 12:00-13:00 ET.
        # If UTC: 12:00 ET is 16:00/17:00 UTC depending on DST.
        # Let's use a robust check if possible, or just rely on the user to configure 'entry_windows'.
        pass 

    # Read fields (tolerant)
    ema_fast = float(last.get("ema_fast", np.nan))
    ema_slow = float(last.get("ema_slow", np.nan))
    adx = float(last.get("adx", 0.0))
    rsi = float(last.get("rsi", 50.0))
    slope = float(last.get("ema_slope_pct", 0.0))
    plus_di = float(last.get("+di", 0.0))
    minus_di = float(last.get("-di", 0.0))
    close = float(last.get("close", np.nan))

    # Price floor
    if min_price is not None and math.isfinite(close) and close < float(min_price):
        reasons.append(f"price {close:.2f} < {float(min_price):.2f}")

    # EMA alignment
    if require_fast_above_slow and not (ema_fast > ema_slow):
        reasons.append("ema_fast ≤ ema_slow")

    # Fresh cross filter (uses column created in attach_verifiers)
    if require_cross and not bool(last.get("fresh_cross_up", False)):
        reasons.append("no fresh cross (fast>slow)")

    # ADX strength
    if adx < adx_thr:
        reasons.append(f"ADX {adx:.1f} < {adx_thr:.1f}")

    # DI direction
    if require_di_trend and not (plus_di > minus_di):
        reasons.append("+DI ≤ -DI (trend mismatch)")

    # RSI window
    if rsi < rsi_min:
        reasons.append(f"RSI {rsi:.1f} < {rsi_min:.1f}")
    if rsi > rsi_max:
        reasons.append(f"RSI {rsi:.1f} > {rsi_max:.1f}")

    # Slope
    if slope < slope_thr:
        reasons.append(f"EMA_slope {slope:.5f} < {slope_thr:.5f}")

    # RVOL
    rvol = float(last.get("rvol", 0.0))
    if min_rvol > 0 and rvol < min_rvol:
        reasons.append(f"RVOL {rvol:.2f} < {min_rvol:.2f}")

    # Lunch Block (Hardcoded for 12:00-13:00 ET approx for now, or use config windows)
    # Actually, let's rely on 'entry_windows' in config.yaml for time filtering 
    # as it's cleaner than hardcoding timezones here.

    # Optional dollar volume gate if present
    if (min_dv is not None) and ("dollar_vol_avg" in last.index):
        dv = float(last.get("dollar_vol_avg", 0.0))
        if dv < float(min_dv):
            reasons.append(f"dollar_vol_avg {dv:,.0f} < {float(min_dv):,.0f}")

    ok = (len(reasons) == 0)
    return ok, reasons
