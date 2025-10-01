# EMAMerged/src/config_compat.py
from __future__ import annotations
from typing import Dict, Any, List
import re

_MINF = float("-inf")

def _map_interval_to_alpaca_tf(s: str) -> str:
    # Map "15m" -> "15Min", "5m" -> "5Min", "1h" -> "1Hour", etc.
    s = s.strip().lower()
    if s.endswith("m"):
        return f"{int(s[:-1])}Min"
    if s.endswith("h"):
        return f"{int(s[:-1])}Hour"
    if s.endswith("d"):
        return "1Day"
    return s  # fallback

def _parse_days(s: str) -> int:
    """Accepts '10d', '60d', etc. Returns int days (default 30)."""
    if not s:
        return 30
    m = re.match(r"^\s*(\d+)\s*d\s*$", str(s), re.I)
    return int(m.group(1)) if m else 30

def normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mutates/returns cfg so that reference-style keys are honored:
      - intervals / periods -> timeframe, history_days
      - atr_length -> atr_period
      - rsi_length -> filters.rsi_period
      - rsi_long_min/max -> filters.rsi_min/max
      - slope_threshold_pct (top-level) -> filters.slope_threshold_pct
      - risk/targets/stops keys -> brackets.* when applicable
      - entry_windows kept as-is (live loop enforces if present)
      - max_open_trades_per_ticker honored by live loop
      - max_shares_per_trade honored by live loop
    """
    cfg = dict(cfg or {})

    # --- timeframe from intervals ---
    intervals = cfg.get("intervals")
    if isinstance(intervals, list) and intervals:
        cfg["timeframe"] = _map_interval_to_alpaca_tf(str(intervals[0]))
        # periods mapping "15m": "10d" -> history_days
        periods = cfg.get("periods", {})
        first_iv = str(intervals[0]).lower()
        if isinstance(periods, dict) and periods.get(first_iv):
            cfg["history_days"] = _parse_days(periods[first_iv])
    else:
        # backward-compat: keep existing cfg["timeframe"] / history_days
        pass

    # --- ATR length / EMA, already supported; add fallback for ATR length naming ---
    if "atr_period" not in cfg and "atr_length" in cfg:
        cfg["atr_period"] = int(cfg.get("atr_length", 14))

    # --- Filters ---
    fcfg = dict(cfg.get("filters", {}) or {})
    # RSI length
    if "rsi_period" not in fcfg and "rsi_length" in cfg:
        fcfg["rsi_period"] = int(cfg.get("rsi_length", 14))
    # RSI longs (from reference)
    if "rsi_long_min" in cfg:
        fcfg["rsi_min"] = float(cfg["rsi_long_min"])
    if "rsi_long_max" in cfg:
        fcfg["rsi_max"] = float(cfg["rsi_long_max"])
    # EMA slope threshold (reference may specify at top-level)
    if "slope_threshold_pct" in cfg and "slope_threshold_pct" not in fcfg:
        fcfg["slope_threshold_pct"] = float(cfg["slope_threshold_pct"])
    cfg["filters"] = fcfg

    # --- Stops & Targets mapping to our brackets ---
    bcfg = dict(cfg.get("brackets", {}) or {})
    stop_type = cfg.get("stop_type")
    if stop_type == "atr":
        # map reference atr_mult to our atr_mult_sl
        if "atr_mult" in cfg:
            bcfg["atr_mult_sl"] = float(cfg["atr_mult"])
    elif stop_type == "percent":
        # percent_stop like 0.0075 => 0.75%
        if "percent_stop" in cfg:
            bcfg["sl_pct"] = float(cfg["percent_stop"]) * 100.0
    # swing stop unsupported here; leave to your existing code if any

    # target: use_rr_target and rr_target to our take_profit_r (R multiples)
    if cfg.get("use_rr_target") and "rr_target" in cfg:
        bcfg["take_profit_r"] = float(cfg["rr_target"])
    # fixed_target_pct could map to tp_pct (%)
    if "fixed_target_pct" in cfg and not cfg.get("use_rr_target", False):
        bcfg["tp_pct"] = float(cfg["fixed_target_pct"]) * 100.0
    cfg["brackets"] = bcfg

    # --- History fallback if only default_period present ---
    if "history_days" not in cfg and "default_period" in cfg:
        cfg["history_days"] = _parse_days(cfg.get("default_period"))

    # Leave entry_windows, max_open_trades_per_ticker, max_shares_per_trade as-is;
    # live loop will honor them if present.
    return cfg
