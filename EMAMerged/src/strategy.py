from __future__ import annotations
import pandas as pd
from typing import Dict
from .indicators import ema, atr as _atr, rsi as _rsi
from .filters import attach_verifiers

def compute_indicators(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    out = df.copy()
    f = int(cfg.get("ema_fast", 9))
    s = int(cfg.get("ema_slow", 21))
    ap = int(cfg.get("atr_period", cfg.get("atr_length", 14)))

    out["ema_fast"] = ema(out["close"], f)
    out["ema_slow"] = ema(out["close"], s)
    out["atr"] = _atr(out, period=ap)

    # If you ever want RSI directly here, you can add it; attach_verifiers provides it now.
    return attach_verifiers(out, cfg, ema_fast_col="ema_fast", ema_slow_col="ema_slow")

def crossover(df: pd.DataFrame) -> int:
    if len(df) < 2:
        return 0
    prev = df.iloc[-2]
    last = df.iloc[-1]
    diff_prev = float(prev["ema_fast"]) - float(prev["ema_slow"])
    diff_now  = float(last["ema_fast"]) - float(last["ema_slow"])
    if diff_prev <= 0 and diff_now > 0:
        return 1
    if diff_prev >= 0 and diff_now < 0:
        return -1
    return 0
