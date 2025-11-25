from __future__ import annotations
import pandas as pd
from typing import Dict
from .filters import attach_verifiers

def compute_indicators(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """
    Compute all indicators. Delegates to filters.attach_verifiers.
    """
    return attach_verifiers(df, cfg)

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
