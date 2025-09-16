from __future__ import annotations
import os, yaml, datetime as dt, pytz
import pandas as pd

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def read_tickers(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def new_york_now() -> dt.datetime:
    return dt.datetime.now(tz=pytz.timezone("US/Eastern"))

def round2(x) -> float:
    try:
        return float(f"{float(x):.2f}")
    except Exception:
        return 0.0
