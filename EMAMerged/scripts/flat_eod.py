# EMAMerged/scripts/flat_eod.py
from __future__ import annotations
import os, sys
from EMAMerged.src.data import cancel_all_orders, close_all_positions

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default

def _creds():
    key = _env("ALPACA_API_KEY") or _env("ALPACA_KEY")
    sec = _env("ALPACA_API_SECRET") or _env("ALPACA_SECRET")
    base = _env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    if not key or not sec:
        raise RuntimeError("Missing ALPACA creds")
    return base, key, sec

def main():
    base, key, sec = _creds()
    try:
        cancel_all_orders(base, key, sec)
        print("[EOD] canceled all open orders", flush=True)
    except Exception as e:
        print(f"[EOD] cancel orders error: {e}", flush=True)
    try:
        close_all_positions(base, key, sec)
        print("[EOD] submitted flatten all positions", flush=True)
    except Exception as e:
        print(f"[EOD] close positions error: {e}", flush=True)

if __name__ == "__main__":
    main()
