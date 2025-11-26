# EMAMerged/scripts/flat_eod.py
from __future__ import annotations
import os, sys
from EMAMerged.src.data import cancel_all_orders, close_all_positions

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default

def _creds():
    # Try to resolve using the same logic as data.py, or fall back to env vars
    # We want to support APCA_API_KEY_ID as well.
    key = _env("APCA_API_KEY_ID") or _env("ALPACA_API_KEY") or _env("ALPACA_KEY") or _env("ALPACA_KEY_ID")
    sec = _env("APCA_API_SECRET_KEY") or _env("ALPACA_API_SECRET") or _env("ALPACA_SECRET") or _env("ALPACA_SECRET_KEY")
    
    # Base URL
    base = _env("APCA_API_BASE_URL") or _env("APCA_BASE_URL") or _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets"
    base = base.rstrip("/")
    
    if not key or not sec:
        # If we can't find them here, we'll return None and let data.py try to find them (or fail there)
        # But data.py functions take optional args.
        pass
        
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
