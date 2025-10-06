# === EMAMerged/src/data.py ===
# (full file — unchanged except for the new close_position at the noted spot)
import os, json, time, math, datetime as dt, random
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd

# ... (all your existing imports, helpers, and functions remain unchanged)

def cancel_all_orders(base_url: str, key: str, secret: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    params = {"status": "open"}
    r = _req_with_retry("GET", url, headers=_headers(key, secret), params=params, timeout=20)
    try:
        arr = r.json() if r.text else []
    except Exception:
        arr = []
    # Cancel each open order
    deleted = []
    for o in arr:
        oid = o.get("id")
        if not oid:
            continue
        try:
            r2 = _req_with_retry("DELETE", f"{base_url.rstrip('/')}/v2/orders/{oid}", headers=_headers(key, secret), timeout=20)
            deleted.append({"id": oid, "status_code": r2.status_code})
        except Exception as e:
            deleted.append({"id": oid, "error": str(e)})
    return {"deleted": deleted, "count": len(deleted)}

def close_all_positions(base_url: str, key: str, secret: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/positions"
    r = _req_with_retry("DELETE", url, headers=_headers(key, secret), timeout=30)
    try:
        return r.json() if r.text else {}
    except Exception:
        return {"status_code": r.status_code, "text": r.text}

def close_position(base_url: str, key: str, secret: str, symbol: str) -> dict:
    """Close (liquidate) a single open position by symbol.
    Mirrors Alpaca DELETE /v2/positions/{symbol}.
    Returns JSON response if any, else an empty dict.
    """
    url = f"{base_url.rstrip('/')}/v2/positions/{symbol}"
    r = _req_with_retry("DELETE", url, headers=_headers(key, secret), timeout=20)
    try:
        return r.json() if r.text else {}
    except Exception:
        return {"status_code": r.status_code, "text": r.text}

# ... (rest of file unchanged: fetch_latest_bars, submit_bracket_order, get_positions, etc.)
