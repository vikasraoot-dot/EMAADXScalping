from __future__ import annotations
import datetime as dt
from typing import Optional, List, Dict, Any
import pandas as pd
import pytz
import requests
import time

# ------------------------
# Resilient HTTP wrapper
# ------------------------
RETRY_STATUS = {429, 500, 502, 503, 504}

def _req_with_retry(method: str, url: str, headers: dict, timeout: int = 20,
                    max_retries: int = 5, backoff_base: float = 0.7, **kwargs) -> requests.Response:
    attempt = 0
    while True:
        try:
            r = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
            if r.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"Retryable status {r.status_code}", response=r)
            r.raise_for_status()
            return r
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise
            sleep_s = backoff_base * (2 ** (attempt - 1)) + (0.1 * attempt)
            time.sleep(sleep_s)

# ------------------------
# Alpaca helpers
# ------------------------
def _headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

def _now_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)

def alpaca_market_open(base_url: str, key: str, secret: str) -> bool:
    url = f"{base_url.rstrip('/')}/v2/clock"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=10)
    j = r.json()
    return bool(j.get("is_open", False))

def _iso(dt_obj: dt.datetime) -> str:
    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=dt.timezone.utc)
    return dt_obj.isoformat()

def get_alpaca_bars(
    key: str,
    secret: str,
    symbol: str,
    timeframe: str = "5Min",
    history_days: int = 90,
    bar_limit: int = 500,
    feed: str = "iex",
) -> pd.DataFrame:
    end = _now_utc()
    start = end - dt.timedelta(days=history_days)
    url = "https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": _iso(start),
        "end": _iso(end),
        "limit": bar_limit,
        "feed": feed,
        "adjustment": "raw",
    }
    r = _req_with_retry("GET", url, headers=_headers(key, secret), params=params, timeout=30)
    j = r.json()
    arr = j.get("bars", {}).get(symbol, [])
    if not arr:
        return pd.DataFrame()
    df = pd.DataFrame(arr)
    df = df.rename(columns={"t":"timestamp","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["open","high","low","close","volume"]]

def filter_rth(df: pd.DataFrame, tz_name: str, rth_start: str, rth_end: str,
               allowed_windows: Optional[List[dict]] = None) -> pd.DataFrame:
    if df.empty:
        return df
    idx = df.index.tz_convert(tz_name) if df.index.tz is not None else df.index.tz_localize("UTC").tz_convert(tz_name)
    local = df.copy()
    local.index = idx
    t0 = pd.to_datetime(local.index.date.astype(str) + " " + rth_start).tz_localize(tz_name)
    t1 = pd.to_datetime(local.index.date.astype(str) + " " + rth_end).tz_localize(tz_name)
    mask = (local.index >= t0) & (local.index <= t1)
    local = local[mask]
    if allowed_windows:
        m2 = False
        for w in allowed_windows:
            ws = pd.to_datetime(local.index.date.astype(str) + " " + w["start"]).tz_localize(tz_name)
            we = pd.to_datetime(local.index.date.astype(str) + " " + w["end"]).tz_localize(tz_name)
            m2 = m2 | ((local.index >= ws) & (local.index <= we))
        local = local[m2]
    local.index = local.index.tz_convert("UTC")
    return local

def drop_unclosed_last_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    return df.iloc[:-1] if len(df) > 1 else df

# --- Orders / Positions / Open orders ---
def get_positions(base_url: str, key: str, secret: str) -> Dict[str, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v2/positions"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=15)
    if r.status_code == 404:
        return {}
    out = {}
    for p in r.json():
        out[p["symbol"]] = p
    return out

def get_open_orders(base_url: str, key: str, secret: str, symbol: str | None = None) -> list[dict]:
    url = f"{base_url.rstrip('/')}/v2/orders"
    params = {"status": "open"}
    if symbol:
        params["symbols"] = symbol
    r = _req_with_retry("GET", url, headers=_headers(key, secret), params=params, timeout=15)
    return r.json()

def submit_market_order(base_url: str, key: str, secret: str,
                        symbol: str, qty: int, side: str, client_order_id: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "client_order_id": client_order_id,
    }
    r = _req_with_retry("POST", url, headers=_headers(key, secret), json=payload, timeout=20)
    return r.json()

def submit_bracket_order(base_url: str, key: str, secret: str,
                         symbol: str, qty: int, side: str,
                         entry_type: str, client_order_id: str,
                         take_profit_price: float, stop_price: float, stop_limit_price: float | None = None) -> dict:
    """
    Alpaca bracket order. entry_type: "market" or "limit" (we use "market").
    """
    url = f"{base_url.rstrip('/')}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": entry_type,
        "time_in_force": "day",
        "order_class": "bracket",
        "client_order_id": client_order_id,
        "take_profit": {"limit_price": round(take_profit_price, 2)},
        "stop_loss":   {"stop_price": round(stop_price, 2)} if not stop_limit_price else
                       {"stop_price": round(stop_price, 2), "limit_price": round(stop_limit_price, 2)},
    }
    r = _req_with_retry("POST", url, headers=_headers(key, secret), json=payload, timeout=20)
    return r.json()
