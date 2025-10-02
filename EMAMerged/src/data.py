from __future__ import annotations
import datetime as dt
from typing import Optional, List, Dict, Any
import pandas as pd
import pytz
import requests
import time
import os


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
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            return r
        except Exception:
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
    timeframe: str,
    symbol: str,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    *,
    # Backward-compatible convenience args (used by your backtester)
    days: int | None = None,
    history_days: int | None = None,
    feed: str = "iex",
    limit: int = 500,
    adjustment: str = "raw",
    **_ignore: Any,  # safely swallow any unknown kwargs from older callers
) -> pd.DataFrame:
    """
    Fetch bars from Alpaca v2. Accepts either explicit start/end OR a days/history_days lookback.
    Backtester passes history_days=30; live code usually passes explicit start/end.
    """
    # Resolve lookback window
    if start is None or end is None:
        lookback = days if days is not None else history_days
        if lookback is None:
            lookback = 30  # sensible default
        end = _now_utc()
        start = end - dt.timedelta(days=int(lookback))

    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": timeframe,
        "start": _iso(start),
        "end": _iso(end),
        "adjustment": adjustment,
        "feed": feed,
        "limit": int(limit),
    }
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=20, params=params)
    js = r.json()
    bars = js.get("bars", [])
    if not bars:
        return pd.DataFrame()
    df = pd.DataFrame(bars)
    # Normalize schema
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"t":"time","o":"open","h":"high","l":"low","c":"close","v":"volume"})
    df = df.set_index("time").sort_index()
    return df

def filter_rth(df: pd.DataFrame, tz_name: str = "US/Eastern", rth_start: str = "09:30", rth_end: str = "16:00",
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

def _tf_minutes(timeframe: str) -> int:
    digits = "".join(ch for ch in timeframe if ch.isdigit())
    return int(digits or "1")

def drop_unclosed_last_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty or len(df) == 1:
        return df
    last = df.index[-1].tz_convert("UTC")
    now  = _now_utc()
    mins = _tf_minutes(timeframe)
    # only drop if bar is truly still in-progress for this timeframe
    return df.iloc[:-1] if (now - last) < pd.Timedelta(minutes=mins) else df

# --- Orders / Positions / Open orders ---
def get_positions(base_url: str, key: str, secret: str) -> Dict[str, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v2/positions"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=15)
    if r.status_code == 404:
        return {}
    arr = r.json() if r.text else []
    out: Dict[str, Dict[str, Any]] = {}
    for p in arr:
        out[p["symbol"]] = p
    return out

def get_open_orders(base_url: str, key: str, secret: str, symbol: str | None = None) -> list[dict]:
    url = f"{base_url.rstrip('/')}/v2/orders?status=open&nested=true"
    if symbol:
        url += f"&symbols={symbol}"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=15)
    return r.json() if r.text else []

def submit_market_order(base_url: str, key: str, secret: str,
                        symbol: str, qty: int, side: str, client_order_id: str | None = None) -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    payload = {
        "symbol": symbol, "qty": qty, "side": side, "type": "market",
        "time_in_force": "day"
    }
    if client_order_id:
        payload["client_order_id"] = client_order_id
    r = _req_with_retry("POST", url, headers=_headers(key, secret), json=payload, timeout=20)
    return r.json()

def submit_bracket_order(base_url: str, key: str, secret: str,
                         symbol: str, qty: int, side: str,
                         entry_type: str = "market", client_order_id: str | None = None,
                         take_profit_price: float = 0.0, stop_price: float = 0.0,
                         stop_limit_price: float | None = None) -> dict:
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
                       {"stop_price": round(stop_limit_price if stop_limit_price < stop_price else stop_price, 2),
                        "limit_price": round(stop_limit_price, 2)},
    }
    r = _req_with_retry("POST", url, headers=_headers(key, secret), json=payload, timeout=20)
    return r.json()

def cancel_all_orders(base_url: str, key: str, secret: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    r = _req_with_retry("DELETE", url, headers=_headers(key, secret), timeout=20)
    return r.json() if r.text else {}

def close_all_positions(base_url: str, key: str, secret: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/positions"
    r = _req_with_retry("DELETE", url, headers=_headers(key, secret), timeout=30)
    return r.json() if r.text else {}

def list_open_orders(base_url: str, key: str, secret: str, symbols: list[str] | None = None) -> list[dict]:
    url = f"{base_url.rstrip('/')}/v2/orders?status=open&nested=true"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=20)
    orders = r.json() if r.text else []
    if symbols:
        syms = set(s.upper() for s in symbols)
        orders = [o for o in orders if o.get("symbol","").upper() in syms]
    return orders

def patch_order(base_url: str, key: str, secret: str, order_id: str, **fields) -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders/{order_id}"
    r = _req_with_retry("PATCH", url, headers=_headers(key, secret), timeout=20, json=fields)
    return r.json() if r.text else {}
def fetch_latest_bars(
    symbols: list[str],
    *,
    timeframe: str = "15Min",
    history_days: int = 30,
    feed: str = "iex",
    # RTH controls (align with your config defaults)
    rth_only: bool = True,
    tz_name: str = "US/Eastern",
    rth_start: str = "09:30",
    rth_end: str = "15:55",
    allowed_windows: Optional[List[dict]] = None,
    # Limits/creds
    bar_limit: int = 10000,
    key: Optional[str] = None,
    secret: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience wrapper used by live_paper_loop:
      - fetches recent bars for each symbol
      - drops the still-forming last bar (to avoid lookahead)
      - optionally filters to regular trading hours and allowed entry windows
    """
    # Resolve API creds from args or environment
    key = key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY", "")
    secret = secret or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET", "")

    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = get_alpaca_bars(
            key=key,
            secret=secret,
            timeframe=timeframe,
            symbol=sym,
            history_days=history_days,
            feed=feed,
            limit=int(bar_limit),
        )
        if df.empty:
            out[sym] = df
            continue

        # Drop any unclosed/partial last bar for the given timeframe
        df = drop_unclosed_last_bar(df, timeframe)

        # Filter to RTH + optional entry windows
        if rth_only:
            df = filter_rth(df, tz_name=tz_name, rth_start=rth_start, rth_end=rth_end, allowed_windows=allowed_windows)

        out[sym] = df

    return out
