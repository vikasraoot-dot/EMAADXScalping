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
                raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
            return r
        except Exception:
            attempt += 1
            if attempt > max_retries:
                raise
            time.sleep(backoff_base * (2 ** (attempt - 1)))

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

# ------------------------
# Bars / Market data
# ------------------------
def get_alpaca_bars(
    key: str,
    secret: str,
    timeframe: str,
    symbol: str,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    *,
    days: int | None = None,
    history_days: int | None = None,
    feed: str = "iex",
    limit: int = 500,
    adjustment: str = "raw",
    **_ignore: Any,
) -> pd.DataFrame:
    if end is None:
        end = _now_utc()
    if start is None:
        hd = history_days if history_days is not None else (days if days is not None else 5)
        start = end - pd.Timedelta(days=int(hd))

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
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
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
    return df.iloc[:-1] if (now - last) < pd.Timedelta(minutes=mins) else df

# ------------------------
# Orders / Positions helpers (unchanged)
# ------------------------
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

def get_open_orders(base_url: str, key: str, secret: str) -> List[dict]:
    url = f"{base_url.rstrip('/')}/v2/orders?status=open&nested=true"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=15)
    return r.json() if r.text else []

def submit_market_order(base_url: str, key: str, secret: str, symbol: str, qty: int, side: str, tif: str = "day") -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    order = {"symbol": symbol, "qty": int(qty), "side": side, "type": "market", "time_in_force": tif}
    r = _req_with_retry("POST", url, headers=_headers(key, secret), timeout=20, json=order)
    return r.json() if r.text else {}

def submit_bracket_order(base_url: str, key: str, secret: str, symbol: str, qty: int, side: str,
                         limit_price: float | None, take_profit_price: float, stop_loss_price: float,
                         tif: str = "day") -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    payload = {
        "symbol": symbol,
        "qty": int(qty),
        "side": side,
        "type": "limit" if limit_price is not None else "market",
        "time_in_force": tif,
        "limit_price": round(float(limit_price), 2) if limit_price is not None else None,
        "order_class": "bracket",
        "take_profit": {"limit_price": round(float(take_profit_price), 2)},
        "stop_loss": {"stop_price": round(float(stop_loss_price), 2)},
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    r = _req_with_retry("POST", url, headers=_headers(key, secret), timeout=20, json=payload)
    return r.json() if r.text else {}

def cancel_all_orders(base_url: str, key: str, secret: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    r = _req_with_retry("DELETE", url, headers=_headers(key, secret), timeout=20)
    return r.json() if r.text else {}

def close_all_positions(base_url: str, key: str, secret: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/positions"
    r = _req_with_retry("DELETE", url, headers=_headers(key, secret), timeout=20)
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

# ------------------------
# Option B: compute dollar_vol_avg BEFORE window trim
# ------------------------
def fetch_latest_bars(
    symbols: list[str],
    *,
    timeframe: str = "15Min",
    history_days: int = 30,
    feed: str = "iex",
    # RTH controls
    rth_only: bool = True,
    tz_name: str = "US/Eastern",
    rth_start: str = "09:30",
    rth_end: str = "15:55",
    allowed_windows: Optional[List[dict]] = None,
    # Limits/creds
    bar_limit: int = 10000,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    # Rolling config (wired from config via caller)
    dollar_vol_window: int = 20,
    dollar_vol_min_periods: int = 7,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict[symbol]->DataFrame. Computes `dollar_vol_avg` on the full RTH
    slice BEFORE any allowed_windows trimming for stability from the first bar.
    """
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

        # Avoid lookahead
        df = drop_unclosed_last_bar(df, timeframe)

        # Build RTH slice WITHOUT windows
        rth_df = filter_rth(df, tz_name=tz_name, rth_start=rth_start, rth_end=rth_end, allowed_windows=None) if rth_only else df

        # Rolling dollar volume on full RTH slice
        if not rth_df.empty and {"close", "volume"}.issubset(rth_df.columns):
            dv = (rth_df["close"].astype(float) * rth_df["volume"].astype(float)).rolling(
                int(dollar_vol_window),
                min_periods=int(dollar_vol_min_periods),
            ).mean()
            rth_df["dollar_vol_avg"] = dv.fillna(method="ffill").fillna(0.0)
        else:
            rth_df["dollar_vol_avg"] = 0.0

        # Now optionally trim to allowed_windows (preserves computed columns)
        final_df = (
            filter_rth(rth_df, tz_name=tz_name, rth_start=rth_start, rth_end=rth_end, allowed_windows=allowed_windows)
            if (rth_only and allowed_windows) else rth_df
        )
        out[sym] = final_df

    return out
