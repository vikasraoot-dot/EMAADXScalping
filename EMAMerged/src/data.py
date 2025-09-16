from __future__ import annotations
import datetime as dt
from typing import Optional, List
import pandas as pd
import pytz
import requests

def _headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

def _now_utc() -> dt.datetime:
    return dt.datetime.now(tz=dt.timezone.utc)

def alpaca_market_open(base_url: str, key: str, secret: str) -> bool:
    url = f"{base_url.rstrip('/')}/v2/clock"
    r = requests.get(url, headers=_headers(key, secret), timeout=10)
    r.raise_for_status()
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
    history_days: int = 30,
    bar_limit: int = 500,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Pulls recent bars from Alpaca Market Data v2.
    """
    end = _now_utc()
    start = end - dt.timedelta(days=history_days)
    url = f"https://data.alpaca.markets/v2/stocks/bars"
    params = {
        "symbols": symbol,
        "timeframe": timeframe,
        "start": _iso(start),
        "end": _iso(end),
        "limit": bar_limit,
        "feed": feed,
        "adjustment": "raw",
    }
    r = requests.get(url, headers=_headers(key, secret), params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    bars = j.get("bars", {})
    arr = bars.get(symbol, [])
    if not arr:
        return pd.DataFrame()
    df = pd.DataFrame(arr)
    # Alpaca returns: t (ISO), o,h,l,c,v, n (trade ct), vw (avg)
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
    # Keep only RTH range
    t0 = pd.to_datetime(local.index.date.astype(str) + " " + rth_start).tz_localize(tz_name)
    t1 = pd.to_datetime(local.index.date.astype(str) + " " + rth_end).tz_localize(tz_name)
    mask = (local.index >= t0) & (local.index <= t1)
    local = local[mask]
    # Optional narrow entry windows
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
    # conservative: disable last bar always; live loop runs on schedule shortly after close anyway
    return df.iloc[:-1] if len(df) > 1 else df
