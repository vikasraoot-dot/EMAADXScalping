# === EMAMerged/src/data.py ===
from __future__ import annotations

import os, json, time, math, datetime as dt
from typing import Dict, Any, List, Optional, Tuple
import requests
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────
def _headers(key: str, secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": key or os.getenv("APCA_API_KEY_ID", ""),
        "APCA-API-SECRET-KEY": secret or os.getenv("APCA_API_SECRET_KEY", ""),
        "Content-Type": "application/json",
    }

def _req_with_retry(method: str, url: str, retries: int = 2, backoff: float = 0.6, **kw) -> requests.Response:
    last = None
    for i in range(retries + 1):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code} {r.text}")
            return r
        except Exception as e:
            last = e
            if i == retries:
                raise
            time.sleep(backoff * (2 ** i))
    # Shouldn’t get here
    if isinstance(last, requests.Response):
        return last
    raise RuntimeError(str(last) if last else "unknown error")

# ──────────────────────────────────────────────────────────────────────────────
# Market status / account data
# ──────────────────────────────────────────────────────────────────────────────
def alpaca_market_open(base_url: str, key: str, secret: str) -> bool:
    url = f"{base_url.rstrip('/')}/v2/clock"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=15)
    try:
        return bool(r.json().get("is_open"))
    except Exception:
        return False

def get_positions(base_url: str, key: str, secret: str) -> Dict[str, Dict[str, Any]]:
    """Return positions keyed by symbol."""
    url = f"{base_url.rstrip('/')}/v2/positions"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=20)
    out: Dict[str, Dict[str, Any]] = {}
    try:
        arr = r.json() if r.text else []
        for p in arr or []:
            sym = p.get("symbol")
            if sym:
                out[sym] = p
    except Exception:
        pass
    return out

def list_open_orders(base_url: str, key: str, secret: str, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v2/orders"
    params = {"status": "open", "nested": "true"}
    r = _req_with_retry("GET", url, headers=_headers(key, secret), params=params, timeout=20)
    try:
        arr = r.json() if r.text else []
        if symbols:
            sy = set(s.upper() for s in symbols)
            arr = [o for o in arr if (o.get("symbol") or "").upper() in sy]
        return arr
    except Exception:
        return []

# Back-compat alias used in some earlier code/tests
def get_open_orders(base_url: str, key: str, secret: str, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    return list_open_orders(base_url, key, secret, symbols)

def patch_order(base_url: str, key: str, secret: str, order_id: str, **fields) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v2/orders/{order_id}"
    r = _req_with_retry("PATCH", url, headers=_headers(key, secret), json=fields, timeout=20)
    try:
        return r.json() if r.text else {}
    except Exception:
        return {"status_code": r.status_code, "text": r.text}

# ──────────────────────────────────────────────────────────────────────────────
# Risk ops
# ──────────────────────────────────────────────────────────────────────────────
def cancel_all_orders(base_url: str, key: str, secret: str) -> dict:
    url = f"{base_url.rstrip('/')}/v2/orders"
    params = {"status": "open"}
    r = _req_with_retry("GET", url, headers=_headers(key, secret), params=params, timeout=20)
    try:
        arr = r.json() if r.text else []
    except Exception:
        arr = []
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
    """Close (liquidate) a single open position by symbol."""
    url = f"{base_url.rstrip('/')}/v2/positions/{symbol}"
    r = _req_with_retry("DELETE", url, headers=_headers(key, secret), timeout=20)
    try:
        return r.json() if r.text else {}
    except Exception:
        return {"status_code": r.status_code, "text": r.text}

# ──────────────────────────────────────────────────────────────────────────────
# Bar fetch (Option B, with dollar_vol rolling & RTH filtering)
# ──────────────────────────────────────────────────────────────────────────────
def _et_window_mask(idx_utc: pd.DatetimeIndex,
                    tz_name: str,
                    rth_start: str,
                    rth_end: str,
                    allowed_windows: Optional[List[Dict[str, str]]] = None) -> pd.Series:
    """Build mask for regular session and (optionally) sub-windows."""
    if idx_utc.tz is None:
        idx_utc = idx_utc.tz_localize("UTC")
    idx_et = idx_utc.tz_convert(tz_name)
    hm = idx_et.strftime("%H:%M")

    # Base RTH mask
    s_h, s_m = map(int, rth_start.split(":"))
    e_h, e_m = map(int, rth_end.split(":"))
    base = (hm >= f"{s_h:02d}:{s_m:02d}") & (hm <= f"{e_h:02d}:{e_m:02d}")

    if not allowed_windows:
        return base

    # Apply “allowed” sub-windows inside RTH
    allow = pd.Series(False, index=idx_utc)
    for w in allowed_windows:
        ws = w.get("start", rth_start); we = w.get("end", rth_end)
        allow |= (hm >= ws) & (hm <= we)
    return base & allow

def fetch_latest_bars(
    symbols: List[str],
    timeframe: str = "15Min",
    history_days: int = 10,
    feed: str = "iex",
    rth_only: bool = True,
    tz_name: str = "US/Eastern",
    rth_start: str = "09:30",
    rth_end: str = "15:55",
    allowed_windows: Optional[List[Dict[str,str]]] = None,
    bar_limit: int = 10000,
    key: str = "",
    secret: str = "",
    dollar_vol_window: int = 20,
    dollar_vol_min_periods: int = 7,
) -> Dict[str, pd.DataFrame]:
    """Return {symbol: DataFrame(OHLCV + dollar_vol_avg)}."""
    if not symbols:
        return {}

    base_data = "https://data.alpaca.markets"
    end = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    start = (dt.datetime.now(dt.UTC) - dt.timedelta(days=history_days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    params = {
        "symbols": ",".join([s.upper() for s in symbols]),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "limit": bar_limit,
        "feed": feed,
    }
    url = f"{base_data}/v2/stocks/bars"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), params=params, timeout=30)
    try:
        root = r.json()
        raw = root.get("bars", {})
    except Exception:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    for sym, rows in (raw or {}).items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # Normalize timestamp index
        ts_col = "t" if "t" in df.columns else "timestamp"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.set_index(ts_col).sort_index()
        # Normalize columns to expected names
        rename_map = {
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        }
        df = df.rename(columns=rename_map)
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].copy()

        # RTH / windows
        if rth_only:
            mask = _et_window_mask(df.index, tz_name, rth_start, rth_end, allowed_windows)
            df = df.loc[mask].copy()
            if df.empty:
                out[sym] = df
                continue

        # Dollar volume rolling mean (deprecation-safe)
        dv = (pd.to_numeric(df["close"], errors="coerce").fillna(0.0) *
              pd.to_numeric(df["volume"], errors="coerce").fillna(0.0))
        rth_df = df.copy()
        rth_df["dollar_vol_avg"] = (
            dv.rolling(window=int(max(1, dollar_vol_window)),
                       min_periods=int(max(1, dollar_vol_min_periods)))
              .mean()
              .ffill()        # <- replaces fillna(method="ffill")
              .fillna(0.0)
        )

        out[sym] = rth_df

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Bracket entry (with exchange tick guard)
# ──────────────────────────────────────────────────────────────────────────────
def _q_tick(px: float, tick: float, mode: str) -> float:
    """Quantize to exchange tick with directional bias (ceil/floor)."""
    if tick <= 0:
        return round(px, 2)
    q = int(round(px / tick))
    # directional bias
    if mode == "up":
        if abs(q * tick - px) < 1e-12:
            return q * tick
        return (math.floor(px / tick) + 1) * tick
    if mode == "down":
        if abs(q * tick - px) < 1e-12:
            return q * tick
        return math.floor(px / tick) * tick
    # nearest
    return q * tick

def submit_bracket_order(
    base_url: str,
    key: str,
    secret: str,
    *,
    symbol: str,
    qty: int,
    side: str,
    limit_price: Optional[float] = None,   # None => market entry
    take_profit_price: float,
    stop_loss_price: float,
    tif: str = "day",
    tick_size: float = 0.01,               # equities tick
    ensure_tp_sl_buffer_ticks: int = 1,    # extra cushion to avoid 422 on base_price
) -> Dict[str, Any]:
    """
    Create a simple bracket (parent entry + TP/SL children). Prices are snapped
    to tick grid with directional bias to respect Alpaca's '>= base + 0.01' rule.
    """
    s = (side or "").lower()
    if s not in ("buy", "sell"):
        raise ValueError("side must be 'buy' or 'sell'")

    q = max(1, int(qty))

    # Snap child prices to tick grid with correct bias
    if s == "buy":
        tp = _q_tick(float(take_profit_price), tick_size, "up")
        sl = _q_tick(float(stop_loss_price),  tick_size, "down")
        # Add a small buffer (1 tick by default) to survive tiny base_price moves
        tp += ensure_tp_sl_buffer_ticks * tick_size
        sl -= ensure_tp_sl_buffer_ticks * tick_size
    else:  # sell/short
        tp = _q_tick(float(take_profit_price), tick_size, "down")
        sl = _q_tick(float(stop_loss_price),  tick_size, "up")
        tp -= ensure_tp_sl_buffer_ticks * tick_size
        sl += ensure_tp_sl_buffer_ticks * tick_size

    # If a limit entry is used, snap it too
    entry_payload: Dict[str, Any] = {"symbol": symbol.upper(), "side": s, "time_in_force": tif}
    if limit_price is None:
        entry_payload["type"] = "market"
        # parent market entry
    else:
        # Snap limit entry opposite to adverse direction (conservative):
        #  - buy limit: round DOWN (more likely to be accepted)
        #  - sell limit: round UP
        lim_mode = "down" if s == "buy" else "up"
        lp = _q_tick(float(limit_price), tick_size, lim_mode)
        entry_payload.update({"type": "limit", "limit_price": float(f"{lp:.2f}")})

    # Compose bracket
    payload = {
        **entry_payload,
        "qty": q,
        "order_class": "bracket",
        "take_profit": {"limit_price": float(f"{tp:.2f}")},
        "stop_loss":   {"stop_price":  float(f"{sl:.2f}")},
    }

    url = f"{base_url.rstrip('/')}/v2/orders"
    r = _req_with_retry("POST", url, headers=_headers(key, secret), json=payload, timeout=25)
    try:
        return r.json() if r.text else {}
    except Exception:
        return {"status_code": r.status_code, "text": r.text}
