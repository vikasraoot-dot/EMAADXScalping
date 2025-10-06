# === EMAMerged/src/data.py ===
from __future__ import annotations

import os, json, math, time, datetime as dt
from typing import Dict, List, Any, Optional, Tuple
import requests
import pandas as pd

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _now_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def _print_json(obj: Dict[str, Any]) -> None:
    """
    Emit a compact JSON line for logs (UTF-8, single line).
    """
    try:
        print(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), flush=True)
    except Exception:
        # never fail the trading loop on logging
        pass

def _alpaca_headers(key: str, secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": key or "",
        "APCA-API-SECRET-KEY": secret or "",
        "Content-Type": "application/json",
    }

def _req_json(method: str, url: str, headers: Dict[str, str], timeout: int = 20, **kw) -> Tuple[int, Dict[str, Any], str]:
    """
    Safe HTTP wrapper — always returns (status_code, json_or_empty, text).
    """
    try:
        r = requests.request(method, url, headers=headers, timeout=timeout, **kw)
        try:
            j = r.json() if (r.text and r.headers.get("content-type", "").lower().startswith("application/json")) else {}
        except Exception:
            j = {}
        return r.status_code, j, (r.text or "")
    except Exception as e:
        return -1, {}, f"{type(e).__name__}: {e}"

def _chunk_symbols(symbols: List[str], max_per_chunk: int = 45) -> List[List[str]]:
    """
    Alpaca multi-symbol bars endpoint works fine up to ~50 symbols per call.
    We stay under that (45) to avoid 414 URL or internal limits.
    """
    clean = [s.strip().upper() for s in symbols if s and s.strip()]
    return [clean[i:i + max_per_chunk] for i in range(0, len(clean), max_per_chunk)]

def _parse_hhmm(s: str) -> Tuple[int, int]:
    hh, mm = s.split(":")
    return int(hh), int(mm)

def filter_rth(df: pd.DataFrame, tz_name: str, start_hm: str, end_hm: str) -> pd.DataFrame:
    if df.empty:
        return df
    # Convert to local tz, apply intraday time mask, convert back to UTC
    local = df.copy()
    local.index = local.index.tz_convert(tz_name)

    sh, sm = _parse_hhmm(start_hm)
    eh, em = _parse_hhmm(end_hm)
    mask = (local.index.hour > sh) | ((local.index.hour == sh) & (local.index.minute >= sm))
    mask &= (local.index.hour < eh) | ((local.index.hour == eh) & (local.index.minute <= em))

    kept = local[mask]
    kept.index = kept.index.tz_convert("UTC")
    return kept

def drop_unclosed_last_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Drop the last row if it is likely still forming.
    Heuristic: if now_utc < last_ts + bar_delta.
    """
    if df.empty:
        return df
    tf = timeframe.lower()
    if tf.endswith("min"):
        try:
            minutes = int(tf.replace("min", ""))
        except Exception:
            minutes = 15
        delta = dt.timedelta(minutes=minutes)
    elif tf.endswith("hour"):
        try:
            hours = int(tf.replace("hour", ""))
        except Exception:
            hours = 1
        delta = dt.timedelta(hours=hours)
    else:
        # Fallback for e.g. "15Min" formats
        num = "".join(ch for ch in tf if ch.isdigit())
        minutes = int(num) if num else 15
        delta = dt.timedelta(minutes=minutes)

    last_ts = df.index[-1]
    if dt.datetime.now(dt.UTC) < (last_ts + delta):
        return df.iloc[:-1]
    return df

def _to_df(bars_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert Alpaca bars list (dicts with keys t,o,h,l,c,v,...) into DataFrame.
    Index is UTC Timestamp.
    """
    if not bars_list:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype="float64")

    raw = pd.DataFrame(bars_list)
    # Expect 't','o','h','l','c','v'
    rename = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "t": "time"}
    for k in rename:
        if k not in raw.columns:
            raw[k] = None
    raw = raw.rename(columns=rename)

    # Parse UTC time
    idx = pd.to_datetime(raw["time"], utc=True, errors="coerce")
    raw = raw.assign(index_ts=idx).dropna(subset=["index_ts"]).set_index("index_ts").sort_index()

    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c not in raw.columns:
            raw[c] = float("nan")
    return raw[cols]

def _rolling_dollar_vol(df: pd.DataFrame, window: int, min_periods: int) -> pd.Series:
    dv = (df["close"] * df["volume"]).rolling(window=window, min_periods=min_periods).mean()
    # Avoid deprecated fillna(method=...) — use .ffill() / .bfill()
    return dv.ffill().fillna(0.0)

# ---------------------------------------------------------------------
# Public: Market status & risk ops
# ---------------------------------------------------------------------

def alpaca_market_open(base_url: str, key: str, secret: str) -> bool:
    url = f"{base_url.rstrip('/')}/v2/clock"
    st, j, _ = _req_json("GET", url, _alpaca_headers(key, secret), timeout=10)
    if st != 200:
        _print_json({"type": "CLOCK_ERROR", "status": st, "when": _now_iso()})
        return False
    return bool(j.get("is_open"))

def cancel_all_orders(base_url: str, key: str, secret: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v2/orders"
    st, j, txt = _req_json("DELETE", url, _alpaca_headers(key, secret), timeout=20)
    if st not in (200, 204):
        _print_json({"type": "CANCEL_ALL_ERROR", "status": st, "text": txt[:200]})
    return j or {}

def close_all_positions(base_url: str, key: str, secret: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v2/positions"
    st, j, txt = _req_json("DELETE", url, _alpaca_headers(key, secret), timeout=20)
    if st not in (200, 207):  # 207 Multi-Status sometimes
        _print_json({"type": "CLOSE_ALL_ERROR", "status": st, "text": txt[:200]})
    return j or {}

def get_positions(base_url: str, key: str, secret: str) -> Dict[str, Dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/v2/positions"
    st, j, txt = _req_json("GET", url, _alpaca_headers(key, secret), timeout=12)
    if st != 200:
        _print_json({"type": "GET_POSITIONS_ERROR", "status": st, "text": txt[:200]})
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for pos in (j if isinstance(j, list) else []):
        sym = (pos.get("symbol") or "").upper()
        if sym:
            out[sym] = pos
    return out

def submit_bracket_order(
    base_url: str,
    key: str,
    secret: str,
    *,
    symbol: str,
    qty: int,
    side: str,
    limit_price: Optional[float],
    take_profit_price: float,
    stop_loss_price: float,
    tif: str = "day",
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v2/orders"
    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "qty": int(qty),
        "side": side.lower(),
        "type": "market" if (limit_price in (None, 0)) else "limit",
        "time_in_force": str(tif).lower(),
        "order_class": "bracket",
        "take_profit": {"limit_price": float(take_profit_price)},
        "stop_loss": {"stop_price": float(stop_loss_price)},
    }
    if limit_price not in (None, 0):
        payload["limit_price"] = float(limit_price)

    st, j, txt = _req_json("POST", url, _alpaca_headers(key, secret), timeout=20, json=payload)
    if st not in (200, 201, 202):
        _print_json({"type": "ORDER_SUBMIT_ERROR", "status": st, "text": txt[:300], "symbol": symbol, "when": _now_iso()})
        # Return the error body so the caller can log it
        return j or {"status": st, "error": txt[:300]}
    return j or {}

# ---------------------------------------------------------------------
# Public: Bars fetch with rich diagnostics
# ---------------------------------------------------------------------

def fetch_latest_bars(
    symbols: List[str],
    *,
    timeframe: str,
    history_days: int,
    feed: str,
    rth_only: bool,
    tz_name: str,
    rth_start: str,
    rth_end: str,
    allowed_windows: Optional[List[Dict[str, str]]] = None,
    bar_limit: int = 10000,
    key: str,
    secret: str,
    dollar_vol_window: int = 20,
    dollar_vol_min_periods: int = 7,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch bars for all symbols with detailed chunk diagnostics.
    Returns a dict mapping symbol -> DataFrame (UTC index).
    Adds a special '__meta__' key containing a diagnostics dict.
    """
    symbols = [s.strip().upper() for s in symbols if s and s.strip()]
    H = _alpaca_headers(key, secret)
    base_data_url = "https://data.alpaca.markets/v2/stocks/bars"

    # Time bounds
    end_iso = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    start_iso = (dt.datetime.now(dt.UTC) - dt.timedelta(days=int(history_days))).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    chunks = _chunk_symbols(symbols)
    diag: Dict[str, Any] = {
        "type": "BARS_FETCH",
        "requested": len(symbols),
        "timeframe": timeframe,
        "history_days": int(history_days),
        "feed": feed,
        "chunks": [],
        "http_errors": [],
        "symbols_with_data": [],
        "symbols_empty": [],
        "stale_symbols": [],  # last bar older than 1d
        "when": _now_iso(),
    }

    out: Dict[str, pd.DataFrame] = {}

    # Header log
    _print_json({
        "type": "BARS_FETCH_START",
        "requested": len(symbols),
        "chunks": len(chunks),
        "timeframe": timeframe,
        "feed": feed,
        "start": start_iso,
        "end": end_iso,
        "when": _now_iso(),
    })

    for i, chunk in enumerate(chunks, 1):
        params = {
            "symbols": ",".join(chunk),
            "timeframe": timeframe,
            "start": start_iso,
            "end": end_iso,
            "limit": int(bar_limit),
            "feed": feed,
        }
        st, j, txt = _req_json("GET", base_data_url, H, timeout=20, params=params)
        chunk_info = {"idx": i, "count": len(chunk), "status": st, "symbols": chunk[:6] + (["..."] if len(chunk) > 6 else [])}
        diag["chunks"].append(chunk_info)

        if st != 200:
            diag["http_errors"].append({"idx": i, "status": st, "text": (txt or "")[:200]})
            _print_json({"type": "BARS_FETCH_CHUNK_ERROR", "idx": i, "status": st, "text": (txt or "")[:200], "when": _now_iso()})
            # Continue to next chunk — we want the rest to load
            continue

        bars_map = (j.get("bars") or {}) if isinstance(j, dict) else {}
        # Per symbol build
        for sym in chunk:
            recs = bars_map.get(sym) or []
            df = _to_df(recs)

            # RTH trimming
            if rth_only:
                df = filter_rth(df, tz_name=tz_name, start_hm=rth_start, end_hm=rth_end)

            # Drop an unclosed last bar
            df = drop_unclosed_last_bar(df, timeframe=timeframe)

            # Dollar volume rolling average column (used by filters)
            if not df.empty:
                try:
                    df["dollar_vol_avg"] = _rolling_dollar_vol(df, window=int(dollar_vol_window), min_periods=int(dollar_vol_min_periods))
                except Exception:
                    # Never break flow on an indicator calc issue
                    df["dollar_vol_avg"] = 0.0

            out[sym] = df

    # Summarize
    with_data, empty = [], []
    stale_cut = dt.datetime.now(dt.UTC) - dt.timedelta(days=1)
    stale_syms = []

    for sym in symbols:
        df = out.get(sym)
        if df is not None and not df.empty:
            with_data.append(sym)
            try:
                if df.index[-1] < stale_cut:
                    stale_syms.append(sym)
            except Exception:
                pass
        else:
            empty.append(sym)

    diag["symbols_with_data"] = with_data
    diag["symbols_empty"] = empty
    diag["stale_symbols"] = stale_syms

    _print_json({
        "type": "BARS_FETCH_SUMMARY",
        "requested": len(symbols),
        "with_data": len(with_data),
        "empty": len(empty),
        "stale": len(stale_syms),
        "sample_with_data": with_data[:10],
        "sample_empty": empty[:10],
        "when": _now_iso(),
    })

    # Attach meta (non-symbol key that live loop can optionally print)
    out["__meta__"] = diag
    return out