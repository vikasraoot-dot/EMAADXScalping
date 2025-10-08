# === EMAMerged/src/data.py ===
from __future__ import annotations
import os, json, time, math, datetime as dt
from typing import Dict, List, Any, Optional, Iterable, Tuple
import requests
import pandas as pd

ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

# ──────────────────────────────────────────────────────────────────────────────
# Back-compat shim for older callers
# ──────────────────────────────────────────────────────────────────────────────
def load_symbols_from_file(path: str) -> list[str]:
    """
    Compatibility wrapper used by some scripts.
    Delegates to EMAMerged.src.utils.read_tickers.
    """
    try:
        from EMAMerged.src.utils import read_tickers
        return read_tickers(path)
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Small time helpers
# ──────────────────────────────────────────────────────────────────────────────
def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.UTC)

def _iso_utc(ts: Optional[dt.datetime] = None) -> str:
    ts = ts or _now_utc()
    return ts.strftime(ISO_UTC)

def _minutes(td: dt.timedelta) -> int:
    return int(td.total_seconds() // 60)

# ──────────────────────────────────────────────────────────────────────────────
# Creds / Alpaca helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_key_secret(key: Optional[str], secret: Optional[str]) -> Tuple[str, str]:
    """
    Prefer explicit args; otherwise fall back to common env names.
    """
    k = (key or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY_ID") or "").strip()
    s = (secret or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY") or "").strip()
    return k, s

def _alpaca_headers(key: Optional[str], secret: Optional[str]) -> Dict[str, str]:
    k, s = _resolve_key_secret(key, secret)
    return {
        "APCA-API-KEY-ID": k,
        "APCA-API-SECRET-KEY": s,
        "Content-Type": "application/json",
    }

def _base_url_from_env(base_url: Optional[str] = None) -> str:
    # Support all common envs; prefer APCA_API_BASE_URL if set.
    return (
        (base_url or os.getenv("APCA_API_BASE_URL") or os.getenv("APCA_BASE_URL") or os.getenv("ALPACA_BASE_URL")
         or "https://paper-api.alpaca.markets")
        .rstrip("/")
    )

def alpaca_market_open(base_url: Optional[str] = None, key: Optional[str] = None, secret: Optional[str] = None) -> bool:
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/clock", headers=_alpaca_headers(key, secret), timeout=10)
        if r.status_code == 401:
            return False
        r.raise_for_status()
        j = r.json() or {}
        return bool(j.get("is_open"))
    except Exception:
        return False

# Risk ops
def cancel_all_orders(base_url: Optional[str], key: Optional[str], secret: Optional[str]) -> Dict[str, Any]:
    base = _base_url_from_env(base_url)
    r = requests.delete(f"{base}/v2/orders", headers=_alpaca_headers(key, secret), timeout=15)
    if r.status_code not in (200, 204):
        r.raise_for_status()
    return {"ok": True}

def close_all_positions(base_url: Optional[str], key: Optional[str], secret: Optional[str]) -> Dict[str, Any]:
    base = _base_url_from_env(base_url)
    r = requests.delete(f"{base}/v2/positions", headers=_alpaca_headers(key, secret), timeout=20)
    if r.status_code not in (200, 204):
        r.raise_for_status()
    return {"ok": True}

# Positions (return a LIST to match live loop usage)
def get_positions(base_url: Optional[str] = None, key: Optional[str] = None, secret: Optional[str] = None) -> List[Dict[str, Any]]:
    base = _base_url_from_env(base_url)
    try:
        r = requests.get(f"{base}/v2/positions", headers=_alpaca_headers(key, secret), timeout=12)
        if r.status_code == 404:
            return []
        r.raise_for_status()
        arr = r.json() or []
        # Ensure list-of-dicts
        return arr if isinstance(arr, list) else []
    except Exception:
        return []

# ──────────────────────────────────────────────────────────────────────────────
# Multi-symbol bars (robust shape + pagination)
# ──────────────────────────────────────────────────────────────────────────────
def _chunked(seq: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def _bars_json_to_map(j: Dict[str, Any]) -> Tuple[Dict[str, List[Dict[str, Any]]], str]:
    """
    Normalize Alpaca bars JSON into {symbol: [bar, ...]} and return (map, shape_tag).

    Alpaca /v2/stocks/bars can return:
      A) {"bars": {"AAPL":[{...}], "TSLA":[{...}]}, "next_page_token": "..."}  # dict keyed by symbol
      B) {"bars": [{...,"S":"AAPL"}, {...,"S":"TSLA"}, ...], "next_page_token": "..."}  # flat list
         (Sometimes uses "symbol" instead of "S")
    """
    bars = j.get("bars")
    out: Dict[str, List[Dict[str, Any]]] = {}
    if isinstance(bars, dict):
        # shape A
        for s, recs in bars.items():
            if not s:
                continue
            out.setdefault(s.upper(), []).extend(recs or [])
        return out, "dict"
    elif isinstance(bars, list):
        # shape B
        sym_key = None
        for b in bars[:3]:
            if "S" in b:
                sym_key = "S"; break
            if "symbol" in b:
                sym_key = "symbol"; break
        if sym_key is None:
            sym_key = "S"
        for b in bars:
            s = (b.get(sym_key) or b.get("S") or b.get("symbol") or "").upper()
            if s:
                out.setdefault(s, []).append(b)
        return out, "list"
    else:
        return {}, "empty"

def _build_params(symbols: List[str], timeframe: str, start_iso: str, end_iso: str, feed: str, limit: int, page_token: Optional[str]) -> Dict[str, Any]:
    p = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start_iso,
        "end": end_iso,
        "limit": int(limit),
        "feed": feed,
        # "adjustment": "raw",
    }
    if page_token:
        p["page_token"] = page_token
    return p

def _build_df_from_bars(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert list of Alpaca bar dicts into a DataFrame with UTC index.
    Accepts both single-symbol endpoint bar shape and multi list shape.
    """
    if not records:
        return pd.DataFrame()
    times, o, h, l, c, v = [], [], [], [], [], []
    for b in records:
        t = b.get("t") or b.get("time")
        try:
            ts = pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")
        except Exception:
            ts = pd.Timestamp(str(t), tz="UTC")
        times.append(ts)
        o.append(float(b.get("o") or b.get("open") or 0.0))
        h.append(float(b.get("h") or b.get("high") or 0.0))
        l.append(float(b.get("l") or b.get("low") or 0.0))
        c.append(float(b.get("c") or b.get("close") or 0.0))
        v.append(float(b.get("v") or b.get("volume") or 0.0))
    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": v},
        index=pd.DatetimeIndex(times, tz="UTC"),
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df

def filter_rth(df: pd.DataFrame, tz_name: str = "US/Eastern", start_hm: str = "09:30", end_hm: str = "15:55") -> pd.DataFrame:
    if df.empty:
        return df
    local = df.tz_convert(tz_name)
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    mask = (local.index.hour*60 + local.index.minute >= sh*60 + sm) & \
           (local.index.hour*60 + local.index.minute <= eh*60 + em)
    return df.loc[mask].copy()

def drop_unclosed_last_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    tfm = (timeframe or "15Min").lower()
    step_min = 1
    if "min" in tfm:
        step_min = int(tfm.replace("min", "").replace("m", ""))
    elif tfm in ("1h", "60min", "60m"):
        step_min = 60
    last_ts = df.index[-1]
    if (_now_utc() - last_ts) < dt.timedelta(minutes=step_min):
        return df.iloc[:-1].copy()
    return df

def fetch_latest_bars(
    symbols: List[str],
    timeframe: str = "15Min",
    history_days: int = 30,
    feed: str = "iex",
    rth_only: bool = True,
    tz_name: str = "US/Eastern",
    rth_start: str = "09:30",
    rth_end: str = "15:55",
    allowed_windows: Optional[List[Dict[str, str]]] = None,  # currently unused here
    bar_limit: int = 10000,
    key: Optional[str] = None,
    secret: Optional[str] = None,
    dollar_vol_window: int = 20,
    dollar_vol_min_periods: int = 7,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """
    Robust multi-symbol fetch with pagination & dual-shape handling. Emits rich diagnostics.
    Returns: (bars_map, meta) where meta={"http_errors":[...], "stale_symbols":[...]}
    """
    bars_map: Dict[str, pd.DataFrame] = {}
    http_errors: List[Dict[str, Any]] = []
    syms_with_data: List[str] = []
    syms_empty: List[str] = []
    stale_syms: List[str] = []

    if not symbols:
        return bars_map, {"http_errors": [], "stale_symbols": []}

    # Universe log (kept to match your current behavior)
    print(json.dumps({
        "type": "UNIVERSE",
        "loaded": len(symbols),
        "sample": symbols[:10],
        "when": _iso_utc()
    }, separators=(",", ":"), ensure_ascii=False), flush=True)

    base_data_url = "https://data.alpaca.markets/v2/stocks/bars"
    # Resolve creds from args or env (fixes 401 when args are not provided)
    H = _alpaca_headers(key, secret)
    if not H.get("APCA-API-KEY-ID") or not H.get("APCA-API-SECRET-KEY"):
        http_errors.append({
            "chunk": 0,
            "error": "missing_api_keys",
            "symbols": symbols[:min(10, len(symbols))],
            "hint": "Set APCA_API_KEY_ID / APCA_API_SECRET_KEY (or pass key/secret to fetch_latest_bars).",
        })
        # still continue; requests will 401, but we won’t crash

    end = _now_utc().replace(microsecond=0)
    start = end - dt.timedelta(days=max(1, int(history_days)))
    start_iso = start.strftime(ISO_UTC)
    end_iso = end.strftime(ISO_UTC)

    chunks = list(_chunked([s.upper() for s in symbols], 25))
    print(json.dumps({
        "type": "BARS_FETCH_START",
        "requested": len(symbols),
        "chunks": len(chunks),
        "timeframe": timeframe,
        "feed": feed,
        "start": start_iso,
        "end": end_iso,
        "when": _iso_utc()
    }, separators=(",", ":"), ensure_ascii=False), flush=True)

    for idx, chunk in enumerate(chunks, start=1):
        collected: Dict[str, List[Dict[str, Any]]] = {}
        page_token = None
        pages = 0
        shape_seen = None
        MAX_PAGES = 20

        while True:
            pages += 1
            params = _build_params(chunk, timeframe, start_iso, end_iso, feed, bar_limit, page_token)
            try:
                r = requests.get(base_data_url, headers=H, params=params, timeout=20)
                if r.status_code in (401, 403):
                    http_errors.append({
                        "chunk": idx, "code": r.status_code,
                        "msg": "Unauthorized" if r.status_code == 401 else "Forbidden (data entitlement)",
                        "symbols": chunk
                    })
                    break
                r.raise_for_status()
                j = r.json() or {}
            except Exception as e:
                http_errors.append({"chunk": idx, "error": str(e), "symbols": chunk})
                break

            per_sym, shape = _bars_json_to_map(j)
            shape_seen = shape_seen or shape
            for s, recs in per_sym.items():
                collected.setdefault(s, []).extend(recs or [])

            page_token = j.get("next_page_token") or None
            if not page_token or pages >= MAX_PAGES:
                break

        print(json.dumps({
            "type": "BARS_FETCH_CHUNK_COLLECTED",
            "chunk_index": idx,
            "symbols_in_chunk": len(chunk),
            "pages": pages,
            "source_shape": shape_seen or "none",
            "collected_symbols": len(collected)
        }, separators=(",", ":"), ensure_ascii=False), flush=True)

        for s in chunk:
            recs = collected.get(s) or []
            df = _build_df_from_bars(recs)
            if df.empty:
                syms_empty.append(s)
                continue

            if rth_only:
                df = filter_rth(df, tz_name=tz_name, start_hm=rth_start, end_hm=rth_end)
            df = drop_unclosed_last_bar(df, timeframe)

            if df.empty:
                syms_empty.append(s)
                continue

            dv = (df["close"] * df["volume"]).rolling(
                window=int(dollar_vol_window),
                min_periods=int(dollar_vol_min_periods)
            ).mean()
            df["dollar_vol_avg"] = dv.ffill().fillna(0.0)

            last_ts = df.index[-1]
            if (end - last_ts) > dt.timedelta(days=7):
                stale_syms.append(s)

            bars_map[s] = df
            syms_with_data.append(s)

    print(json.dumps({
        "type": "BARS_FETCH_SUMMARY",
        "requested": len(symbols),
        "with_data": len(syms_with_data),
        "empty": len(syms_empty),
        "stale": len(stale_syms),
        "sample_with_data": syms_with_data[:4],
        "sample_empty": syms_empty[:10],
        "when": _iso_utc()
    }, separators=(",", ":"), ensure_ascii=False), flush=True)

    snap = []
    for s in (syms_with_data[:2] + stale_syms[:1] + syms_empty[:1]):
        if s in bars_map and not bars_map[s].empty:
            snap.append({"s": s, "ok": True, "last": str(bars_map[s].index[-1])})
        else:
            snap.append({"s": s, "ok": False})

    print(json.dumps({
        "type": "BARS_FETCH",
        "requested": len(symbols),
        "timeframe": timeframe,
        "history_days": int(history_days),
        "feed": feed,
        "http_errors": http_errors,
        "symbols_with_data": syms_with_data,
        "symbols_empty": syms_empty,
        "stale_symbols": stale_syms,
        "when": _iso_utc()
    }, separators=(",", ":"), ensure_ascii=False), flush=True)

    print(json.dumps({"type": "BARS_SNAPSHOT", "sample": snap},
                     separators=(",", ":"), ensure_ascii=False), flush=True)

    # >>> Minimal churn change here: return bars_map **and** meta <<<
    return bars_map, {"http_errors": http_errors, "stale_symbols": stale_syms}

# ──────────────────────────────────────────────────────────────────────────────
# Order placement — bracket with min-tick guard (avoids 42210000)
# ──────────────────────────────────────────────────────────────────────────────
def _round_to_tick(px: float, tick: float = 0.01) -> float:
    if tick <= 0:
        return round(float(px), 2)
    steps = math.floor(px / tick + 1e-9)
    return round(steps * tick, 2)

def submit_bracket_order(
    base_url: Optional[str], key: Optional[str], secret: Optional[str],
    symbol: str, qty: int, side: str = "buy",
    limit_price: Optional[float] = None,   # None => market entry
    take_profit_price: float = 0.0,
    stop_loss_price: float = 0.0,
    tif: str = "day",
) -> Dict[str, Any]:
    """
    Places a "market/limit + OCO" style bracket.
    Enforces min-tick on TP vs base: TP >= base + 1 * tick (0.01 default for stocks).
    """
    base = _base_url_from_env(base_url)
    H = _alpaca_headers(key, secret)

    side = (side or "buy").lower()
    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "qty": int(qty),
        "side": side,
        "time_in_force": tif,
    }

    if limit_price is None:
        payload["type"] = "market"
    else:
        payload["type"] = "limit"
        payload["limit_price"] = float(limit_price)

    # Attach take-profit / stop-loss
    tick = 0.01
    base_hint = float(limit_price) if limit_price else float(take_profit_price) - 10 * tick  # harmless bias
    tp = _round_to_tick(float(take_profit_price), tick)
    sl = _round_to_tick(float(stop_loss_price), tick)

    min_tp = _round_to_tick(base_hint + tick, tick)
    if tp < min_tp:
        tp = min_tp

    payload["order_class"] = "bracket"
    payload["take_profit"] = {"limit_price": tp}
    payload["stop_loss"]  = {"stop_price": sl}

    r = requests.post(f"{base}/v2/orders", headers=H, json=payload, timeout=20)
    # Return JSON body even on 4xx to let caller log error details
    try:
        return r.json()
    except Exception:
        r.raise_for_status()
        return {}