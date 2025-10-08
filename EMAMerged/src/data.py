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
# Alpaca basic helpers
# ──────────────────────────────────────────────────────────────────────────────
def _alpaca_headers(key: str, secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": key or "",
        "APCA-API-SECRET-KEY": secret or "",
        "Content-Type": "application/json",
    }

def _base_url_from_env(base_url: Optional[str] = None) -> str:
    return (base_url or os.getenv("APCA_BASE_URL") or os.getenv("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")

def alpaca_market_open(base_url: Optional[str], key: str, secret: str) -> bool:
    base = _base_url_from_env(base_url)
    r = requests.get(f"{base}/v2/clock", headers=_alpaca_headers(key, secret), timeout=10)
    if r.status_code == 401:
        return False
    r.raise_for_status()
    j = r.json() or {}
    return bool(j.get("is_open"))

# Risk ops
def cancel_all_orders(base_url: Optional[str], key: str, secret: str) -> Dict[str, Any]:
    base = _base_url_from_env(base_url)
    r = requests.delete(f"{base}/v2/orders", headers=_alpaca_headers(key, secret), timeout=15)
    if r.status_code not in (200, 204):
        r.raise_for_status()
    return {"ok": True}

def close_all_positions(base_url: Optional[str], key: str, secret: str) -> Dict[str, Any]:
    base = _base_url_from_env(base_url)
    r = requests.delete(f"{base}/v2/positions", headers=_alpaca_headers(key, secret), timeout=20)
    if r.status_code not in (200, 204):
        r.raise_for_status()
    return {"ok": True}

# Positions (map by symbol for convenience)
def get_positions(base_url: Optional[str], key: str, secret: str) -> Dict[str, Dict[str, Any]]:
    base = _base_url_from_env(base_url)
    r = requests.get(f"{base}/v2/positions", headers=_alpaca_headers(key, secret), timeout=12)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    arr = r.json() or []
    out = {}
    for pos in arr:
        sym = (pos.get("symbol") or "").upper()
        if sym:
            out[sym] = pos
    return out

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
        # peek to choose key
        for b in bars[:3]:
            if "S" in b:
                sym_key = "S"
                break
            if "symbol" in b:
                sym_key = "symbol"
                break
        if sym_key is None:
            # fallback try both
            sym_key = "S"
        for b in bars:
            s = (b.get(sym_key) or b.get("S") or b.get("symbol") or "").upper()
            if s:
                out.setdefault(s, []).append(b)
        return out, "list"
    else:
        # unknown / empty
        return {}, "empty"

def _build_params(symbols: List[str], timeframe: str, start_iso: str, end_iso: str, feed: str, limit: int, page_token: Optional[str]) -> Dict[str, Any]:
    p = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start_iso,
        "end": end_iso,
        "limit": int(limit),
        "feed": feed,
        # "adjustment": "raw",  # optional; uncomment if you prefer raw (no splits)
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
    # Detect field names robustly
    # Typical fields: t (time), o (open), h (high), l (low), c (close), v (volume)
    times, o, h, l, c, v = [], [], [], [], [], []
    for b in records:
        t = b.get("t") or b.get("time")
        try:
            ts = pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")
        except Exception:
            # last resort parse
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
    # Keep only bars whose local (ET) time is within [start_hm, end_hm]
    local = df.tz_convert(tz_name)
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    mask = (local.index.hour*60 + local.index.minute >= sh*60 + sm) & \
           (local.index.hour*60 + local.index.minute <= eh*60 + em)
    return df.loc[mask].copy()

def drop_unclosed_last_bar(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if df.empty:
        return df
    # Drop the last bar if it is still "forming" (not closed yet)
    tfm = (timeframe or "15Min").lower()
    step_min = 1
    if "min" in tfm:
        step_min = int(tfm.replace("min", "").replace("m", ""))
    elif tfm in ("1h", "60min", "60m"):
        step_min = 60
    last_ts = df.index[-1]
    # If the last bar's close boundary is in the future, drop it
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
    allowed_windows: Optional[List[Dict[str, str]]] = None,  # currently unused in fetch; gating is done elsewhere
    bar_limit: int = 10000,
    key: str = "",
    secret: str = "",
    dollar_vol_window: int = 20,
    dollar_vol_min_periods: int = 7,
) -> Dict[str, pd.DataFrame]:
    """
    Robust multi-symbol fetch with pagination & dual-shape handling. Emits rich diagnostics.
    Returns: {symbol: DataFrame[open,high,low,close,volume,dollar_vol_avg]}
    """
    if not symbols:
        return {}

    # Universe log
    print(json.dumps({
        "type": "UNIVERSE",
        "loaded": len(symbols),
        "sample": symbols[:10],
        "when": _iso_utc()
    }, separators=(",", ":"), ensure_ascii=False), flush=True)

    base_data_url = "https://data.alpaca.markets/v2/stocks/bars"
    H = _alpaca_headers(key, secret)

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

    bars_map: Dict[str, pd.DataFrame] = {}
    http_errors: List[Dict[str, Any]] = []
    syms_with_data: List[str] = []
    syms_empty: List[str] = []
    stale_syms: List[str] = []

    for idx, chunk in enumerate(chunks, start=1):
        # Collect across pages
        collected: Dict[str, List[Dict[str, Any]]] = {}
        page_token = None
        pages = 0
        shape_seen = None
        MAX_PAGES = 20  # hard guard

        while True:
            pages += 1
            params = _build_params(chunk, timeframe, start_iso, end_iso, feed, bar_limit, page_token)
            try:
                r = requests.get(base_data_url, headers=H, params=params, timeout=20)
                if r.status_code == 403:
                    # Not entitled to this feed; treat as empty but log it once
                    http_errors.append({"chunk": idx, "code": 403, "msg": "Forbidden (data entitlement)", "symbols": chunk})
                    break
                r.raise_for_status()
                j = r.json() or {}
            except Exception as e:
                http_errors.append({"chunk": idx, "error": str(e), "symbols": chunk})
                break

            per_sym, shape = _bars_json_to_map(j)
            shape_seen = shape_seen or shape  # remember first shape
            for s, recs in per_sym.items():
                collected.setdefault(s, []).extend(recs or [])

            page_token = j.get("next_page_token") or None
            if not page_token or pages >= MAX_PAGES:
                break

        # Emit chunk collection summary
        print(json.dumps({
            "type": "BARS_FETCH_CHUNK_COLLECTED",
            "chunk_index": idx,
            "symbols_in_chunk": len(chunk),
            "pages": pages,
            "source_shape": shape_seen or "none",
            "collected_symbols": len(collected)
        }, separators=(",", ":"), ensure_ascii=False), flush=True)

        # Build DataFrames per symbol in this chunk
        for s in chunk:
            recs = collected.get(s) or []
            df = _build_df_from_bars(recs)
            if df.empty:
                syms_empty.append(s)
                continue

            # Optional RTH trim
            if rth_only:
                df = filter_rth(df, tz_name=tz_name, start_hm=rth_start, end_hm=rth_end)

            # Drop forming last bar (if any)
            df = drop_unclosed_last_bar(df, timeframe)

            if df.empty:
                syms_empty.append(s)
                continue

            # Dollar volume average (fix FutureWarning by using .ffill())
            dv = (df["close"] * df["volume"]).rolling(
                window=int(dollar_vol_window),
                min_periods=int(dollar_vol_min_periods)
            ).mean()
            df["dollar_vol_avg"] = dv.ffill().fillna(0.0)

            # Staleness: last bar too old relative to end?
            last_ts = df.index[-1]
            if (end - last_ts) > dt.timedelta(days=7):
                stale_syms.append(s)

            bars_map[s] = df
            syms_with_data.append(s)

    # High-level summary
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

    # One more compact snapshot (last timestamp per a few syms)
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

    print(json.dumps({
        "type": "BARS_SNAPSHOT",
        "sample": snap
    }, separators=(",", ":"), ensure_ascii=False), flush=True)

    return bars_map

# ──────────────────────────────────────────────────────────────────────────────
# Order placement — bracket with min-tick guard (avoids 42210000)
# ──────────────────────────────────────────────────────────────────────────────
def _round_to_tick(px: float, tick: float = 0.01) -> float:
    if tick <= 0:
        return round(float(px), 2)
    steps = math.floor(px / tick + 1e-9)
    return round(steps * tick, 2)

def submit_bracket_order(
    base_url: Optional[str], key: str, secret: str,
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
    # If we have a limit entry, the base is that; else base is unknown until filled — Alpaca still accepts TP/SL
    base_hint = float(limit_price) if limit_price else float(take_profit_price) - 10 * tick  # harmless bias

    tp = _round_to_tick(float(take_profit_price), tick)
    sl = _round_to_tick(float(stop_loss_price), tick)

    # Ensure TP >= base + tick (to avoid 42210000)
    min_tp = _round_to_tick(base_hint + tick, tick)
    if tp < min_tp:
        tp = min_tp

    payload["take_profit"] = {"limit_price": tp}
    payload["stop_loss"] = {"stop_price": sl}

    r = requests.post(f"{base}/v2/orders", headers=H, json=payload, timeout=20)
    # Even if 422 is returned, we surface it to caller which already logs the message.
    if r.status_code not in (200, 201, 202):
        try:
            return r.json()
        except Exception:
            r.raise_for_status()
    return r.json() or {}