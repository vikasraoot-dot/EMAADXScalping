#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live loop (paper): EMA + ADX + DI filters, bracket orders.
- Normalizes 15m strategy, scans on a 5m cadence.
- Writes compact JSONL results via TradeLogger.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import pandas as pd

# Local imports
from EMAMerged.src.data import (
    load_symbols_from_file,
    fetch_latest_bars,
    alpaca_market_open,
    submit_bracket_order,
    get_positions,
)
from EMAMerged.src.indicators import (
    ta_add_emas,
    ta_add_rsi,
    ta_add_adx_di,
    ta_add_vol_dollar,
)
from EMAMerged.src.trade_logger import TradeLogger


# -------- helpers --------
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(fmt: str = "%Y%m%d") -> str:
    # used for log filename (e.g., live_20251007.jsonl)
    return datetime.now(timezone.utc).strftime(fmt)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--tickers", required=True)
    p.add_argument("--dry-run", type=int, default=0)
    p.add_argument("--force-run", type=int, default=0)
    return p.parse_args()


def _load_config(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _session_label(now_et: datetime, rth_start: str, rth_end: str) -> str:
    # AM = from open to 12:30, PM = 12:30 -> close (just a label for logs)
    hhmm = now_et.strftime("%H:%M")
    return "AM" if hhmm < "12:30" else "PM"


def _in_entry_window(now_et: datetime, windows: List[dict]) -> bool:
    if not windows:
        return True
    hhmm = now_et.strftime("%H:%M")
    for w in windows:
        if w["start"] <= hhmm <= w["end"]:
            return True
    return False


def _cap_qty_by_notional(qty: int, entry: float, max_notional: float | None) -> int:
    if not max_notional:
        return qty
    if entry <= 0:
        return 0
    return min(qty, int(max_notional // entry))


def _pretty_timeframe(tf: str) -> str:
    # normalize like "15m" -> "15Min"
    tf = tf.strip()
    if tf.endswith("m"):
        return f"{tf[:-1]}Min"
    return tf


def _broker_creds(cfg: dict) -> tuple[str, str, str]:
    """Pull broker creds from config or environment, with safe defaults."""
    broker = cfg.get("broker", {}) or {}
    base_url = broker.get("base_url") or os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    # Support both env name variants
    key = broker.get("key") or os.getenv("ALPACA_KEY_ID") or os.getenv("APCA_API_KEY_ID") or ""
    secret = broker.get("secret") or os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY") or ""
    return base_url, key, secret


def _alpaca_market_open_compat(base_url: str, key: str, secret: str) -> bool:
    """Call alpaca_market_open with credentials; fall back to 0-arg signature if present."""
    try:
        return bool(alpaca_market_open(base_url, key, secret))
    except TypeError:
        return bool(alpaca_market_open())


def _get_positions_compat(base_url: str, key: str, secret: str):
    """Call get_positions with credentials; fall back to 0-arg signature if present."""
    try:
        return get_positions(base_url, key, secret)
    except TypeError:
        return get_positions()


def attach_verifiers(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Append all indicators + helper columns used by gates.
    """
    df = df.copy()
    fast = int(cfg.get("ema_fast", 9))
    slow = int(cfg.get("ema_slow", 21))
    rsi_len = int(cfg.get("rsi_length", cfg.get("filters", {}).get("rsi_period", 14)))
    adx_per = int(cfg.get("adx_period", 14))
    vol_win = int(cfg.get("vol_sma_length", 10))

    df = ta_add_emas(df, fast, slow)
    df = ta_add_rsi(df, rsi_len)
    df = ta_add_adx_di(df, adx_per)
    df = ta_add_vol_dollar(df, vol_win)

    # Helpers for slope and fresh cross
    df["ema_fast_slope_pct"] = df["ema_fast"].pct_change()
    df["fresh_cross_up"] = (df["ema_fast"].shift(1) <= df["ema_slow"].shift(1)) & (
        df["ema_fast"] > df["ema_slow"]
    )
    df["fresh_cross_down"] = (df["ema_fast"].shift(1) >= df["ema_slow"].shift(1)) & (
        df["ema_fast"] < df["ema_slow"]
    )
    # Directional trend gate
    df["+DI_gt_-DI"] = df["+DI"] > df["-DI"]
    return df


# -------- main --------
def main() -> int:
    args = _parse_args()
    cfg = _load_config(args.config)

    # Reference timeframe
    intervals = cfg.get("intervals") or ["15m"]
    timeframe = _pretty_timeframe(intervals[0])  # "15Min"
    history_days = int(cfg.get("history_days", 30))

    # Session clock
    import pytz

    tz = pytz.timezone(cfg.get("timezone", "US/Eastern"))
    rth_start = cfg.get("rth_start", "09:30")
    rth_end = cfg.get("rth_end", "15:55")
    now_utc = _now_utc()
    now_et = now_utc.astimezone(tz)
    session = _session_label(now_et, rth_start, rth_end)

    # Logger
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"live_{_utc_stamp('%Y%m%d')}.jsonl")
    log = TradeLogger(log_path)

    # Broker creds (for market/positions/order calls) + set env for Market Data
    base_url, key, secret = _broker_creds(cfg)
    # Ensure data helpers that rely on env headers are authenticated
    if key and secret:
        os.environ["APCA_API_KEY_ID"] = key
        os.environ["APCA_API_SECRET_KEY"] = secret

    # Universe
    symbols = load_symbols_from_file(args.tickers)
    print(
        json.dumps(
            {"type": "UNIVERSE", "loaded": len(symbols), "sample": symbols[:10], "when": now_utc.isoformat()}
        ),
        flush=True,
    )

    # Entry windows?
    windows = cfg.get("entry_windows", [])
    if not _in_entry_window(now_et, windows):
        print(json.dumps({"type": "HEARTBEAT", "session": session, "market_open": True, "ts": now_utc.isoformat()}))
        return 0

    # Market open check (paper)
    if not _alpaca_market_open_compat(base_url, key, secret):
        print(json.dumps({"type": "HEARTBEAT", "session": session, "market_open": False, "ts": now_utc.isoformat()}))
        return 0
    print(json.dumps({"type": "HEARTBEAT", "session": session, "market_open": True, "ts": now_utc.isoformat()}))

    # Fetch bars  (NOTE: function expects (symbols, timeframe, history_days, feed))
    feed = cfg.get("feed", "iex")
    print(
        json.dumps(
            {
                "type": "BARS_FETCH_START",
                "requested": len(symbols),
                "chunks": 2,
                "timeframe": timeframe,
                "feed": feed,
                "start": (now_utc - timedelta(days=history_days)).isoformat(),
                "end": now_utc.isoformat(),
                "when": now_utc.isoformat(),
            }
        ),
        flush=True,
    )

    res = None
    try:
        res = fetch_latest_bars(symbols, timeframe, history_days, feed)
    except Exception as e:
        # Hard failure from the data helper
        print(
            json.dumps(
                {
                    "type": "BARS_FETCH_ERROR",
                    "reason": "exception",
                    "message": str(e),
                    "timeframe": timeframe,
                    "feed": feed,
                    "when": now_utc.isoformat(),
                }
            ),
            flush=True,
        )
        return 1

    # Defensive unpack (some implementations return None on 401 etc.)
    if not isinstance(res, tuple) or len(res) != 2:
        print(
            json.dumps(
                {
                    "type": "BARS_FETCH_ERROR",
                    "reason": "no_result",
                    "message": "fetch_latest_bars returned no data",
                    "timeframe": timeframe,
                    "feed": feed,
                    "when": now_utc.isoformat(),
                }
            ),
            flush=True,
        )
        return 1

    bars_map, bars_meta = res

    # Summary
    with_data = sorted([s for s, d in bars_map.items() if isinstance(d, pd.DataFrame) and not d.empty])
    empty = sorted(list(set(symbols) - set(with_data)))
    stale = sorted(list((bars_meta or {}).get("stale_symbols", [])))
    http_errors = (bars_meta or {}).get("http_errors", [])

    print(
        json.dumps(
            {
                "type": "BARS_FETCH_SUMMARY",
                "requested": len(symbols),
                "with_data": len(with_data),
                "empty": len(empty),
                "stale": len(stale),
                "sample_with_data": with_data[:4],
                "sample_empty": empty[:10],
                "when": now_utc.isoformat(),
            }
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "type": "BARS_FETCH",
                "requested": len(symbols),
                "timeframe": timeframe,
                "history_days": history_days,
                "feed": feed,
                "http_errors": http_errors,
                "symbols_with_data": with_data,
                "symbols_empty": empty,
                "stale_symbols": stale,
                "when": now_utc.isoformat(),
            }
        ),
        flush=True,
    )

    # Snapshot
    snap = []
    for s in symbols[:50]:
        df = bars_map.get(s)
        if isinstance(df, pd.DataFrame) and not df.empty:
            snap.append({"s": s, "ok": True, "last": str(df.index[-1])})
        else:
            snap.append({"s": s, "ok": False})
    print(json.dumps({"type": "BARS_SNAPSHOT", "sample": snap[:50]}), flush=True)

    # Current positions (for "already in position" gate)
    positions_raw = _get_positions_compat(base_url, key, secret) or []
    positions = {p["symbol"]: p for p in positions_raw}

    # Filters
    fcfg = cfg.get("filters", {}) or {}
    adx_threshold = float(fcfg.get("adx_threshold", 23.0))
    rsi_min = float(fcfg.get("rsi_min", 50))
    rsi_max = float(fcfg.get("rsi_max", 80))
    slope_thr = float(fcfg.get("slope_threshold_pct", 0.0010))
    require_fast_above_slow = bool(fcfg.get("require_fast_above_slow", True))

    # Risk knobs
    brackets = cfg.get("brackets", {}) or {}

    # Size guard
    base_qty = int(cfg.get("qty", 1))
    max_shares = int(cfg.get("max_shares_per_trade", base_qty))
    max_notional = cfg.get("max_notional_per_trade", None)
    max_notional = float(max_notional) if max_notional not in (None, "", "0") else None

    # Iterate symbols
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 3:
            continue

        df = attach_verifiers(df, cfg)
        last = df.iloc[-1]

        # Build signal row
        cross_up = bool(last["fresh_cross_up"])
        ema_ok = bool(last["ema_fast"] > last["ema_slow"])
        di_ok = bool(last["+DI_gt_-DI"])
        adx_val = float(last["ADX"]) if not pd.isna(last["ADX"]) else np.nan
        rsi_val = float(last["rsi"]) if not pd.isna(last["rsi"]) else np.nan
        slope = float(last["ema_fast_slope_pct"]) if not pd.isna(last["ema_fast_slope_pct"]) else 0.0

        sig_row = {
            "symbol": sym,
            "session": session,
            "cid": f"EMA_{sym}_{now_utc.strftime('%Y%m%d_%H%M%S')}",
            "tf": timeframe,
            "cross": 1 if cross_up else 0,
            "ref_bar_ts": str(df.index[-1]),
            "last_close": float(last["close"]),
            "adx": float(adx_val) if not np.isnan(adx_val) else None,
            "ema_slope_pct": slope,
            "rsi": float(rsi_val) if not np.isnan(rsi_val) else None,
        }
        TradeLogger.signal(log, **sig_row)

        # Gate
        reasons = []
        if require_fast_above_slow and not ema_ok:
            reasons.append("ema_fast ≤ ema_slow")
        if cross_up is False:
            reasons.append("no fresh cross (fast>slow)")
        if not di_ok:
            reasons.append("+DI ≤ -DI (trend mismatch)")
        if not np.isnan(adx_val) and adx_val < adx_threshold:
            reasons.append(f"ADX {adx_val:.1f} < {adx_threshold:.1f}")
        if rsi_val < rsi_min:
            reasons.append(f"RSI {rsi_val:.1f} < {rsi_min:.1f}")
        if rsi_val > rsi_max:
            reasons.append(f"RSI {rsi_val:.1f} > {rsi_max:.1f}")
        if slope < slope_thr:
            reasons.append(f"EMA_slope {slope:.5f} < {slope_thr:.5f}")

        if sym in positions:
            reasons = ["already in position"]

        if reasons:
            log.gate(symbol=sym, session=session, cid=sig_row["cid"], decision="BLOCK", reasons=reasons)
            continue
        else:
            log.gate(symbol=sym, session=session, cid=sig_row["cid"], decision="ALLOW", reasons=[])

        # ATR-based brackets
        if not brackets.get("enabled", True):
            continue

        atr_len = int(cfg.get("atr_length", 14))
        # Simple ATR from True Range mean over last atr_len
        tr = pd.concat(
            [
                (df["high"] - df["low"]),
                (df["high"] - df["close"].shift(1)).abs(),
                (df["low"] - df["close"].shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.rolling(atr_len, min_periods=atr_len).mean()
        atr_val = float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

        entry = float(last["close"])
        sl_mult = float(brackets.get("atr_mult_sl", 1.2))
        tp_r = float(brackets.get("take_profit_r", 1.8))

        stop_price = round(entry - sl_mult * atr_val, 2)
        r = entry - stop_price
        take_profit = round(entry + tp_r * r, 2)

        # Min-tick guard to avoid 422 "take_profit.limit_price >= base_price + 0.01"
        tick_size = float(brackets.get("min_tick", 0.01))
        if take_profit <= entry + tick_size - 1e-9:
            take_profit = round(entry + tick_size, 2)

        if not (stop_price < entry < take_profit):
            print(f"[order] skip {sym}: bad TP/SL (entry={entry}, sl={stop_price}, tp={take_profit})", flush=True)
            try:
                log.entry_reject(
                    symbol=sym,
                    session=session,
                    cid=sig_row["cid"],
                    reason="BAD_TP_SL",
                    details={"entry": round(entry, 2), "sl": stop_price, "tp": take_profit},
                )
            except Exception:
                pass
            continue

        # Size
        qty = max(1, min(base_qty, max_shares))
        qty = _cap_qty_by_notional(qty, entry, max_notional)
        if qty < 1:
            print(
                f"[order] skip {sym}: qty<1 after notional cap (entry={entry}, max_notional={max_notional})",
                flush=True,
            )
            try:
                log.entry_reject(
                    symbol=sym,
                    session=session,
                    cid=sig_row["cid"],
                    reason="QTY_LT_1_AFTER_CAP",
                    details={"entry": round(entry, 2), "max_notional": max_notional},
                )
            except Exception:
                pass
            continue

        # Log the planned order (visible in results jsonl)
        try:
            log.entry_submit(
                symbol=sym,
                session=session,
                cid=sig_row["cid"],
                qty=qty,
                side="buy",
                entry_price=round(entry, 2),
                stop_loss_price=stop_price,
                take_profit_price=take_profit,
                atr=round(atr_val, 4),
                method="bracket",
                tif="day",
                dry_run=bool(args.dry_run),
            )
        except Exception:
            pass

        if args.dry_run:
            print(
                f"[order] DRY-RUN: would submit BRACKET {sym} qty={qty} entry≈{entry:.2f} "
                f"sl={stop_price:.2f} tp={take_profit:.2f} (ATR={atr_val:.3f})",
                flush=True,
            )
            try:
                log.entry_ack(
                    symbol=sym,
                    session=session,
                    cid=sig_row["cid"],
                    order_id=None,
                    client_order_id=None,
                    broker_resp=None,
                    dry_run=True,
                )
            except Exception:
                pass
        else:
            try:
                resp = submit_bracket_order(
                    base_url,
                    key,
                    secret,
                    symbol=sym,
                    qty=qty,
                    side="buy",
                    limit_price=None,  # market entry
                    take_profit_price=take_profit,
                    stop_loss_price=stop_price,
                    tif="day",
                )
                # Even if 422 returns JSON error, we print it here for visibility
                print(f"[order] BRACKET submitted for {sym}: {json.dumps(resp)[:300]}", flush=True)
                try:
                    oid = (resp.get("id") if isinstance(resp, dict) else None)
                    coid = (resp.get("client_order_id") if isinstance(resp, dict) else None)
                    log.entry_ack(
                        symbol=sym,
                        session=session,
                        cid=sig_row["cid"],
                        order_id=oid,
                        client_order_id=coid,
                        broker_resp=resp,
                        dry_run=False,
                    )
                except Exception:
                    pass
            except Exception as e:
                print(f"[order] ERROR submitting bracket for {sym}: {e}", flush=True)
                try:
                    log.entry_reject(
                        symbol=sym,
                        session=session,
                        cid=sig_row["cid"],
                        reason="BROKER_ERROR",
                        details={"error": str(e)},
                    )
                except Exception:
                    pass

    try:
        log.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())