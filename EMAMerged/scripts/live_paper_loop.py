# === EMAMerged/scripts/live_paper_loop.py ===
from __future__ import annotations

import os, sys, json, argparse
import datetime as dt
from typing import Dict, List, Any, Optional

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.filters import attach_verifiers, explain_long_gate, long_ok
from EMAMerged.src.indicators import atr as _atr
from EMAMerged.src.data import (
    fetch_latest_bars,       # dict[symbol] -> DataFrame (+ "__meta__")
    alpaca_market_open,      # (base_url, key, secret) -> bool
    cancel_all_orders,       # risk ops
    close_all_positions,     # risk ops
    submit_bracket_order,    # order placement
    get_positions,           # avoid stacking
)

ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

def now_iso_utc() -> str:
    return dt.datetime.now(dt.UTC).strftime(ISO_UTC)

def _utc_stamp(fmt: str) -> str:
    return dt.datetime.now(dt.UTC).strftime(fmt)

def _build_cid(symbol: str) -> str:
    return f"EMA_{symbol}_{_utc_stamp('%Y%m%d_%H%M%S')}"

def _print_json(obj: Dict[str, Any]) -> None:
    try:
        print(json.dumps(obj, separators=(",", ":"), ensure_ascii=False), flush=True)
    except Exception:
        pass

# ---------------------------------------------------------------------
# Broker creds resolution (env first, then config)
# ---------------------------------------------------------------------
def _resolve_broker(cfg: Dict) -> Dict[str, str]:
    bk = dict(cfg.get("broker", {}))
    base_url = os.getenv("APCA_BASE_URL", bk.get("base_url", "https://paper-api.alpaca.markets"))

    key_id = os.getenv("APCA_API_KEY_ID", bk.get("key") or "")
    secret = os.getenv("APCA_API_SECRET_KEY", bk.get("secret") or "")
    key_id = os.getenv("ALPACA_KEY", key_id)       # back-compat
    secret = os.getenv("ALPACA_SECRET", secret)    # back-compat

    return {"base_url": base_url.rstrip("/"), "key": key_id, "secret": secret}

# ---------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------
def minutes_to_close_et(close_hm: tuple[int, int] = (16, 0)) -> int:
    now_et = new_york_now()
    close_et = now_et.replace(hour=close_hm[0], minute=close_hm[1], second=0, microsecond=0)
    delta = close_et - now_et
    return int(delta.total_seconds() // 60)

# ---------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------
def _cap_qty_by_notional(qty: int, price: float, max_notional: Optional[float]) -> int:
    if max_notional in (None, 0, "0", "", "None"):
        return max(0, int(qty))
    try:
        cap = float(max_notional)
        max_by_notional = int(max(cap // max(price, 0.01), 0))
        return max(0, min(int(qty), max_by_notional))
    except Exception:
        return max(0, int(qty))

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--dry-run", type=int, default=1)
    ap.add_argument("--force-run", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    timeframe = cfg.get("timeframe", "15Min")
    history_days = int(cfg.get("history_days", 10))
    feed = cfg.get("feed", "iex")
    rth_only = bool(cfg.get("rth_only", True))

    # Risk settings
    entry_cutoff_min = int(cfg.get("entry_cutoff_min", 0))
    flatten_min_before_close = int(cfg.get("flatten_minutes_before_close", 0))

    # Broker + market status
    broker = _resolve_broker(cfg)
    market_open = alpaca_market_open(broker["base_url"], broker["key"], broker["secret"])

    # Tickers
    symbols = read_tickers(args.tickers)
    _print_json({"type": "UNIVERSE", "loaded": len(symbols), "sample": symbols[:10], "when": now_iso_utc()})

    # Session + heartbeat
    session = "AM" if new_york_now().hour < 12 else "PM"
    _print_json({"session": session, "market_open": bool(market_open), "type": "HEARTBEAT", "ts": now_iso_utc()})

    if not market_open and not args.force_run:
        return 0

    # Logger
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"live_{_utc_stamp('%Y%m%d')}.jsonl")
    log = TradeLogger(log_path)

    # Risk Ops: Flatten window
    mins_to_close = minutes_to_close_et((16, 0))
    entry_allowed = True
    if flatten_min_before_close > 0 and mins_to_close <= flatten_min_before_close:
        if args.dry_run:
            print(f'[risk] DRY-RUN: would FLATTEN (cancel orders & close positions) with {mins_to_close} min to close', flush=True)
        else:
            try:
                print(f'[risk] FLATTEN: canceling all orders (mins_to_close={mins_to_close})', flush=True)
                _ = cancel_all_orders(broker["base_url"], broker["key"], broker["secret"])
                print(f'[risk] FLATTEN: closing all positions', flush=True)
                _ = close_all_positions(broker["base_url"], broker["key"], broker["secret"])
            except Exception as e:
                print(f"[risk] ERROR during flatten: {e}", flush=True)
        entry_allowed = False
    else:
        entry_allowed = (mins_to_close > entry_cutoff_min) if entry_cutoff_min > 0 else True

    # Pull bars with diagnostics
    fcfg = dict(cfg.get("filters", {}))
    bars_map = fetch_latest_bars(
        symbols,
        timeframe=timeframe,
        history_days=history_days,
        feed=feed,
        rth_only=rth_only,
        tz_name=cfg.get("timezone", "US/Eastern"),
        rth_start=cfg.get("rth_start", "09:30"),
        rth_end=cfg.get("rth_end", "15:55"),
        allowed_windows=cfg.get("entry_windows"),
        bar_limit=int(cfg.get("bar_limit", 10000)),
        key=broker["key"],
        secret=broker["secret"],
        dollar_vol_window=int(fcfg.get("dollar_vol_window", 20)),
        dollar_vol_min_periods=int(fcfg.get("dollar_vol_min_periods", 7)),
    )

    # If data.py attached a meta diagnostics block, surface it once
    if isinstance(bars_map, dict) and "__meta__" in bars_map:
        meta = dict(bars_map["__meta__"])
        meta.pop("chunks", None)  # chunks are already logged by data.py
        _print_json({"type": "BARS_META", **meta})

    # Quick per-symbol snapshot (compact): did we get bars and what's the last ts?
    snapshot = []
    for s in symbols[:50]:  # cap to avoid giant lines
        df = bars_map.get(s)
        if df is None or df.empty:
            snapshot.append({"s": s, "ok": False})
        else:
            try:
                snapshot.append({"s": s, "ok": True, "last": str(df.index[-1])})
            except Exception:
                snapshot.append({"s": s, "ok": True})
    _print_json({"type": "BARS_SNAPSHOT", "sample": snapshot, "when": now_iso_utc()})

    # Current positions (avoid stacking)
    try:
        pos_map = get_positions(broker["base_url"], broker["key"], broker["secret"]) or {}
    except Exception:
        pos_map = {}

    # Bracket configuration
    br_cfg = dict(cfg.get("brackets", {}))
    brackets_enabled = bool(br_cfg.get("enabled", True))
    atr_mult_sl = float(br_cfg.get("atr_mult_sl", 1.2))
    tp_r = float(br_cfg.get("take_profit_r", 1.8))
    tick_size = float(cfg.get("breakeven", {}).get("tick_size", 0.01))  # reuse for min-tick guard

    allow_shorts = bool(cfg.get("allow_shorts", False))  # long-only here
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

        # Signal snapshot
        try:
            cross_now = float(last.get("ema_fast", 0.0)) > float(last.get("ema_slow", 0.0))
        except Exception:
            cross_now = False

        sig_row = {
            "symbol": sym,
            "session": session,
            "cid": _build_cid(sym),
            "tf": timeframe,
            "cross": 1 if cross_now else 0,
            "ref_bar_ts": str(df.index[-1]),
            "last_close": float(last.get("close", 0.0)),
            "adx": float(last.get("adx", 0.0)),
            "ema_slope_pct": float(last.get("ema_slope_pct", 0.0)),
            "rsi": float(last.get("rsi", 50.0)) if "rsi" in df.columns else None,
        }
        sig_row = {k: v for k, v in sig_row.items() if v is not None}
        log.signal(**sig_row)

        # Base gate
        ok, reasons = explain_long_gate(last, cfg)

        # Overlay: entry cutoff and existing position
        if not entry_allowed:
            reasons = list(reasons) + [f"entry_cutoff_min: {max(mins_to_close, 0)} min to close"]
            ok = False

        if ok and sym in pos_map and float(pos_map[sym].get("qty", 0)) > 0:
            reasons = list(reasons) + ["already in position"]
            ok = False

        # Log gate
        log.gate(symbol=sym, session=session, cid=sig_row["cid"], decision=("ALLOW" if ok else "BLOCK"), reasons=reasons)

        # Place order if allowed
        if not ok or not brackets_enabled:
            continue

        # ATR-based brackets
        atr_len = int(cfg.get("atr_length", 14))
        try:
            atr_series = _atr(df, period=atr_len)
            atr_val = float(atr_series.iloc[-1])
        except Exception:
            atr_val = float("nan")

        entry = float(last.get("close", 0.0))
        if not (atr_val > 0 and entry > 0):
            print(f"[order] skip {sym}: invalid ATR/entry (ATR={atr_val}, entry={entry})", flush=True)
            continue

        risk_per_share = atr_mult_sl * atr_val
        stop_price = round(entry - risk_per_share, 2)
        take_profit = round(entry + tp_r * risk_per_share, 2)

        # Min-tick guard to avoid 422 "take_profit.limit_price >= base_price + 0.01"
        if take_profit <= entry + tick_size - 1e-9:
            take_profit = round(entry + tick_size, 2)

        if not (stop_price < entry < take_profit):
            print(f"[order] skip {sym}: bad TP/SL (entry={entry}, sl={stop_price}, tp={take_profit})", flush=True)
            continue

        # Size
        qty = max(1, min(base_qty, max_shares))
        qty = _cap_qty_by_notional(qty, entry, max_notional)
        if qty < 1:
            print(f"[order] skip {sym}: qty<1 after notional cap (entry={entry}, max_notional={max_notional})", flush=True)
            continue

        if args.dry_run:
            print(f'[order] DRY-RUN: would submit BRACKET {sym} qty={qty} entry≈{entry:.2f} '
                  f'sl={stop_price:.2f} tp={take_profit:.2f} (ATR={atr_val:.3f})', flush=True)
        else:
            try:
                resp = submit_bracket_order(
                    broker["base_url"], broker["key"], broker["secret"],
                    symbol=sym, qty=qty, side="buy",
                    limit_price=None,  # market entry
                    take_profit_price=take_profit,
                    stop_loss_price=stop_price,
                    tif="day",
                )
                # Even if 422 returns JSON error, we print it here for visibility
                print(f"[order] BRACKET submitted for {sym}: {json.dumps(resp)[:300]}", flush=True)
            except Exception as e:
                print(f"[order] ERROR submitting bracket for {sym}: {e}", flush=True)

    try:
        log.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())