# === EMAMerged/scripts/live_paper_loop.py ===
from __future__ import annotations
import os, sys, json, argparse
import datetime as dt
from typing import Dict, List, Any, Optional

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.filters import attach_verifiers, explain_long_gate
from EMAMerged.src.data import (
    fetch_latest_bars,      # dict[symbol] -> DataFrame (Option B dollar_vol computed before window trim)
    alpaca_market_open,     # (base_url, key, secret) -> bool
    cancel_all_orders,      # risk ops
    close_all_positions,    # risk ops
)

# If you later re-enable order placement / OCO, import here (kept unused to minimize churn)
# from EMAMerged.src.oco import ensure_oco_for_long  # noqa: F401

ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"


# ──────────────────────────────────────────────────────────────────────────────
# UTC helpers (no utcnow() deprecation warnings)
# ──────────────────────────────────────────────────────────────────────────────
def now_iso_utc() -> str:
    return dt.datetime.now(dt.UTC).strftime(ISO_UTC)


def _utc_stamp(fmt: str) -> str:
    return dt.datetime.now(dt.UTC).strftime(fmt)


def _build_cid(symbol: str) -> str:
    # Preserve prior format, just timezone-aware now
    return f"EMA_{symbol}_{_utc_stamp('%Y%m%d_%H%M%S')}"


# ──────────────────────────────────────────────────────────────────────────────
# Broker creds resolution (env first, then config)
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_broker(cfg: Dict) -> Dict[str, str]:
    bk = dict(cfg.get("broker", {}))
    base_url = os.getenv("APCA_BASE_URL", bk.get("base_url", "https://paper-api.alpaca.markets"))

    # Primary env names
    key_id = os.getenv("APCA_API_KEY_ID", bk.get("key") or "")
    secret = os.getenv("APCA_API_SECRET_KEY", bk.get("secret") or "")
    # Back-compat aliases
    key_id = os.getenv("ALPACA_KEY", key_id)
    secret = os.getenv("ALPACA_SECRET", secret)

    return {"base_url": base_url, "key": key_id, "secret": secret}


# ──────────────────────────────────────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────────────────────────────────────
def minutes_to_close_et(close_hm: tuple[int, int] = (16, 0)) -> int:
    """
    Minutes from 'now' (New York time) until today's regular close (default 16:00 ET).
    Negative if after close.
    """
    now_et = new_york_now()   # timezone-aware
    close_et = now_et.replace(hour=close_hm[0], minute=close_hm[1], second=0, microsecond=0)
    delta = close_et - now_et
    return int(delta.total_seconds() // 60)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--dry-run", type=int, default=1)
    ap.add_argument("--force-run", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Timeframe + history lookback (keep your existing keys; intervals/periods optional)
    timeframe = cfg.get("timeframe", "15Min")
    history_days = int(cfg.get("history_days", 10))
    feed = cfg.get("feed", "iex")
    rth_only = bool(cfg.get("rth_only", True))

    # Risk settings from config
    entry_cutoff_min = int(cfg.get("entry_cutoff_min", 0))  # block NEW entries within last N min
    flatten_min_before_close = int(cfg.get("flatten_minutes_before_close", 0))  # flatten N min before close

    # Broker + market status
    broker = _resolve_broker(cfg)
    market_open = alpaca_market_open(broker["base_url"], broker["key"], broker["secret"])

    # Tickers
    symbols = read_tickers(args.tickers)

    # Session heuristic + heartbeat
    session = "AM" if new_york_now().hour < 12 else "PM"
    hb = {"session": session, "market_open": bool(market_open), "type": "HEARTBEAT", "ts": now_iso_utc()}
    print(json.dumps(hb, separators=(",", ":"), ensure_ascii=False), flush=True)

    # Early exit if market closed and not forced
    if not market_open and not args.force_run:
        return 0

    # Logger path (results/YYYYMMDD)
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    dstr = _utc_stamp("%Y%m%d")
    log_path = os.path.join(results_dir, f"live_{dstr}.jsonl")
    log = TradeLogger(log_path)

    # ── Risk Ops: Flatten before close (idempotent; safe to call repeatedly) ──
    # Note: only run if market_open; dry-run respected.
    mins_to_close = minutes_to_close_et((16, 0))
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
        # After flatten window starts, we do not allow new entries
        entry_allowed = False
    else:
        # Entry cutoff near close (only blocks NEW entries; still logs signals/gates)
        entry_allowed = (mins_to_close > entry_cutoff_min) if entry_cutoff_min > 0 else True

    # Pull bars (Option B) — wire rolling params from config.filters (with defaults)
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

    # Iterate symbols → build signal & gate
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 3:
            continue

        # Ensure indicators/verifiers present (computes EMA_fast/slow if missing; slope, RSI, ADX)
        df = attach_verifiers(df, cfg)
        last = df.iloc[-1]

        # Quick cross flag (informational)
        try:
            cross = 1 if float(last.get("ema_fast", 0.0)) > float(last.get("ema_slow", 0.0)) else 0
        except Exception:
            cross = 0

        # Signal snapshot
        row = {
            "symbol": sym,
            "session": session,
            "cid": _build_cid(sym),
            "tf": timeframe,
            "cross": cross,
            "ref_bar_ts": str(df.index[-1]),
            "last_close": float(last.get("close", 0.0)),
            "adx": float(last.get("adx", 0.0)),
            "ema_slope_pct": float(last.get("ema_slope_pct", 0.0)),
            "rsi": float(last.get("rsi", 50.0)) if "rsi" in df.columns else None,
        }
        row = {k: v for k, v in row.items() if v is not None}
        log.signal(**row)

        # Base gate decision + reasons (config-driven inside explain_long_gate)
        ok, reasons = explain_long_gate(last, cfg)

        # Apply entry cutoff overlay (blocks NEW entries near close)
        if not entry_allowed:
            reasons = list(reasons) + [f"entry_cutoff_min: {max(mins_to_close, 0)} min to close"]
            ok = False

        # Log gate decision
        log.gate(symbol=sym, session=session, cid=row["cid"], decision=("ALLOW" if ok else "BLOCK"), reasons=reasons)

        # If you later re-enable order placement, handle ok==True here (and respect entry_allowed).
        # Keeping intentionally out to minimize churn.

    try:
        log.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
