# === EMAMerged/scripts/live_paper_loop.py ===
from __future__ import annotations
import os, sys, json, time, argparse
import datetime as dt
from typing import Dict, List, Any, Optional

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.filters import attach_verifiers, explain_long_gate, long_ok
from EMAMerged.src.data import (
    fetch_latest_bars,     # expects (symbols, timeframe, history_days, feed) → dict[sym]->DataFrame
    alpaca_market_open,    # (base_url, key, secret) → bool
)
from EMAMerged.src.oco import ensure_oco_for_long

ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

# ── UTC helpers (fix deprecation) ─────────────────────────────────────────────
# Python 3.12+ deprecates datetime.utcnow(). Use timezone-aware UTC instead.
def now_iso_utc() -> str:
    return dt.datetime.now(dt.UTC).strftime(ISO_UTC)

def _utc_stamp(fmt: str) -> str:
    return dt.datetime.now(dt.UTC).strftime(fmt)

def _build_cid(symbol: str) -> str:
    # unchanged format, just timezone-aware
    return f"EMA_{symbol}_{_utc_stamp('%Y%m%d_%H%M%S')}"

def _log_signal(log: TradeLogger, row: Dict[str, Any]) -> None:
    log.signal(**row)

def _log_gate(log: TradeLogger, decision: str, reasons: List[str], symbol: str, session: str, cid: str) -> None:
    log.gate(symbol=symbol, session=session, cid=cid, decision=decision, reasons=reasons)

# ── main ─────────────────────────────────────────────────────────────────────
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

    # broker creds
    broker = dict(cfg.get("broker", {}))
    key = broker.get("key") or os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY", "")
    secret = broker.get("secret") or os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET", "")
    base_url = broker.get("base_url", "https://paper-api.alpaca.markets")

    # tickers
    symbols = read_tickers(args.tickers)

    # Heartbeat
    session = "AM" if new_york_now().hour < 12 else "PM"
    market_open = alpaca_market_open(base_url, key, secret)
    hb = {"session": session, "market_open": bool(market_open), "type": "HEARTBEAT", "ts": now_iso_utc()}
    print(json.dumps(hb, separators=(",", ":"), ensure_ascii=False), flush=True)

    if not market_open and not args.force_run:
        return 0

    # Logger (date dir stamp also fixed to timezone-aware)
    results_dir = cfg.get("results_dir", "results")
    dstr = _utc_stamp("%Y%m%d")
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, f"live_{dstr}.jsonl")
    log = TradeLogger(log_path)

    # Pull bars
    bars_map = fetch_latest_bars(
        symbols,
        timeframe=timeframe,
        history_days=history_days,
        feed=feed,
        rth_only=rth_only,
        tz_name=cfg.get("timezone", "US/Eastern"),
        rth_start=cfg.get("rth_start", "09:30"),
        rth_end=cfg.get("rth_end", "15:55"),
        allowed_windows=cfg.get("entry_windows"),   # ok if None
        bar_limit=int(cfg.get("bar_limit", 10000)),
        key=key,
        secret=secret,
    )

    # Iterate symbols
    fcfg = dict(cfg.get("filters", {}))
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty or len(df) < 3:
            continue

        # Ensure indicators & verifiers are present; this now also creates EMA cols & slope
        df = attach_verifiers(df, cfg)

        last = df.iloc[-1]

        # Simple cross flag (uses EMA columns now present)
        try:
            cross = 1 if float(last.get("ema_fast", 0.0)) > float(last.get("ema_slow", 0.0)) else 0
        except Exception:
            cross = 0

        # Build and log signal snapshot
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
        }
        if "rsi" in df.columns:
            row["rsi"] = float(last.get("rsi", 50.0))

        _log_signal(log, row)

        # Gate decision (reasons visible via explain_long_gate)
        ok, reasons = explain_long_gate(last, cfg)
        _log_gate(log, "ALLOW" if ok else "BLOCK", reasons, sym, session, row["cid"])

        # (Order placement & OCO stays as-is in your existing code base; keeping minimal churn.)
        # If you want to place orders here, your existing execution code can be invoked as before.

    # Clean exit
    try:
        log.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
