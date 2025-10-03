# === EMAMerged/scripts/live_paper_loop.py ===
from __future__ import annotations
import os, sys, json, argparse
import datetime as dt
from typing import Dict, List, Any, Optional

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.filters import attach_verifiers, explain_long_gate
from EMAMerged.src.data import (
    fetch_latest_bars,     # dict[symbol] -> DataFrame
    alpaca_market_open,    # (base_url, key, secret) -> bool
)
# Imported for future order placement (left unused here to minimize churn)
from EMAMerged.src.oco import ensure_oco_for_long  # noqa: F401

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

    # Broker + market status
    broker = _resolve_broker(cfg)
    market_open = alpaca_market_open(broker["base_url"], broker["key"], broker["secret"])

    # Tickers
    symbols = read_tickers(args.tickers)

    # Session heuristic + heartbeat
    session = "AM" if new_york_now().hour < 12 else "PM"
    hb = {"session": session, "market_open": bool(market_open), "type": "HEARTBEAT", "ts": now_iso_utc()}
    print(json.dumps(hb, separators=(",", ":"), ensure_ascii=False), flush=True)

    if not market_open and not args.force_run:
        return 0

    # Logger path (results/YYYYMMDD)
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    dstr = _utc_stamp("%Y%m%d")
    log_path = os.path.join(results_dir, f"live_{dstr}.jsonl")
    log = TradeLogger(log_path)

    # Pull bars (Option B) — wire window/min_periods from config.filters (with defaults)
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

        # Quick cross flag (purely informational here)
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
            # Note: dollar_vol_avg is computed upstream (data.py, Option B).
            # We do not gate on it unless `min_dollar_vol` is set in config.filters.
        }
        # Remove None for cleaner JSON
        row = {k: v for k, v in row.items() if v is not None}
        log.signal(**row)

        # Gate decision + reasons (purely config-driven)
        ok, reasons = explain_long_gate(last, cfg)
        log.gate(symbol=sym, session=session, cid=row["cid"], decision=("ALLOW" if ok else "BLOCK"), reasons=reasons)

        # If you later re-enable order placement, this is the branch to do it.
        # Kept intentionally out to minimize churn per your request.

    try:
        log.close()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
