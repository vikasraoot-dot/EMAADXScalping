from __future__ import annotations
import os, sys
import argparse
import requests
import pandas as pd

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.data import get_alpaca_bars, filter_rth, drop_unclosed_last_bar, alpaca_market_open
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok

ALPACA_ORDERS = "/v2/orders"
MAX_ORDER_NOTIONAL_ENV = "MAX_ORDER_NOTIONAL"

def get_api_base() -> str:
    return os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")

def _hdrs():
    key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY") or ""
    sec = os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_SECRET") or ""
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

def get_positions() -> dict[str, dict]:
    url = get_api_base() + "/v2/positions"
    r = requests.get(url, headers=_hdrs(), timeout=15)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    out = {}
    for p in r.json():
        out[p["symbol"]] = p
    return out

def get_position(sym: str):
    pos = get_positions().get(sym)
    if not pos:
        return 0, None, 0.0
    qty = int(float(pos.get("qty", "0")))
    side = "long" if qty > 0 else "short"
    avg = float(pos.get("avg_entry_price", 0.0))
    return qty, side, avg

def submit_market_order(sym: str, qty: int, side: str, client_order_id: str):
    url = get_api_base() + ALPACA_ORDERS
    payload = {
        "symbol": sym,
        "qty": qty,
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "client_order_id": client_order_id,
    }
    r = requests.post(url, headers=_hdrs(), json=payload, timeout=15)
    r.raise_for_status()
    return r.json()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="EMAMerged/config.yaml")
    ap.add_argument("--tickers", default="EMAMerged/tickers.txt")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--dry-run", type=int, default=int(os.environ.get("DRY_RUN", "1")))
    ap.add_argument("--force-run", type=int, default=int(os.environ.get("FORCE_RUN", "0")))
    args = ap.parse_args()

    cfg = load_config(args.config)
    tf = cfg.get("timeframe", "5Min")
    feed = cfg.get("feed", "iex")
    hist_days = int(cfg.get("history_days", 30))
    limit = int(cfg.get("bar_limit", 500))
    tz = cfg.get("timezone", "US/Eastern")

    # Market hours gate
    if not args.force_run:
        if not alpaca_market_open(get_api_base(), _hdrs()["APCA-API-KEY-ID"], _hdrs()["APCA-API-SECRET-KEY"]):
            print("[GATE] Market closed; exiting fast.")
            return

    symbols = []
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)

    any_action = False
    for sym in symbols:
        # 1) Pull bars
        df = get_alpaca_bars(
            key=_hdrs()["APCA-API-KEY-ID"],
            secret=_hdrs()["APCA-API-SECRET-KEY"],
            symbol=sym,
            timeframe=tf,
            history_days=hist_days,
            bar_limit=limit,
            feed=feed,
        )
        if df.empty:
            print(f"[{sym}] no bars")
            continue

        # 2) RTH & remove last unclosed
        if bool(cfg.get("rth_only", True)):
            df = filter_rth(df, tz_name=tz, rth_start=cfg.get("rth_start", "09:30"),
                            rth_end=cfg.get("rth_end", "15:55"), allowed_windows=cfg.get("entry_windows"))
        df = drop_unclosed_last_bar(df, timeframe=tf)
        if len(df) < 3:
            print(f"[{sym}] not enough bars after filtering")
            continue

        # 3) Indicators
        df = compute_indicators(df, cfg)

        # 4) Signal
        x = crossover(df)
        close = float(df["close"].iat[-1])
        i = len(df) - 1
        print(f"[{sym}] {new_york_now().strftime('%Y-%m-%d %H:%M')} close={round2(close)} cross={x}")

        qty_open, side_open, avg = get_position(sym)

        # ENTRY
        if qty_open == 0 and x == 1:
            row = df.iloc[i]
            if not long_ok(row, cfg, ema_fast_col="ema_fast", ema_slow_col="ema_slow"):
                print(f"[{sym}] long gate BLOCKED | rsi={round2(row.get('rsi',0))} "
                      f"adx={round2(row.get('adx',0))} +DI={round2(row.get('plus_di',0))} "
                      f"-DI={round2(row.get('minus_di',0))} "
                      f"slope9%={round2(100*row.get('ema_fast_slope_pct',0))}bp "
                      f"slope21%={round2(100*row.get('ema_slow_slope_pct',0))}bp")
                continue

            qty = int(cfg.get("qty", 1))
            max_notional_env = float(os.environ.get(MAX_ORDER_NOTIONAL_ENV, "0") or 0.0)
            max_notional_cfg = float(cfg.get("max_notional_per_trade", 0) or 0.0)
            max_notional = max(max_notional_cfg, max_notional_env)
            if max_notional > 0:
                qty = max(1, min(qty, int(max_notional // max(0.01, close))))

            order_id = f"EMA-{sym}-{df.index[-1].strftime('%Y%m%d%H%M')}"
            print(f"[{sym}] ENTER BUY qty={qty} @ ~{round2(close)} DRY={bool(args.dry_run)}")
            if not args.dry_run:
                try:
                    submit_market_order(sym, qty, "buy", order_id)
                    any_action = True
                except Exception as e:
                    print(f"[{sym}] order error: {e}")

        # EXIT (simple: cross-down)
        if qty_open > 0 and x == -1:
            print(f"[{sym}] EXIT signal (cross down). You can add order-close logic here if desired.")

    if not any_action:
        print("[DONE] No orders placed.")

if __name__ == "__main__":
    main()
