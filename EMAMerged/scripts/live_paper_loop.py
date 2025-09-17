from __future__ import annotations
import os, sys, argparse
import pandas as pd
from datetime import datetime, timedelta
import pytz

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.data import (
    get_alpaca_bars, filter_rth, drop_unclosed_last_bar, alpaca_market_open,
    get_positions, get_open_orders, submit_market_order, submit_bracket_order,
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok

# -------------------------
# Env helpers / overrides
# -------------------------
def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default

def _env_int(name: str, default: int) -> int:
    try:
        s = _env(name, "")
        return int(s) if s.strip() else default
    except:
        return default

def _env_float(name: str, default: float) -> float:
    try:
        s = _env(name, "")
        return float(s) if s.strip() else default
    except:
        return default

def _alpaca_creds():
    key = _env("ALPACA_API_KEY") or _env("ALPACA_KEY")
    secret = _env("ALPACA_API_SECRET") or _env("ALPACA_SECRET")
    base = _env("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")
    return base, key, secret

def _results_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _append_csv(path: str, row: dict):
    df = pd.DataFrame([row])
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)

# -------------------------
# Main per-symbol routine
# -------------------------
def manage_symbol(sym: str, cfg: dict, args) -> None:
    base, key, secret = _alpaca_creds()
    tz = cfg.get("timezone", "US/Eastern")
    tf = cfg.get("timeframe", "5Min")
    feed = cfg.get("feed", "iex")
    hist_days = int(cfg.get("history_days", 30))
    limit = int(cfg.get("bar_limit", 500))

    # Fetch bars
    try:
        df = get_alpaca_bars(key, secret, sym, timeframe=tf, history_days=hist_days, bar_limit=limit, feed=feed)
    except Exception as e:
        print(f"[{sym}] data error: {e}")
        return

    if df.empty:
        print(f"[{sym}] no bars")
        return

    # RTH + remove last unclosed
    if bool(cfg.get("rth_only", True)):
        df = filter_rth(df, tz_name=tz, rth_start=cfg.get("rth_start", "09:30"),
                        rth_end=cfg.get("rth_end", "15:55"), allowed_windows=cfg.get("entry_windows"))
    df = drop_unclosed_last_bar(df, timeframe=tf)
    if len(df) < 3:
        print(f"[{sym}] not enough bars after filtering")
        return

    # Indicators & signal
    df = compute_indicators(df, cfg)
    x = crossover(df)
    close = float(df["close"].iat[-1])
    i = len(df) - 1
    ts_ny = new_york_now().strftime("%Y-%m-%d %H:%M")
    print(f"[{sym}] {ts_ny} close={round2(close)} cross={x}")

    # Positions / open orders
    try:
        positions = get_positions(base, key, secret)
    except Exception as e:
        print(f"[{sym}] positions error: {e}")
        return
    try:
        open_orders = get_open_orders(base, key, secret, symbol=sym)
    except Exception as e:
        print(f"[{sym}] open-orders error: {e}")
        open_orders = []

    qty_open = int(float(positions.get(sym, {}).get("qty", "0") or 0))
    side_open = "long" if qty_open > 0 else ("short" if qty_open < 0 else None)

    # Near close guard: no NEW entries within last N minutes
    if qty_open == 0 and x == 1 and cfg.get("no_new_entries_last_min", 0):
        try:
            et_tz = pytz.timezone(tz)
            now_et = new_york_now().astimezone(et_tz)
            rth_end = datetime.strptime(cfg.get("rth_end","15:55"), "%H:%M").time()
            rth_end_dt = now_et.replace(hour=rth_end.hour, minute=rth_end.minute, second=0, microsecond=0)
            remaining = (rth_end_dt - now_et).total_seconds() / 60.0
            if remaining <= float(cfg["no_new_entries_last_min"]):
                print(f"[{sym}] blocked: within last {cfg['no_new_entries_last_min']} min before close")
                x = 0
        except Exception:
            pass

    results_dir = cfg.get("results_dir","results")
    _results_dir(results_dir)

    # =========================
    # ENTRY
    # =========================
    if qty_open == 0 and x == 1:
        # Apply overrides (optional): FILTER_ADX, FILTER_RSI_MIN/MAX, FILTER_SLOPE_PCT
        overrides = {}
        if os.environ.get("FILTER_ADX"):
            overrides["adx_threshold"] = _env_float("FILTER_ADX", 22.0)
        if os.environ.get("FILTER_RSI_MIN"):
            overrides["rsi_min"] = _env_float("FILTER_RSI_MIN", 50.0)
        if os.environ.get("FILTER_RSI_MAX"):
            overrides["rsi_max"] = _env_float("FILTER_RSI_MAX", 80.0)
        if os.environ.get("FILTER_SLOPE_PCT"):
            overrides["slope_threshold_pct"] = _env_float("FILTER_SLOPE_PCT", 0.0012)

        # Evaluate gate
        gated_cfg = dict(cfg)
        if overrides:
            f = dict(cfg.get("filters", {}))
            f.update(overrides)
            gated_cfg["filters"] = f

        if not long_ok(df.iloc[i], gated_cfg, ema_fast_col="ema_fast", ema_slow_col="ema_slow"):
            print(f"[{sym}] long gate BLOCKED (overrides={overrides or 'none'})")
            return

        # qty=1 and notional cap
        qty = int(cfg.get("qty", 1)) or 1
        max_notional = float(cfg.get("max_notional_per_trade", 0) or 0.0)
        if max_notional > 0 and close > max_notional:
            print(f"[{sym}] skip: price {round2(close)} exceeds notional cap {round2(max_notional)}")
            return

        # Brackets?
        brackets_cfg = cfg.get("brackets", {})
        use_brackets = bool(brackets_cfg.get("enabled", True))

        # Compute SL/TP using ATR R multiple
        atr_mult = float(brackets_cfg.get("atr_mult_sl", 1.2))
        take_r  = float(brackets_cfg.get("take_profit_r", 1.0))
        atr_val = float(df["atr"].iat[-1]) if "atr" in df.columns else 0.0
        r_dist  = atr_mult * atr_val
        stop_price = max(0.01, close - r_dist)  # long stop below
        tp_price   = close + take_r * r_dist

        # idempotent COID
        coid = f"EMA-ENTRY-{sym}-{df.index[-1].strftime('%Y%m%d%H%M')}"

        if args.dry_run:
            print(f"[{sym}] DRY ENTRY qty=1 price~{round2(close)} stop~{round2(stop_price)} tp~{round2(tp_price)}")
        else:
            try:
                if use_brackets:
                    submit_bracket_order(
                        base, key, secret,
                        symbol=sym, qty=qty, side="buy",
                        entry_type="market", client_order_id=coid,
                        take_profit_price=tp_price, stop_price=stop_price, stop_limit_price=None
                    )
                else:
                    submit_market_order(base, key, secret, sym, qty, "buy", coid)
                print(f"[{sym}] ENTRY BUY qty={qty} @ ~{round2(close)} (brackets={use_brackets})")
                _append_csv(os.path.join(results_dir,"live_orders.csv"), {
                    "ts": df.index[-1].isoformat(),
                    "symbol": sym, "side": "buy", "qty": qty, "price": round2(close),
                    "stop": round2(stop_price), "take_profit": round2(tp_price),
                    "coid": coid, "brackets": int(use_brackets)
                })
            except Exception as e:
                print(f"[{sym}] entry error: {e}")

    # =========================
    # EXIT (auto, safe & idempotent)
    # =========================
    if qty_open > 0 and x == -1:
        # If there is already an open SELL/exit order, do nothing (idempotence)
        has_exit_open = any(o.get("symbol")==sym and o.get("side")=="sell" for o in open_orders)
        if has_exit_open:
            print(f"[{sym}] exit already pending; skip duplicate")
            return
        coid = f"EMA-EXIT-{sym}-{df.index[-1].strftime('%Y%m%d%H%M')}"
        if args.dry_run:
            print(f"[{sym}] DRY EXIT SELL qty={qty_open} @ ~{round2(close)}")
        else:
            try:
                submit_market_order(base, key, secret, sym, qty_open, "sell", coid)
                print(f"[{sym}] EXIT SELL qty={qty_open} @ ~{round2(close)}")
                _append_csv(os.path.join(results_dir,"live_orders.csv"), {
                    "ts": df.index[-1].isoformat(),
                    "symbol": sym, "side": "sell", "qty": qty_open, "price": round2(close),
                    "coid": coid, "reason": "ema_cross_down"
                })
            except Exception as e:
                print(f"[{sym}] exit error: {e}")

    # Snapshot positions for audit
    try:
        positions = get_positions(base, key, secret)
        pos = positions.get(sym)
        row = {
            "ts": df.index[-1].isoformat(),
            "symbol": sym,
            "qty": int(float(pos["qty"])) if pos else 0,
            "avg_entry_price": float(pos["avg_entry_price"]) if pos else 0.0,
            "market_price": round2(close),
        }
        _append_csv(os.path.join(results_dir,"live_positions.csv"), row)
    except Exception as e:
        print(f"[{sym}] positions snapshot error: {e}")

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="EMAMerged/config.yaml")
    ap.add_argument("--tickers", default="EMAMerged/tickers.txt")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--dry-run", type=int, default=int(os.environ.get("DRY_RUN", "0")))
    ap.add_argument("--force-run", type=int, default=int(os.environ.get("FORCE_RUN", "0")))
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Market-hours gate unless forced
    base, key, secret = _alpaca_creds()
    if not args.force_run:
        try:
            if not alpaca_market_open(base, key, secret):
                print("[GATE] Market closed; exiting.")
                return
        except Exception as e:
            print(f"[GATE] clock error: {e}")
            return

    # Load symbols
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)

    # Per-symbol isolation
    for sym in symbols:
        try:
            manage_symbol(sym, cfg, args)
        except Exception as e:
            print(f"[{sym}] fatal: {e}")

if __name__ == "__main__":
    main()
