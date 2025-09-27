from __future__ import annotations
import os, sys, argparse
import pandas as pd
from datetime import datetime, timedelta
import pytz

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.data import (
    get_alpaca_bars, filter_rth, drop_unclosed_last_bar, alpaca_market_open,
    get_positions, get_open_orders, submit_market_order, submit_bracket_order,
    list_open_orders, patch_order
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok, explain_long_gate

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

_ERROR_COOLDOWN = {}  # sym -> datetime until retries allowed on 422

# -------------------------
# Optional: breakeven bump helper
# -------------------------
def _maybe_breakeven_bump(sym: str, cfg: dict, entry_price: float, r_dist: float, last_price: float):
    be = cfg.get("breakeven", {}) or {}
    if not be.get("enabled", True):
        return
    if r_dist <= 0:
        return

    trigger_r = float(be.get("trigger_r", 0.5))
    tick = float(be.get("tick_size", 0.01))
    bump_ticks = int(be.get("bump_ticks", 2))
    bump_price = round2(entry_price + bump_ticks * tick)

    r_mult = (last_price - entry_price) / r_dist
    if r_mult < trigger_r:
        return

    base, key, secret = _alpaca_creds()
    # find open stop for this symbol; nested open orders lists children
    try:
        orders = list_open_orders(base, key, secret, symbols=[sym])
    except Exception as e:
        print(f"[{sym}] breakeven list orders error: {e}")
        return

    stop_os = [o for o in orders
               if o.get("symbol") == sym
               and o.get("side") == "sell"
               and (o.get("type") in ("stop", "stop_limit"))]
    if not stop_os:
        return

    stop_o = stop_os[0]
    cur_stop = float(stop_o.get("stop_price") or 0.0)
    if bump_price <= cur_stop:
        return

    try:
        patch_order(base, key, secret, stop_o["id"], stop_price=bump_price)
        print(f"[{sym}] breakeven bump: stop {cur_stop} → {bump_price} (r={r_mult:.2f})", flush=True)
    except Exception as e:
        print(f"[{sym}] breakeven bump error: {e}", flush=True)

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
        end = new_york_now()
        start = end - timedelta(days=hist_days)
        df = get_alpaca_bars(key, secret, tf, sym, start, end, feed=feed, limit=limit)
        if bool(cfg.get("rth_only", True)):
            df = filter_rth(df, tz_name=tz, rth_start=cfg.get("rth_start","09:30"), rth_end=cfg.get("rth_end","15:55"))
        df = drop_unclosed_last_bar(df, tf)
        if df.empty:
            print(f"[{sym}] no bars")
            return
    except Exception as e:
        print(f"[{sym}] bars error: {e}")
        return

    df = compute_indicators(df, cfg)
    x = crossover(df)
    close = float(df["close"].iat[-1])

    # C) Log the last CLOSED bar time (bar timestamp), not wall-clock
    bar_ts = df.index[-1].tz_convert(pytz.timezone(tz)).strftime("%Y-%m-%d %H:%M")
    print(f"[{sym}] {bar_ts} close={round2(close)} cross={x}")

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

    # A) Late entry cutoff: no NEW entries within last N minutes
    if qty_open == 0 and x == 1 and cfg.get("entry_cutoff_min", 0):
        try:
            et_tz = pytz.timezone(tz)
            now_et = new_york_now().astimezone(et_tz)
            rth_end = datetime.strptime(cfg.get("rth_end","15:55"), "%H:%M").time()
            rth_end_dt = now_et.replace(hour=rth_end.hour, minute=rth_end.minute, second=0, microsecond=0)
            remaining = (rth_end_dt - now_et).total_seconds() / 60.0
            cutoff = int(cfg.get("entry_cutoff_min", 20))
            if remaining <= cutoff:
                print(f"[{sym}] skip entry: {remaining:.0f}m to close ≤ cutoff {cutoff}m")
                return
        except Exception as e:
            print(f"[{sym}] cutoff check error: {e}")

    # Long entry path
    if qty_open == 0 and x == 1 and long_ok(df, cfg):
        reasons = explain_long_gate(df, cfg)
        if reasons:
            print(f"[{sym}] gate reasons: {reasons}")

        qty = int(cfg.get("qty", 1) or 1)
        max_notional = float(cfg.get("max_notional_per_trade", 0) or 0.0)
        if max_notional > 0 and close > max_notional:
            print(f"[{sym}] skip: price {round2(close)} exceeds notional cap {round2(max_notional)}")
            return

        # Brackets?
        brackets_cfg = cfg.get("brackets", {})
        use_brackets = bool(brackets_cfg.get("enabled", True))

        # Compute SL/TP using ATR R multiple — B) harden spacing
        atr_mult = float(brackets_cfg.get("atr_mult_sl", 1.2))
        take_r  = float(brackets_cfg.get("take_profit_r", 1.0))
        atr_val = float(df["atr"].iat[-1]) if "atr" in df.columns else 0.0
        r_dist  = max(0.02, atr_mult * atr_val)  # ≥ $0.02 (two ticks) risk
        stop_price = max(0.01, close - r_dist)
        tp_price   = close + take_r * r_dist
        if tp_price <= close + 0.01:
            tp_price = close + 0.02

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
                    submit_market_order(base, key, secret, sym, qty=qty, side="buy", client_order_id=coid)
                print(f"[{sym}] ENTRY BUY qty={qty} @ ~{round2(close)} (brackets={use_brackets})")
            except Exception as e:
                msg = str(e)
                print(f"[{sym}] entry error: {msg}")
                # B) 422 cooldown (15m)
                if "422" in msg or "Unprocessable" in msg:
                    _ERROR_COOLDOWN[sym] = new_york_now() + timedelta(minutes=15)
                return

    # If we have a position, optionally try breakeven bump
    if qty_open > 0:
        try:
            pos = positions.get(sym)
            entry_price = float(pos["avg_entry_price"]) if pos else 0.0
            # approximate r_dist using the same ATR logic used above
            atr_val = float(df["atr"].iat[-1]) if "atr" in df.columns else 0.0
            atr_mult = float(cfg.get("brackets", {}).get("atr_mult_sl", 1.2))
            r_dist  = max(0.02, atr_mult * atr_val)
            _maybe_breakeven_bump(sym, cfg, entry_price, r_dist, last_price=close)
        except Exception as e:
            print(f"[{sym}] breakeven check error: {e}")

    # Snapshot
    try:
        results_dir = cfg.get("results_dir", "results")
        _results_dir(results_dir)
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
            from EMAMerged.src.data import alpaca_market_open
            if not alpaca_market_open(base, key, secret):
                print("[GATE] Market closed; exiting.")
                return
        except Exception as e:
            print(f"[GATE] clock error: {e}")
            return

    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)

    for sym in symbols:
        try:
            # B) per-symbol cooldown check
            until = _ERROR_COOLDOWN.get(sym)
            if until and new_york_now() < until:
                print(f"[{sym}] on error cooldown until {until.strftime('%H:%M')} (skip)")
                continue
            manage_symbol(sym, cfg, args)
        except Exception as e:
            print(f"[{sym}] fatal: {e}")

if __name__ == "__main__":
    main()
