#!/usr/bin/env python3
from __future__ import annotations
import os, sys, argparse
import pandas as pd

# --- Add repo root to sys.path so "EMAMerged" imports resolve when run as a script ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from EMAMerged.src.utils import load_config, read_tickers
from EMAMerged.src.data import (
    get_alpaca_bars, filter_rth, drop_unclosed_last_bar,
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok


def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default


def _alpaca_creds():
    key = _env("ALPACA_API_KEY") or _env("ALPACA_KEY")
    secret = _env("ALPACA_API_SECRET") or _env("ALPACA_SECRET")
    base = (_env("APCA_API_BASE_URL") or _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")
    return base, key, secret


def load_bars_for_symbol(
    sym: str,
    cfg: dict,
    days: int,
    timeframe_override: str | None = None,
    limit_override: int | None = None,
    rth_only_override: bool | None = None,
) -> pd.DataFrame:
    """
    Fetch bars from Alpaca, optionally overriding timeframe/limit and enforcing RTH.
    """
    # creds are read only to keep parity with live; get_alpaca_bars uses key/secret
    base, key, secret = _alpaca_creds()
    tf = timeframe_override or cfg.get("timeframe", "5Min")
    limit = int(limit_override or cfg.get("bar_limit", 500))
    feed = cfg.get("feed", "iex")

    df = get_alpaca_bars(
        key, secret, sym,
        timeframe=tf,
        history_days=days,
        bar_limit=limit,
        feed=feed,
    )
    if df.empty:
        return df

    # RTH filter (force on by default for backtests)
    if rth_only_override is None:
        rth_flag = bool(cfg.get("rth_only", True))
    else:
        rth_flag = bool(rth_only_override)

    if rth_flag:
        tz = cfg.get("timezone", "US/Eastern")
        df = filter_rth(
            df,
            tz_name=tz,
            rth_start=cfg.get("rth_start", "09:30"),
            rth_end=cfg.get("rth_end", "15:55"),
            allowed_windows=cfg.get("entry_windows"),  # respect windows if present
        )

    # Drop currently-forming last bar to match live logic
    df = drop_unclosed_last_bar(df, timeframe=tf)
    return df


def simulate_long_only(df: pd.DataFrame, cfg: dict, start_cash=10_000.0):
    """
    Fixed-quantity (qty=1) long-only backtest:
    - Entry: previous bar signals cross==1 AND long_ok(prev, cfg) → enter at next bar OPEN with qty=1.
    - Exit:  cross==-1 on previous bar → exit next OPEN
             plus bracket-style ATR SL/TP if present in cfg['brackets'].
    - No risk sizing; quantity is always 1 share.
    """
    if df is None or df.empty or len(df) < 30:
        return dict(trades=0, gross=0.0, net=0.0, win_rate=0.0, ret_pct=0.0, equity=start_cash)

    # Build indicators exactly like live (adds ema_fast/slow, atr, rsi, adx, etc.)
    df = compute_indicators(df, cfg).copy()

    # Ensure ATR exists for bracket distances
    if "atr" not in df.columns:
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs()
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    equity = start_cash
    in_pos = False
    qty = 0
    entry_px = 0.0

    wins = 0
    losses = 0
    gross = 0.0

    # Bracket/exit params (same meaning as live config)
    bcfg = cfg.get("brackets", {})
    atr_mult_sl = float(bcfg.get("atr_mult_sl", 1.2))
    take_profit_r = float(bcfg.get("take_profit_r", 1.0))

    trades_printed = []  # (entry_ts, entry_px, exit_ts, exit_px, pnl)

    # We will use ATR as the "R" base for brackets
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        open_px = float(cur["open"])

        if not in_pos:
            # Signal formed on prev bar close → act at next open with qty=1
            x = crossover(df.iloc[:i])  # history up to prev bar
            if x == 1 and long_ok(prev, cfg, ema_fast_col="ema_fast", ema_slow_col="ema_slow"):
                qty = 1
                entry_px = open_px
                in_pos = True
                entry_ts = cur.name  # timestamp of the bar we enter at (next bar open)
        else:
            # Default exit on cross down (prev bar)
            x = crossover(df.iloc[:i])
            exit_now_on_cross = (x == -1)

            # Bracket-style exits on current bar range using ATR as R
            atr_val = float(prev.get("atr", 0.0)) if "atr" in df.columns else 0.0
            stop_px = max(0.01, entry_px - atr_mult_sl * atr_val) if atr_val > 0 else None
            tp_px   = (entry_px + take_profit_r * atr_val) if atr_val > 0 else None

            # Check SL/TP intrabar
            hit_sl = (stop_px is not None) and (cur["low"]  <= stop_px)
            hit_tp = (tp_px   is not None) and (cur["high"] >= tp_px)

            if hit_sl:
                exit_px = stop_px
                pnl = (exit_px - entry_px) * qty
                equity += pnl; gross += pnl
                losses += 1 if pnl <= 0 else 0
                in_pos = False; qty = 0
                trades_printed.append((entry_ts, entry_px, cur.name, exit_px, pnl))
                continue

            if hit_tp:
                exit_px = tp_px
                pnl = (exit_px - entry_px) * qty
                equity += pnl; gross += pnl
                wins += 1 if pnl > 0 else 0
                in_pos = False; qty = 0
                trades_printed.append((entry_ts, entry_px, cur.name, exit_px, pnl))
                continue

            if exit_now_on_cross:
                exit_px = open_px
                pnl = (exit_px - entry_px) * qty
                equity += pnl; gross += pnl
                if pnl > 0: wins += 1
                else:       losses += 1
                in_pos = False; qty = 0
                trades_printed.append((entry_ts, entry_px, cur.name, exit_px, pnl))

    # Print the collected trades for this symbol (caller prints header)
    for (ent_ts, ent_px, ex_ts, ex_px, pnl) in trades_printed:
        # pandas index is tz-aware; print as ET-like string (kept as .isoformat for clarity)
        print(f"    ENTRY {ent_ts} @ {ent_px:.2f}  →  EXIT {ex_ts} @ {ex_px:.2f}  qty=1  PnL={pnl:.2f}")

    trades = wins + losses
    ret_pct = (equity / start_cash - 1.0) * 100.0
    win_rate = (wins / trades * 100.0) if trades > 0 else 0.0
    return dict(trades=trades, gross=round(gross, 2), net=round(gross, 2),
                win_rate=round(win_rate, 1), ret_pct=round(ret_pct, 2), equity=round(equity, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="EMAMerged/config.yaml")
    ap.add_argument("--tickers", default="EMAMerged/tickers.txt")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cash", type=float, default=10_000)
    # NOTE: risk argument kept for backward compatibility but IGNORED (qty is fixed to 1)
    ap.add_argument("--risk", type=float, default=0.01, help="Ignored: backtest uses fixed qty=1")
    # Backtest-only overrides:
    ap.add_argument("--timeframe", default=None, help="Override timeframe for backtest (e.g., 15Min)")
    ap.add_argument("--limit", type=int, default=None, help="Override bar_limit for backtest (e.g., 10000)")
    ap.add_argument("--rth-only", dest="rth_only", action="store_true", help="Force RTH-only (default)")
    ap.add_argument("--no-rth-only", dest="rth_only", action="store_false", help="Allow non-RTH bars")
    ap.set_defaults(rth_only=True)  # default = RTH only, per your request
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Resolve symbols
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)

    # For diagnostics, read current ADX threshold from cfg (default 25.0)
    fcfg = dict(cfg.get("filters", {}))
    adx_th = float(fcfg.get("adx_threshold", 25.0))

    rows = []
    for sym in symbols:
        try:
            df = load_bars_for_symbol(
                sym, cfg, args.days,
                timeframe_override=args.timeframe,
                limit_override=args.limit,
                rth_only_override=args.rth_only,
            )
        except Exception as e:
            print(f"[{sym}] data error: {e}")
            continue

        # --- Diagnostics per symbol ---
        if df.empty:
            print(f"[{sym}] bars=0")
            continue

        di = compute_indicators(df, cfg).copy()
        bars = len(di)

        # Count cross-ups over the series
        crosses_up = 0
        for i in range(1, len(di)):
            if crossover(di.iloc[:i+1]) == 1:
                crosses_up += 1

        pct_adx = (di["adx"] >= adx_th).mean() * 100.0 if "adx" in di.columns else 0.0
        print(f"[{sym}] bars={bars}, crosses_up={crosses_up}, %ADX>={adx_th:.0f}={pct_adx:.1f}%")

        # --- Backtest simulation (qty=1 model) ---
        # Print header if any trade gets printed inside simulate_long_only
        print(f"  TRADES {sym}:")
        res = simulate_long_only(df, cfg, start_cash=args.cash)
        rows.append({"symbol": sym, **res})

    if not rows:
        print("No results.")
        return

    out = pd.DataFrame(rows).sort_values("net", ascending=False)
    print("\n=== Backtest Summary (last {} days) ===".format(args.days))
    print(out.to_string(index=False))

    total_trades = int(out["trades"].sum())
    total_net = round(float(out["net"].sum()), 2)
    avg_winrate = round((out["trades"] * out["win_rate"] / 100.0).sum() / max(total_trades, 1) * 100.0, 1)
    avg_ret = round(float(out["ret_pct"].mean()), 2)
    total_equity = round(float(out["equity"].sum()), 2)

    print("\nTOTAL  trades={}  net={}  win_rate={}%%  avg_ret={}%%  equity_sum={}".format(
        total_trades, total_net, avg_winrate, avg_ret, total_equity
    ))


if __name__ == "__main__":
    main()