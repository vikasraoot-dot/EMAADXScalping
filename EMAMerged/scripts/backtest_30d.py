#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
import pandas as pd
from collections import Counter

pd.set_option('future.no_silent_downcasting', True)

# Global counter across all symbols
FILTER_REJECTS = Counter()


# --- Ensure repo root on sys.path so "EMAMerged" imports resolve when run as a script ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from EMAMerged.src.utils import load_config, read_tickers
from EMAMerged.src.data import (
    fetch_latest_bars,
    filter_rth,
    drop_unclosed_last_bar,
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok


# ------------------------------------------------------------------
# Helpers for instrumentation and light MTF bias (minimal & optional)
# ------------------------------------------------------------------
def _reason_key(reason: str) -> str:
    """
    Normalize reason strings to compact keys for counting.
    Examples:
      "ADX 22.0 < 23.0"        -> "ADX"
      "RSI 86.2 > 80.0"        -> "RSI"
      "EMA_slope 0.0005 < ..." -> "EMA_slope"
      "DollarVol 4200000 < ..."-> "DollarVol"
      "MTF bias false"         -> "MTF"
      "Price 9.50 < 10.00"     -> "Price"
    """
    r = str(reason).strip()
    if r.startswith("MTF"):
        return "MTF"
    head = r.split(" ", 1)[0]
    return head


def _compute_htf_bias(df: pd.DataFrame, cfg: dict, timeframe: str = "60Min", shift_bars: int = 1) -> pd.Series:
    """
    Light multi-timeframe bias: HTF fast EMA > HTF slow EMA on the prior HTF bar.
    Returns a boolean Series aligned to df.index.
    """
    f = int(cfg.get("ema_fast", 9))
    s = int(cfg.get("ema_slow", 21))

    # Resample close to HTF and compute simple fast/slow EMAs there
    htf = df[["close"]].resample(timeframe).last()
    htf["ema_fast_htf"] = htf["close"].ewm(span=f, adjust=False, min_periods=f).mean()
    htf["ema_slow_htf"] = htf["close"].ewm(span=s, adjust=False, min_periods=s).mean()

    # Compare, shift, then make dtype explicit BEFORE fillna
    comp = (htf["ema_fast_htf"] > htf["ema_slow_htf"]).shift(shift_bars)
    bias = (
        comp.infer_objects(copy=False)   # <- key line to avoid "silent downcasting" path
            .fillna(False)
            .astype(bool)
    )

    # Map back to LTF index via ffill and keep dtype boolean
    return bias.reindex(df.index, method="ffill").fillna(False).astype(bool)
    
def _eval_long_reasons(prev: pd.Series, cfg: dict, mtf_bias_ok: bool) -> list:
    """
    Build a list of reasons why an entry is rejected.
    Mirrors filters.long_ok at a high level, but only for logging.
    """
    reasons = []
    fcfg = dict(cfg.get("filters", {}))

    # ADX threshold (you set 23 in config; we do not override it)
    adx_th = float(fcfg.get("adx_threshold", 25.0))
    adx_val = float(prev.get("adx", 0.0))
    if adx_val < adx_th:
        reasons.append("ADX {:.1f} < {:.1f}".format(adx_val, adx_th))

    # RSI band
    rsi_val = float(prev.get("rsi", 0.0))
    rsi_min = fcfg.get("rsi_min", None)
    rsi_max = fcfg.get("rsi_max", None)
    if rsi_min is not None and rsi_val < float(rsi_min):
        reasons.append("RSI {:.1f} < {:.1f}".format(rsi_val, float(rsi_min)))
    if rsi_max is not None and rsi_val > float(rsi_max):
        reasons.append("RSI {:.1f} > {:.1f}".format(rsi_val, float(rsi_max)))

    # EMA slope (if present in your pipeline)
    slope_th = fcfg.get("slope_threshold_pct", None)
    if slope_th is not None and "ema_slope_pct" in prev.index:
        slope_val = float(prev.get("ema_slope_pct", 0.0))
        if slope_val < float(slope_th):
            reasons.append("EMA_slope {:.5f} < {:.5f}".format(slope_val, float(slope_th)))

    # Price / Liquidity (if present)
    min_price = fcfg.get("min_price", None)
    if min_price is not None:
        close_val = float(prev.get("close", 0.0))
        if close_val < float(min_price):
            reasons.append("Price {:.2f} < {:.2f}".format(close_val, float(min_price)))

    min_dv = fcfg.get("min_dollar_vol", None)
    if min_dv is not None and "dollar_vol_avg" in prev.index:
        dv_val = float(prev.get("dollar_vol_avg", 0.0))
        if dv_val < float(min_dv):
            reasons.append("DollarVol {:.0f} < {:.0f}".format(dv_val, float(min_dv)))

    # MTF bias
    if not mtf_bias_ok:
        reasons.append("MTF bias false")

    return reasons


# ------------------------------------------------------------------
# Data loading helper (fixed to match src/data.get_alpaca_bars signature)
# ------------------------------------------------------------------

def load_bars_for_symbol(symbol: str, cfg: dict, days: int,
                         timeframe_override=None, limit_override=None, rth_only_override=True) -> pd.DataFrame:
    """
    Uses EMAMerged/src/data.fetch_latest_bars to get historical bars.
    """
    timeframe = timeframe_override or cfg.get("timeframe", "15Min")
    feed = cfg.get("feed", "iex")
    tz_name = cfg.get("timezone", "US/Eastern")
    rth_start = cfg.get("rth_start", "09:30")
    rth_end = cfg.get("rth_end", "15:55")

    # Pull creds from env
    key = os.getenv("APCA_API_KEY_ID", "")
    secret = os.getenv("APCA_API_SECRET_KEY", "")

    # history_days = CLI --days (ensures we fetch enough)
    history_days = int(days)

    # bar_limit: prefer CLI override if provided, else config bar_limit, else fallback
    bar_limit = int(limit_override) if (limit_override is not None) else int(cfg.get("bar_limit", 10000))

    # fetch_latest_bars returns (bars_map, metadata) tuple
    bars_map, _ = fetch_latest_bars(
        symbols=[symbol],
        timeframe=timeframe,
        history_days=history_days,
        feed=feed,
        rth_only=rth_only_override,
        tz_name=tz_name,
        rth_start=rth_start,
        rth_end=rth_end,
        bar_limit=bar_limit,
        key=key,
        secret=secret,
    )
    
    df = bars_map.get(symbol, pd.DataFrame())
    if df.empty:
        return df

    # Keep approx last N days at 15m cadence for consistency
    bars_per_day_15m = 390 // 15  # 26 bars
    approx_rows = int(days) * bars_per_day_15m
    return df.tail(approx_rows)


# ------------------------------------------------------------------
# Backtest simulator (minimal changes: risk sizing + mtf bias + reasons)
# ------------------------------------------------------------------

def simulate_long_only(df: pd.DataFrame, cfg: dict, start_cash=10_000.0, risk_pct: float = 0.01):
    """
    Long-only backtest with risk-based sizing (+$2,000 cap) and light MTF bias.
    - Entry: previous bar crossover==1 AND long_ok(prev,cfg).
             Additionally require 60m HTF bias (fast>slow on prior HTF bar), unless disabled in config.
    - Exit:  previous bar crossover==-1 -> exit next OPEN.
             Bracket ATR SL/TP unchanged; we do not alter your exit model.
    """
    if df is None or df.empty or len(df) < 30:
        return dict(trades=0, gross=0.0, net=0.0, win_rate=0.0, ret_pct=0.0, equity=start_cash)

    df = compute_indicators(df, cfg).copy()

    # Light MTF bias (reads config if present; defaults to enabled)
    mtf_cfg = dict(cfg.get("filters", {})).get("mtf_bias", {}) if isinstance(cfg.get("filters", {}), dict) else {}
    mtf_enabled = bool(mtf_cfg.get("enabled", True))
    df["htf_bias"] = _compute_htf_bias(
        df, cfg,
        timeframe=mtf_cfg.get("timeframe", "60Min"),
        shift_bars=int(mtf_cfg.get("lookback_align", 1)),
    ) if mtf_enabled else True

    # Bracket params (unchanged)
    bcfg = cfg.get("brackets", {})
    atr_mult_sl = float(bcfg.get("atr_mult_sl", 1.2))
    take_profit_r = float(bcfg.get("take_profit_r", 1.0))
    use_brackets = (atr_mult_sl is not None) and (take_profit_r is not None)

    # Position cap
    max_notional = float(cfg.get("max_notional_per_trade", 0))

    equity = float(start_cash)
    in_pos = False
    qty = 0
    entry_px = 0.0
    entry_ts = None

    wins = 0
    losses = 0
    gross = 0.0
    trades_printed = []
    detailed_trades = [] # List of dicts for CSV export

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        cur = df.iloc[i]
        open_px = float(cur["open"])

        if not in_pos:
            x = crossover(df.iloc[:i])
            if x == 1:
                mtf_ok = True if not mtf_enabled else bool(prev.get("htf_bias", False))
                if not long_ok(prev, cfg) or not mtf_ok:
                    reasons = _eval_long_reasons(prev, cfg, mtf_ok)
                    if reasons:
                        print("    REJECT {}: {}".format(cur.name, "; ".join(reasons)))
                        # --- count by normalized reason key ---
                        for r in reasons:
                            FILTER_REJECTS[_reason_key(r)] += 1
                    else:
                        # still count a generic bucket if no specific reasons were produced
                        FILTER_REJECTS["UNSPEC"] += 1
                    continue


                # Risk-based sizing with $2,000 cap (1R = atr_mult_sl * ATR(prev))
                atr_val = float(prev.get("atr", 0.0))
                if atr_val <= 0:
                    print("    REJECT {}: no ATR for risk sizing".format(cur.name))
                    continue
                stop_dist = max(0.01, atr_mult_sl * atr_val)  # 1R
                risk_dollars = equity * float(risk_pct)
                raw_qty = int(risk_dollars // stop_dist) if stop_dist > 0 else 0
                if raw_qty <= 0:
                    print("    REJECT {}: qty<=0 (risk_dollars={:.2f}, stop_dist={:.2f})".format(cur.name, risk_dollars, stop_dist))
                    continue
                cap_qty = int((max_notional // open_px)) if max_notional > 0 else raw_qty
                qty = max(1, min(raw_qty, cap_qty))
                if qty <= 0:
                    print("    REJECT {}: cap_qty<=0 (open={:.2f})".format(cur.name, open_px))
                    continue

                entry_px = open_px
                in_pos = True
                entry_ts = cur.name
                
                # Capture entry state
                entry_state = {
                    "adx": float(prev.get("adx", 0)),
                    "rsi": float(prev.get("rsi", 0)),
                    "slope": float(prev.get("ema_slope_pct", 0)),
                    "rvol": float(prev.get("rvol", 0)),
                    "atr": float(prev.get("atr", 0)),
                    "hour": entry_ts.hour,
                }

        else:
            x = crossover(df.iloc[:i])
            exit_now_on_cross = (x == -1)

            # Bracket exits (unchanged)
            atr_val = float(prev.get("atr", 0.0))
            stop_px = None
            tp_px = None
            if use_brackets and atr_val > 0:
                stop_px = max(0.01, entry_px - atr_mult_sl * atr_val)
                tp_px = entry_px + take_profit_r * atr_val

            hit_sl = (stop_px is not None) and (cur["low"] <= stop_px)
            hit_tp = (tp_px is not None) and (cur["high"] >= tp_px)

            if hit_sl:
                exit_px = stop_px
                pnl = (exit_px - entry_px) * qty
                equity += pnl
                gross += pnl
                losses += 1
                in_pos = False
                qty = 0
                trades_printed.append((entry_ts, entry_px, cur.name, exit_px, pnl))
                detailed_trades.append({
                    "symbol": "UNKNOWN",
                    "entry_time": entry_ts,
                    "exit_time": cur.name,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "pnl": pnl,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "reason": "SL",
                    **entry_state
                })
                continue

            if hit_tp:
                exit_px = tp_px
                pnl = (exit_px - entry_px) * qty
                equity += pnl
                gross += pnl
                wins += 1
                in_pos = False
                qty = 0
                trades_printed.append((entry_ts, entry_px, cur.name, exit_px, pnl))
                detailed_trades.append({
                    "symbol": "UNKNOWN",
                    "entry_time": entry_ts,
                    "exit_time": cur.name,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "pnl": pnl,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "reason": "TP",
                    **entry_state
                })
                continue

            if exit_now_on_cross:
                exit_px = open_px
                pnl = (exit_px - entry_px) * qty
                equity += pnl
                gross += pnl
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                in_pos = False
                qty = 0
                trades_printed.append((entry_ts, entry_px, cur.name, exit_px, pnl))
                detailed_trades.append({
                    "symbol": "UNKNOWN",
                    "entry_time": entry_ts,
                    "exit_time": cur.name,
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "pnl": pnl,
                    "result": "WIN" if pnl > 0 else "LOSS",
                    "reason": "CROSS",
                    **entry_state
                })

    # Print collected trades (style kept similar)
    for (ent_ts, ent_px, ex_ts, ex_px, pnl) in trades_printed:
        print("    ENTRY {} @ {:.2f}  ->  EXIT {} @ {:.2f}  PnL={:.2f}".format(ent_ts, ent_px, ex_ts, ex_px, pnl))

    trades = wins + losses
    net = gross
    ret_pct = (equity / start_cash - 1.0) * 100.0 if start_cash else 0.0
    wr = (100.0 * wins / max(trades, 1))
    return dict(
        trades=trades,
        gross=round(gross, 2),
        net=round(net, 2),
        win_rate=round(wr, 1),
        ret_pct=round(ret_pct, 2),
        equity=round(equity, 2),
        detailed_trades=detailed_trades
    )


# ------------------------------------------------------------------
# CLI & runner (unchanged except passing risk into simulate_long_only)
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="EMAMerged/config.yaml")
    ap.add_argument("--tickers", default="EMAMerged/tickers.txt")
    ap.add_argument("--symbols", default="")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--cash", type=float, default=10_000)
    ap.add_argument("--risk", type=float, default=0.01, help="Risk per trade as fraction of equity (e.g., 0.01 = 1%)")
    ap.add_argument("--timeframe", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--rth-only", dest="rth_only", action="store_true")
    ap.add_argument("--no-rth-only", dest="rth_only", action="store_false")
    ap.set_defaults(rth_only=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Resolve symbols
    if args.symbols.strip():
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = read_tickers(args.tickers)

    # Diagnostics for header
    fcfg = dict(cfg.get("filters", {}))
    adx_th = float(fcfg.get("adx_threshold", 25.0))

    def _load(sym: str) -> pd.DataFrame:
        return load_bars_for_symbol(
            sym, cfg, args.days,
            timeframe_override=args.timeframe,
            limit_override=args.limit,
            rth_only_override=args.rth_only,
        )

    rows = []
    for sym in symbols:
        try:
            df = _load(sym)
        except Exception as e:
            print(f"[{sym}] data error: {e}")
            continue

        if df.empty:
            print(f"[{sym}] bars=0")
            print(f"  TRADES {sym}:")
            rows.append({"symbol": sym, "trades": 0, "gross": 0.0, "net": 0.0, "win_rate": 0.0, "ret_pct": 0.0, "equity": float(args.cash)})
            continue

        di = compute_indicators(df, cfg)
        crosses_up = sum(crossover(di.iloc[:i]) == 1 for i in range(1, len(di)))
        pct_adx = (di["adx"] >= adx_th).mean() * 100.0 if "adx" in di.columns else 0.0
        print(f"[{sym}] bars={len(df)}, crosses_up={crosses_up}, %ADX>={adx_th:.0f}={pct_adx:.1f}%")

        print(f"  TRADES {sym}:")
        res = simulate_long_only(df, cfg, start_cash=args.cash, risk_pct=args.risk)
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
    
    # Export detailed trades
    all_trades = []
    for r in rows:
        sym = r["symbol"]
        for t in r.get("detailed_trades", []):
            t["symbol"] = sym
            all_trades.append(t)
            
    if all_trades:
        df_trades = pd.DataFrame(all_trades)
        csv_path = "trades.csv"
        df_trades.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df_trades)} trades to {csv_path}")
    # --- Filter Impact Report ---
    print("\n=== Filter Impact Report ===")
    total_rejects = sum(FILTER_REJECTS.values())
    if total_rejects == 0:
        print("(no rejections recorded)")
    else:
        for filt, count in FILTER_REJECTS.most_common():
            pct = 100.0 * count / total_rejects
            print(f"{filt:10s} {count:6d} ({pct:4.1f}%)")

if __name__ == "__main__":
    main()
