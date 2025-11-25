#!/usr/bin/env python3
"""
Grid Search Optimizer for EMA+ADX Strategy
Runs backtest_30d.py logic over a grid of parameters to find the best Sharpe/Return.
"""
import os
import sys
import itertools
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from EMAMerged.src.utils import load_config, read_tickers
from EMAMerged.src.data import fetch_latest_bars, filter_rth, drop_unclosed_last_bar
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok

def load_data(symbols, days=30):
    print(f"Loading data for {len(symbols)} symbols over {days} days...")
    bars_map, _ = fetch_latest_bars(
        symbols=symbols,
        timeframe="15Min",
        history_days=days,
        feed="iex",
        rth_only=True
    )
    # Post-process
    clean_map = {}
    for s, df in bars_map.items():
        if not df.empty:
            df = drop_unclosed_last_bar(df, "15Min")
            clean_map[s] = df
    return clean_map

def run_backtest(params, data_map):
    """
    Run simulation on pre-loaded data with specific params.
    params: dict of {rsi_min, adx_threshold, atr_mult_sl, take_profit_r}
    """
    # Create a temporary config override
    cfg = {
        "ema_fast": 9,
        "ema_slow": 21,
        "atr_length": 14,
        "filters": {
            "adx_threshold": params["adx_threshold"],
            "rsi_min": params["rsi_min"],
            "rsi_max": 85,
            "slope_threshold_pct": 0.0004,
            "require_fast_above_slow": True,
            "require_cross": True,
            "require_di_trend": True,
        },
        "brackets": {
            "atr_mult_sl": params["atr_mult_sl"],
            "take_profit_r": params["take_profit_r"],
        }
    }

    total_trades = 0
    total_net = 0.0
    wins = 0
    
    for s, df in data_map.items():
        if df.empty or len(df) < 50:
            continue
            
        # Calc indicators
        di = compute_indicators(df, cfg)
        
        # Sim loop (simplified from backtest_30d)
        equity = 10000.0
        in_pos = False
        entry_px = 0.0
        qty = 0
        
        for i in range(1, len(di)):
            prev = di.iloc[i-1]
            cur = di.iloc[i]
            
            if not in_pos:
                if crossover(di.iloc[:i]) == 1:
                    if long_ok(prev, cfg):
                        # Entry
                        atr = float(prev.get("atr", 0))
                        if atr <= 0: continue
                        
                        risk_amt = 100.0 # Fixed $100 risk per trade for standardized comparison
                        stop_dist = params["atr_mult_sl"] * atr
                        qty = int(risk_amt / stop_dist) if stop_dist > 0 else 0
                        if qty < 1: continue
                        
                        entry_px = float(cur["open"])
                        in_pos = True
            else:
                # Exit logic
                atr = float(prev.get("atr", 0))
                sl_price = entry_px - (params["atr_mult_sl"] * atr)
                tp_price = entry_px + (params["take_profit_r"] * params["atr_mult_sl"] * atr) # R-multiple of risk
                
                low = float(cur["low"])
                high = float(cur["high"])
                
                exit_px = None
                if low <= sl_price:
                    exit_px = sl_price
                elif high >= tp_price:
                    exit_px = tp_price
                elif crossover(di.iloc[:i]) == -1:
                    exit_px = float(cur["open"])
                    
                if exit_px:
                    pnl = (exit_px - entry_px) * qty
                    total_net += pnl
                    total_trades += 1
                    if pnl > 0: wins += 1
                    in_pos = False
                    qty = 0

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    return {
        **params,
        "trades": total_trades,
        "net": round(total_net, 2),
        "win_rate": round(win_rate, 1),
        "avg_pnl": round(total_net / total_trades, 2) if total_trades > 0 else 0
    }

def main():
    # 1. Load Data (Top 20 Liquid Tickers for speed)
    tickers = ["NVDA", "TSLA", "AMD", "AAPL", "MSFT", "AMZN", "META", "COIN", "MARA", "PLTR", "GME", "AMC", "TQQQ", "SQQQ", "SOXL", "SPY", "QQQ", "IWM", "ARKK", "LABU"]
    data_map = load_data(tickers, days=30)
    
    # 2. Define Grid
    rsi_mins = [30, 40, 45]
    adx_ths = [15, 20, 25]
    sl_mults = [1.0, 1.5, 2.0]
    tp_rs = [1.5, 2.0, 3.0] # Reward/Risk ratio
    
    param_grid = []
    for r, a, s, t in itertools.product(rsi_mins, adx_ths, sl_mults, tp_rs):
        param_grid.append({
            "rsi_min": r,
            "adx_threshold": a,
            "atr_mult_sl": s,
            "take_profit_r": t
        })
        
    print(f"Running {len(param_grid)} combinations...")
    
    # 3. Run Grid Search
    results = []
    # Serial execution for simplicity/stability, can be parallelized
    for i, p in enumerate(param_grid):
        res = run_backtest(p, data_map)
        results.append(res)
        if i % 10 == 0:
            print(f"Done {i}/{len(param_grid)}...")

    # 4. Analyze
    df = pd.DataFrame(results)
    df["score"] = df["net"] * df["win_rate"] / 100.0 # Simple score
    
    print("\n=== TOP 10 CONFIGURATIONS ===")
    print(df.sort_values("net", ascending=False).head(10).to_string(index=False))
    
    best = df.sort_values("net", ascending=False).iloc[0]
    print("\n=== RECOMMENDATION ===")
    print(best)

if __name__ == "__main__":
    main()
