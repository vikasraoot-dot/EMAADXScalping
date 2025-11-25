import pandas as pd
import numpy as np

def analyze_losses(csv_path="trades.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("trades.csv not found. Run backtest first.")
        return

    if df.empty:
        print("No trades found.")
        return

    print(f"Loaded {len(df)} trades.")
    
    # Separate winners and losers
    winners = df[df["pnl"] > 0]
    losers = df[df["pnl"] <= 0]
    
    print(f"Winners: {len(winners)} | Losers: {len(losers)}")
    
    if losers.empty:
        print("No losers to analyze!")
        return

    print("\n=== LOSS ANALYSIS ===")
    
    # 1. Time of Day Analysis
    losers["hour"] = pd.to_numeric(losers["hour"])
    hourly_losses = losers.groupby("hour").size()
    hourly_winrate = df.groupby("hour").apply(lambda x: (x["pnl"] > 0).mean())
    
    print("\n[Time of Day]")
    print("Hour | Losers | Win Rate")
    for h in sorted(hourly_losses.index):
        print(f"{h:02d}   | {hourly_losses[h]:4d}   | {hourly_winrate.get(h, 0):.1%}")

    # 2. Indicator Analysis (Mean values)
    print("\n[Indicator Means]")
    print(f"{'Metric':<10} | {'Winners':<10} | {'Losers':<10} | {'Diff':<10}")
    for col in ["adx", "rsi", "slope", "rvol", "atr"]:
        w_mean = winners[col].mean()
        l_mean = losers[col].mean()
        diff = ((w_mean - l_mean) / l_mean) * 100 if l_mean != 0 else 0
        print(f"{col:<10} | {w_mean:10.4f} | {l_mean:10.4f} | {diff:+.1f}%")

    # 3. "Crash" Analysis (Did we buy a falling knife?)
    # Check if RSI was extremely low (oversold) but it kept dropping
    crash_losers = losers[losers["rsi"] < 35]
    print(f"\n[Crash Candidates] (RSI < 35 at entry)")
    print(f"Count: {len(crash_losers)} ({len(crash_losers)/len(losers):.1%} of losers)")
    
    # 4. "Chop" Analysis (Low ADX)
    chop_losers = losers[losers["adx"] < 25]
    print(f"\n[Chop Candidates] (ADX < 25 at entry)")
    print(f"Count: {len(chop_losers)} ({len(chop_losers)/len(losers):.1%} of losers)")

    # 5. "Lunch Chop" (12:00 - 13:00)
    lunch_losers = losers[(losers["hour"] >= 12) & (losers["hour"] < 13)]
    print(f"\n[Lunch Chop] (12:00-13:00)")
    print(f"Count: {len(lunch_losers)} ({len(lunch_losers)/len(losers):.1%} of losers)")

if __name__ == "__main__":
    analyze_losses()
