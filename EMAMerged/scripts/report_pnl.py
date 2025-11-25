import os
import sys
import requests
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

def get_env_var(keys, default=None):
    for k in keys:
        val = os.getenv(k)
        if val: return val
    return default

def main():
    # 1. Setup Credentials
    key = get_env_var(["ALPACA_KEY_ID", "APCA_API_KEY_ID", "ALPACA_KEY"])
    secret = get_env_var(["ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY", "ALPACA_SECRET"])
    base_url = get_env_var(["ALPACA_BASE_URL", "APCA_API_BASE_URL"], "https://paper-api.alpaca.markets").rstrip("/")

    if not key or not secret:
        print("Error: Alpaca credentials not found in environment.")
        return

    headers = {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json"
    }

    # 2. Define "Today" in ET
    et = ZoneInfo("US/Eastern")
    now = datetime.now(et)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    today_iso = today_start.isoformat()

    print(f"Fetching fills since {today_iso}...")

    # 3. Fetch Account (to check status)
    try:
        r = requests.get(f"{base_url}/v2/account", headers=headers, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Failed to connect to Alpaca: {e}")
        return

    # 4. Fetch Orders (Closed Today)
    # We fetch closed orders and filter locally because 'after' param applies to submission time, not fill time.
    params = {
        "status": "closed",
        "limit": 500,
        "after": today_iso 
    }
    
    try:
        r = requests.get(f"{base_url}/v2/orders", headers=headers, params=params, timeout=10)
        r.raise_for_status()
        orders = r.json()
    except Exception as e:
        print(f"Failed to fetch orders: {e}")
        return

    if not orders:
        print("No closed orders found for today.")
        return

    # 5. Process Fills
    trades = []
    for o in orders:
        # Check if filled
        if not o.get("filled_at"):
            continue
            
        # Parse fill time
        # Alpaca returns UTC ISO strings like '2023-10-27T14:00:00.123456Z'
        fill_ts = datetime.fromisoformat(o["filled_at"].replace("Z", "+00:00")).astimezone(et)
        
        if fill_ts >= today_start:
            trades.append({
                "symbol": o["symbol"],
                "side": o["side"],
                "qty": float(o["filled_qty"]),
                "price": float(o["filled_avg_price"]),
                "time": fill_ts,
                "notional": float(o["filled_qty"]) * float(o["filled_avg_price"])
            })

    if not trades:
        print("No fills found for today (orders might be cancelled/expired).")
        return

    df = pd.DataFrame(trades)
    df.sort_values("time", inplace=True)

    # 6. Calculate PnL
    pnl_summary = []
    
    for sym, group in df.groupby("symbol"):
        buys = group[group["side"] == "buy"]["notional"].sum()
        sells = group[group["side"] == "sell"]["notional"].sum()
        
        # Net Cash Flow (assuming flat)
        net_pnl = sells - buys
        
        pnl_summary.append({
            "Ticker": sym,
            "Trades": len(group),
            "Cost Basis": buys,
            "Sell Price": sells,
            "Net PnL": net_pnl
        })

    summary_df = pd.DataFrame(pnl_summary)
    total_pnl = summary_df["Net PnL"].sum()

    # 7. Print Report
    print("\n=== DAILY TRADING REPORT ===")
    print(summary_df.to_string(index=False, float_format="%.2f"))
    print("-" * 40)
    print(f"TOTAL PnL: ${total_pnl:.2f}")
    print("============================")

if __name__ == "__main__":
    main()
