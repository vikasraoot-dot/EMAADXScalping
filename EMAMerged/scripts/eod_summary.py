from __future__ import annotations
import json, sys, math
from collections import defaultdict
from datetime import datetime

def main():
    if len(sys.argv) < 2:
        print("Usage: python eod_summary.py <path/to/trades_YYYYMMDD.jsonl>")
        sys.exit(1)

    path = sys.argv[1]
    trades = []
    entries = {}
    realized = 0.0
    wins = losses = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            e = json.loads(line)
            typ = e.get("type")
            if typ == "ENTRY_FILL":
                cid = e.get("cid")
                entries[cid] = {
                    "symbol": e.get("symbol"),
                    "entry_ts": e.get("ts"),
                    "entry": float(e.get("fill_price", 0.0)),
                    "qty": float(e.get("fill_qty", 0.0)),
                }
            elif typ == "EXIT_FILL":
                cid = e.get("cid")
                en = entries.pop(cid, None)
                pnl = float(e.get("pnl", 0.0)) if "pnl" in e else float(e.get("exit_price", 0.0)) - float(en["entry"]) if en else 0.0
                realized += pnl
                if pnl > 0: wins += 1
                else: losses += 1
                trades.append({
                    "symbol": e.get("symbol"),
                    "entry_ts": en["entry_ts"] if en else None,
                    "entry": en["entry"] if en else None,
                    "exit_ts": e.get("ts"),
                    "exit": float(e.get("exit_price", 0.0)),
                    "pnl": round(pnl, 2),
                    "reason": e.get("reason","?"),
                })

    win_rate = (wins / max(1, wins+losses)) * 100.0
    out = {
        "realized_pnl": round(realized, 2),
        "win_rate_pct": round(win_rate, 1),
        "n_trades": len(trades),
        "trades": trades,
        "open_cids": list(entries.keys())
    }
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()