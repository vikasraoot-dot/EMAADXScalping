from __future__ import annotations
import os, sys, argparse, time
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

# NEW: logger + Alpaca helpers (no churn in data.py)
from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.alpaca_extensions import get_order, get_activities

# -------------------------
# Minimal globals—kept tiny to reduce churn
_ERROR_COOLDOWN = {}  # sym -> until_dt
LAST_ACTIVITY_TS = None  # for FILL polling
OPEN_TRADES = {}  # cid -> dict(symbol, parent_id, tp_id, sl_id, entry_price, qty, tp, sl)

def _cfg_bool(cfg: dict, path: str, default: bool) -> bool:
    cur = cfg
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur: return default
        cur = cur[k]
    return bool(cur)

def _alpaca_env(cfg: dict):
    key = os.environ.get("ALPACA_KEY") or cfg.get("alpaca", {}).get("key")
    secret = os.environ.get("ALPACA_SECRET") or cfg.get("alpaca", {}).get("secret")
    base_url = os.environ.get("APCA_BASE_URL") or cfg.get("alpaca", {}).get("base_url", "https://paper-api.alpaca.markets")
    if not key or not secret:
        raise RuntimeError("Alpaca API credentials not found (env or config).")
    return base_url, key, secret

def _build_cid(sym: str) -> str:
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"EMA_{sym}_{now}"

def _now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def _poll_until_filled(base_url, key, secret, order_id, timeout_s=30, sleep_s=1):
    """Poll order until filled/canceled or timeout; returns order JSON."""
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        ordj = get_order(base_url, key, secret, order_id)
        last = ordj
        st = (ordj or {}).get("status")
        if st in ("filled", "partially_filled", "canceled", "rejected"):
            return ordj
        time.sleep(sleep_s)
    return last

def _log_snapshot(logger: TradeLogger, base_url, key, secret):
    try:
        pos = get_positions(base_url, key, secret)
        snap = [{"symbol": p["symbol"], "side": p["side"], "qty": float(p.get("qty", 0)),
                 "avg_price": float(p.get("avg_entry_price", 0)), "market": float(p.get("market_value", 0))}
                for p in pos.values()] if isinstance(pos, dict) else []
        logger.snapshot(session="AMPM", open_positions=snap)
    except Exception as e:
        logger.error(stage="SNAPSHOT", error_code="SNAPSHOT_FAIL", error_text=str(e))

def manage_symbol(sym: str, cfg: dict, args, logger: TradeLogger):
    base_url, key, secret = _alpaca_env(cfg)

    # A) bars & indicators (unchanged pattern)
    bars = get_alpaca_bars(base_url, key, secret, sym, timeframe=cfg["data"]["timeframe"], days=int(cfg["data"]["days"]))
    if bars is None or len(bars) < 40:
        return
    bars = filter_rth(bars)
    bars = drop_unclosed_last_bar(bars)
    df = compute_indicators(bars, cfg)
    row = df.iloc[-1]

    # B) signal
    cross = crossover(df, cfg)
    ok = long_ok(row, cfg)
    gates = {"adx_ok": True, "ema_ok": True, "dvol_ok": True}  # filters already check; keep brief
    cid = _build_cid(sym)
    logger.signal(symbol=sym, session="AMPM", cid=cid, tf=cfg["data"]["timeframe"],
                  cross=int(cross), close=float(row["close"]),
                  ref_bar={"t": str(df.index[-1]), "o": float(row["open"]), "h": float(row["high"]),
                           "l": float(row["low"]), "c": float(row["close"]), "v": float(row.get("volume", 0))},
                  gates=gates, decision=("ENTER_LONG" if (cross and ok) else "PASS"))
    if not (cross and ok):
        return

    # C) bracket levels
    bcfg = cfg.get("brackets", {})
    tp = round2(row["close"] * (1.0 + float(bcfg.get("tp_pct", 0.003))))
    sl = round2(row["close"] * (1.0 - float(bcfg.get("sl_pct", 0.003))))
    qty = int(max(1, bcfg.get("qty", 1)))

    # D) sequencing strategy
    single_call_bracket = _cfg_bool(cfg, "orders.single_call_bracket", True)

    if single_call_bracket:
        # Safer: one bracket order → no “insufficient qty”
        logger.entry_submit(symbol=sym, session="AMPM", cid=cid, side="BUY",
                            qty=qty, order_type="market",
                            intended={"tp": tp, "sl": sl})
        try:
            res = submit_bracket_order(base_url, key, secret, symbol=sym, qty=qty,
                                       side="buy", order_type="market",
                                       take_profit={"limit_price": tp},
                                       stop_loss={"stop_price": sl})
            parent_id = res.get("id")
            legs = res.get("legs") or []
            tp_id = next((l.get("id") for l in legs if l.get("side") == "sell" and l.get("type") == "limit"), None)
            sl_id = next((l.get("id") for l in legs if l.get("side") == "sell" and l.get("type") == "stop"), None)
            logger.entry_ack(symbol=sym, session="AMPM", cid=cid, client_order_id=res.get("client_order_id"),
                             broker_order_id=parent_id, status=res.get("status","?"))
            if tp_id or sl_id:
                logger.oco_ack(symbol=sym, session="AMPM", cid=cid, oco_client_id=None,
                               tp
