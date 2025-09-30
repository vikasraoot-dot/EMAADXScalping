from __future__ import annotations
import os, sys, argparse, math
import pandas as pd
from datetime import datetime, timedelta
import pytz

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.data import (
    get_alpaca_bars, filter_rth, drop_unclosed_last_bar, alpaca_market_open
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok, explain_long_gate

# NEW: lifecycle logger & execution sequencing
from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.execution import execute_long_with_oco, list_activities_fills, get_positions

# -------------------------
# Globals (kept minimal)
# -------------------------
_ERROR_COOLDOWN: dict[str, datetime] = {}
ET = pytz.timezone("US/Eastern")

def _cid(sym: str) -> str:
    # deterministically includes date+minute to help correlate loops
    now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return f"EMA_{sym}_{now}"

def _log_path(cfg_path: str) -> str:
    root = os.path.dirname(os.path.abspath(cfg_path))
    dstr = datetime.utcnow().strftime("%Y%m%d")
    logs_dir = os.path.join(root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"trades_{dstr}.jsonl")

def manage_symbol(sym: str, cfg: dict, args: argparse.Namespace, logger: TradeLogger):
    try:
        # 1) bars & indicators
        bars = get_alpaca_bars(sym, timeframe=cfg["timeframe"], days=int(cfg.get("history_days", 30)))
        if bars is None or len(bars) == 0:
            logger.error(symbol=sym, stage="DATA", error_code="NO_BARS")
            return
        bars = filter_rth(drop_unclosed_last_bar(bars))
        if bars is None or len(bars) < 50:
            logger.error(symbol=sym, stage="DATA", error_code="TOO_FEW_BARS", detail=len(bars) if bars is not None else 0)
            return

        df = compute_indicators(bars, cfg).copy()
        cross = crossover(df, cfg)  # typically last-bar cross
        row = df.iloc[-1].to_dict()

        # 2) scanner + gate telemetry
        cid = _cid(sym)
        logger.signal(symbol=sym, session=args.session, cid=cid, tf=cfg["timeframe"],
                      cross=int(cross), ref_bar_ts=str(df.index[-1]),
                      last_close=float(row.get("close", float('nan'))),
                      adx=float(row.get("adx", float('nan'))),
                      ema_slope_pct=float(row.get("ema_slope_pct", float('nan'))),
                      dollar_vol_avg=float(row.get("dollar_vol_avg", float('nan'))))

        ok = long_ok(pd.Series(row), cfg)
        if not ok:
            # Also include verbose reason in a separate line (so it’s parseable)
            # (We call explain_long_gate for human-readable flags)
            gate_ok, reasons = explain_long_gate(pd.Series(row), cfg)
            logger.gate(symbol=sym, session=args.session, cid=cid, decision="BLOCK", reasons=reasons)
            return

        if not cross:
            logger.gate(symbol=sym, session=args.session, cid=cid, decision="PASS", reasons=["cross=0"])
            return

        # 3) compute intended TP/SL (kept your config convention; adjust if your code computes elsewhere)
        bcfg = cfg.get("brackets", {})
        take_profit_at = float(bcfg.get("tp_abs") or 0.0)
        stop_loss_at   = float(bcfg.get("sl_abs") or 0.0)

        px = float(row["close"])
        tp = round2(px + take_profit_at) if take_profit_at > 0 else round2(px * (1 + float(bcfg.get("tp_pct", 0))/100.0))
        sl = round2(px - stop_loss_at)   if stop_loss_at > 0 else round2(px * (1 - float(bcfg.get("sl_pct", 0))/100.0))

        qty = int(cfg.get("qty", 1))
        # 4) EXECUTION (safe sequencing; lifecycle-logged)
        res = execute_long_with_oco(
            logger=logger,
            symbol=sym,
            qty=qty,
            intended_tp=float(tp),
            intended_sl=float(sl),
            session=args.session,
            cid=cid,
            prefer_limit_entry=bool(cfg.get("prefer_limit_entry", False)),
            limit_px=float(cfg.get("limit_entry_offset_px", 0.0)) + px if bool(cfg.get("prefer_limit_entry", False)) else None,
        )

        if res.get("status") != "ok":
            # backoff this symbol for a short period to avoid spamming rejects
            _ERROR_COOLDOWN[sym] = new_york_now() + timedelta(minutes=3)
            return

        # 5) optional: include a lightweight snapshot of positions after entry
        try:
            pos = get_positions(sym)
            if pos:
                p = pos[0]
                logger.snapshot(session=args.session,
                                open_positions=[{
                                    "symbol": p.get("symbol"),
                                    "side": "LONG" if float(p.get("qty", 0)) > 0 else "SHORT",
                                    "qty": float(p.get("qty", 0)),
                                    "avg_price": float(p.get("avg_entry_price", 0.0)),
                                }])
        except Exception as e:
            logger.error(symbol=sym, session=args.session, cid=cid, stage="SNAPSHOT", error_code="POS_ERR", error_text=str(e))

    except Exception as e:
        logger.error(symbol=sym, session=args.session, stage="MANAGE", error_text=str(e))
        # cooldown this symbol to avoid tight loops on errors
        _ERROR_COOLDOWN[sym] = new_york_now() + timedelta(minutes=3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--session", type=str, default="AM")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not alpaca_market_open():
        # Keep old behavior but emit heartbeat for observability
        logger = TradeLogger(_log_path(args.config))
        logger.heartbeat(session=args.session, market_open=False)
        return

    # Initialize logger per run; one JSONL per UTC day
    logger = TradeLogger(_log_path(args.config))
    logger.heartbeat(session=args.session, market_open=True)

    # Determine symbols
    if args.tickers:
        symbols = read_tickers(args.tickers)
    else:
        symbols = cfg.get("symbols", [])
        if isinstance(symbols, str) and os.path.exists(symbols):
            symbols = read_tickers(symbols)

    # Per-symbol cooldown honored (existing behavior preserved)
    for sym in symbols:
        try:
            until = _ERROR_COOLDOWN.get(sym)
            if until and new_york_now() < until:
                # keep your human-readable line AND log
                print(f"[{sym}] on error cooldown until {until.strftime('%H:%M')} (skip)")
                logger.heartbeat(symbol=sym, cooldown_until=until.astimezone(ET).strftime("%H:%M"))
                continue
            manage_symbol(sym, cfg, args, logger)
        except Exception as e:
            print(f"[{sym}] fatal: {e}")
            logger.error(symbol=sym, session=args.session, stage="LOOP", error_text=str(e))

if __name__ == "__main__":
    main()