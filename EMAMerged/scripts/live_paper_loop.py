# EMAMerged/scripts/live_paper_loop.py
from __future__ import annotations
import os, sys, argparse
import pandas as pd
from datetime import datetime, timedelta, timezone, time as dtime
import pytz

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.data import (
    get_alpaca_bars, filter_rth, drop_unclosed_last_bar, alpaca_market_open
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok, explain_long_gate
from EMAMerged.src.config_compat import normalize_config

from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.execution import execute_long_with_oco, get_positions
from EMAMerged.src.execution_creds import configure_alpaca

def _tz(cfg) -> pytz.BaseTzInfo:
    try:
        return pytz.timezone(cfg.get("timezone", "US/Eastern"))
    except Exception:
        return pytz.timezone("US/Eastern")

def _now_tz(cfg) -> datetime:
    return datetime.now(_tz(cfg))

def _within_entry_windows(cfg) -> bool:
    """
    If config provides entry_windows: [{start:'HH:MM', end:'HH:MM'}, ...] in cfg.timezone,
    only allow entries during those windows. If not present, always True.
    """
    wins = cfg.get("entry_windows")
    if not isinstance(wins, list) or not wins:
        return True
    now_local = _now_tz(cfg).time()
    for w in wins:
        try:
            s = w.get("start", "00:00")
            e = w.get("end", "23:59")
            t0 = dtime.fromisoformat(s)
            t1 = dtime.fromisoformat(e)
            if t0 <= now_local <= t1:
                return True
        except Exception:
            # ignore malformed window
            continue
    return False

def _cid(sym: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"EMA_{sym}_{now}"

def _log_path(cfg_path: str) -> str:
    root = os.path.dirname(os.path.abspath(cfg_path))
    dstr = datetime.now(timezone.utc).strftime("%Y%m%d")
    logs_dir = os.path.join(root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    return os.path.join(logs_dir, f"trades_{dstr}.jsonl")

def _resolve_alpaca(cfg: dict) -> tuple[str,str,str]:
    b = (cfg or {}).get("broker", {}) or {}
    base_url = (
        os.getenv("APCA_BASE_URL")
        or b.get("base_url")
        or "https://paper-api.alpaca.markets"
    )
    key = (
        os.getenv("ALPACA_KEY")
        or os.getenv("APCA_API_KEY_ID")
        or b.get("key")
        or ""
    )
    secret = (
        os.getenv("ALPACA_SECRET")
        or os.getenv("APCA_API_SECRET_KEY")
        or b.get("secret")
        or ""
    )
    return base_url, key, secret

def _require_creds_or_bail(logger: TradeLogger, base_url: str, key: str, secret: str) -> bool:
    ok = bool(base_url and key and secret)
    if not ok:
        logger.error(
            stage="CREDENTIALS",
            error_code="MISSING",
            error_text=("Alpaca creds missing: set APCA_BASE_URL + (APCA_API_KEY_ID|ALPACA_KEY) "
                        "+ (APCA_API_SECRET_KEY|ALPACA_SECRET) or set broker.{base_url,key,secret} in config.")
        )
    return ok

def _compute_brackets_from_cfg(px: float, row: dict, bcfg: dict) -> tuple[float,float]:
    """
    Prefer ATR/R style if present:
      R = atr_mult_sl * ATR
      SL = px - 1.0 * R
      TP = px + take_profit_r * R
    Fallback to abs/percent keys if ATR/R not configured.
    """
    atr_mult_sl = bcfg.get("atr_mult_sl")
    take_profit_r = bcfg.get("take_profit_r")
    atr = row.get("atr") or row.get("ATR")
    if atr_mult_sl is not None and take_profit_r is not None and atr:
        try:
            R = float(atr_mult_sl) * float(atr)
            sl = round2(px - 1.0 * R)
            tp = round2(px + float(take_profit_r) * R)
            return tp, sl
        except Exception:
            pass

    tp_abs = bcfg.get("tp_abs")
    sl_abs = bcfg.get("sl_abs")
    tp_pct = bcfg.get("tp_pct")
    sl_pct = bcfg.get("sl_pct")
    if tp_abs is not None or sl_abs is not None:
        tp = round2(px + float(tp_abs or 0.0))
        sl = round2(px - float(sl_abs or 0.0))
        return tp, sl
    tp = round2(px * (1 + float(tp_pct or 0.0)/100.0))
    sl = round2(px * (1 - float(sl_pct or 0.0)/100.0))
    return tp, sl

def _cap_qty_by_limits(cfg: dict, px: float, base_qty: int) -> int:
    # max_shares_per_trade (reference)
    eff = int(base_qty)
    mshares = cfg.get("max_shares_per_trade")
    if mshares is not None:
        try:
            eff = min(eff, int(mshares))
        except Exception:
            pass
    # max_notional_per_trade (already in your cfg)
    cap = cfg.get("max_notional_per_trade")
    if cap is not None:
        try:
            cap = float(cap)
            eff = max(0, min(eff, int(cap // max(px, 1e-9))))
        except Exception:
            pass
    return eff

def manage_symbol(sym: str, cfg: dict, args: argparse.Namespace, logger: TradeLogger, key: str, secret: str):
    try:
        # DATA
        try:
            bars = get_alpaca_bars(key, secret, cfg["timeframe"], sym,
                                   days=int(cfg.get("history_days", 30)))
        except Exception as e:
            logger.error(symbol=sym, session=args.session, stage="DATA",
                         error_code="BARS_FETCH", error_text=str(e))
            return

        if bars is None or len(bars) == 0:
            logger.error(symbol=sym, stage="DATA", error_code="NO_BARS")
            return

        bars = drop_unclosed_last_bar(bars, cfg["timeframe"])
        if cfg.get("rth_only", True):
            bars = filter_rth(bars)
        if bars is None or len(bars) < 50:
            logger.error(symbol=sym, stage="DATA", error_code="TOO_FEW_BARS",
                         detail=len(bars) if bars is not None else 0)
            return

        # INDICATORS + SCANNER
        df = compute_indicators(bars, cfg).copy()
        cross = crossover(df)
        row = df.iloc[-1].to_dict()

        # SIGNAL + GATE LOGGING
        cid = _cid(sym)
        logger.signal(symbol=sym, session=args.session, cid=cid, tf=cfg["timeframe"],
                      cross=int(cross), ref_bar_ts=str(df.index[-1]),
                      last_close=float(row.get("close", float('nan'))),
                      adx=float(row.get("adx", float('nan'))),
                      ema_slope_pct=float(row.get("ema_slope_pct", float('nan'))),
                      dollar_vol_avg=float(row.get("dollar_vol_avg", float('nan'))))

        ok = long_ok(pd.Series(row), cfg)
        if not ok:
            gate_ok, reasons = explain_long_gate(pd.Series(row), cfg)
            logger.gate(symbol=sym, session=args.session, cid=cid, decision="BLOCK", reasons=reasons)
            return

        if not cross:
            logger.gate(symbol=sym, session=args.session, cid=cid, decision="PASS", reasons=["cross=0"])
            return

        # Enforce max open trades per ticker (reference knob)
        max_open = int(cfg.get("max_open_trades_per_ticker", 1))
        try:
            pos = get_positions(sym) or []
            open_qty = 0.0
            if pos:
                # alpaca-like position object; sum qty if multiple (defensive)
                for p in pos:
                    try:
                        open_qty += float(p.get("qty", 0))
                    except Exception:
                        pass
            if max_open <= 0 or open_qty > 0 and max_open <= 1:
                logger.gate(symbol=sym, session=args.session, cid=cid,
                            decision="BLOCK", reasons=[f"max_open_trades_per_ticker={max_open}"])
                return
        except Exception:
            # on error, don't block; we log later if execute fails
            pass

        # BRACKETS
        bcfg = cfg.get("brackets", {}) or {}
        px = float(row["close"])
        tp, sl = _compute_brackets_from_cfg(px, row, bcfg)

        # QTY with safety caps
        base_qty = int(cfg.get("qty", 1))
        qty = _cap_qty_by_limits(cfg, px, base_qty)
        if qty <= 0:
            logger.gate(symbol=sym, session=args.session, cid=cid,
                        decision="BLOCK", reasons=["qty=0 after caps"])
            return

        # EXECUTE
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
            return

        # SNAPSHOT (optional)
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
        except Exception:
            pass

    except Exception as e:
        logger.error(symbol=sym, session=args.session, stage="MANAGE", error_text=str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None)  # comma-separated
    parser.add_argument("--session", type=str, default="AM")
    parser.add_argument("--dry-run", type=int, default=0)
    parser.add_argument("--force-run", type=int, default=0)
    args = parser.parse_args()

    raw_cfg = load_config(args.config)
    cfg = normalize_config(raw_cfg)  # <- make reference-style config work

    logger = TradeLogger(_log_path(args.config))

    # CREDS
    base_url, key, secret = _resolve_alpaca(cfg)
    if not _require_creds_or_bail(logger, base_url, key, secret):
        print("[live] MISSING Alpaca credentials. Set env or add broker section in config.")
        sys.exit(2)
    configure_alpaca(base_url, key, secret)

    # OPEN CHECK (skip only if --force-run 1)
    if args.force_run and int(args.force_run) > 0:
        logger.heartbeat(session=args.session, market_open="FORCED_TRUE")
        is_open = True
    else:
        try:
            is_open = alpaca_market_open(base_url, key, secret)
        except Exception as e:
            logger.error(stage="MARKET_OPEN_CHECK", error_code="HTTP_JSON", error_text=str(e))
            is_open = True  # degrade gracefully

    if not is_open:
        logger.heartbeat(session=args.session, market_open=False)
        return

    logger.heartbeat(session=args.session, market_open=True)

    # Entry windows (if provided)
    if not _within_entry_windows(cfg):
        logger.gate(session=args.session, symbol=None, cid=None,
                    decision="BLOCK", reasons=["outside entry_windows"])
        return

    # SYMBOLS
    symbols: list[str] = []
    if args.symbols:
        parts = [s.strip().upper() for s in args.symbols.replace(",", " ").split() if s.strip()]
        symbols = list(dict.fromkeys(parts))
    elif args.tickers:
        symbols = read_tickers(args.tickers)
    else:
        symbols = cfg.get("symbols", [])
        if isinstance(symbols, str) and os.path.exists(symbols):
            symbols = read_tickers(symbols)

    for sym in symbols:
        try:
            manage_symbol(sym, cfg, args, logger, key=key, secret=secret)
        except Exception as e:
            logger.error(symbol=sym, session=args.session, stage="LOOP", error_text=str(e))

if __name__ == "__main__":
    main()
