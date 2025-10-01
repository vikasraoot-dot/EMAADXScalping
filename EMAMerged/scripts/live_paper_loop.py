from __future__ import annotations
import os, sys, argparse
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.data import (
    get_alpaca_bars, filter_rth, drop_unclosed_last_bar, alpaca_market_open
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok, explain_long_gate

# lifecycle logger & execution sequencing
from EMAMerged.src.trade_logger import TradeLogger
from EMAMerged.src.execution import (
    execute_long_with_oco,
    get_positions,
    configure_alpaca,
)

ET = pytz.timezone("US/Eastern")
_ERROR_COOLDOWN: dict[str, datetime] = {}

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
    """
    Resolve Alpaca creds in this order:
      1) Env: APCA_BASE_URL / APCA_API_KEY_ID / APCA_API_SECRET_KEY
              (or ALPACA_KEY / ALPACA_SECRET as alternates)
      2) Config: cfg['broker'] = {base_url,key,secret}
    """
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
            error_text="Alpaca creds missing: set APCA_BASE_URL + (APCA_API_KEY_ID|ALPACA_KEY) + (APCA_API_SECRET_KEY|ALPACA_SECRET) or set broker.{base_url,key,secret} in config."
        )
    return ok

def manage_symbol(sym: str, cfg: dict, args: argparse.Namespace, logger: TradeLogger, key: str, secret: str):
    try:
        # 1) Pull bars (historic fetch works after-hours)
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

        # IMPORTANT: pass timeframe to drop_unclosed_last_bar
        bars = drop_unclosed_last_bar(bars, cfg["timeframe"])
        bars = filter_rth(bars)
        if bars is None or len(bars) < 50:
            logger.error(symbol=sym, stage="DATA", error_code="TOO_FEW_BARS",
                         detail=len(bars) if bars is not None else 0)
            return

        # 2) Indicators + scanner
        df = compute_indicators(bars, cfg).copy()
        cross = crossover(df, cfg)
        row = df.iloc[-1].to_dict()

        # 3) Scanner + gate telemetry
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

        # 4) Brackets (unchanged logic)
        bcfg = cfg.get("brackets", {})
        take_profit_at = float(bcfg.get("tp_abs") or 0.0)
        stop_loss_at   = float(bcfg.get("sl_abs") or 0.0)

        px = float(row["close"])
        tp = round2(px + take_profit_at) if take_profit_at > 0 else round2(px * (1 + float(bcfg.get("tp_pct", 0))/100.0))
        sl = round2(px - stop_loss_at)   if stop_loss_at > 0 else round2(px * (1 - float(bcfg.get("sl_pct", 0))/100.0))

        qty = int(cfg.get("qty", 1))

        # 5) Execution (uses execution.configure_alpaca creds set in main())
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
            # Cooldown to avoid thrashing on the same symbol
            _ERROR_COOLDOWN[sym] = new_york_now() + timedelta(minutes=3)
            return

        # 6) Optional snapshot
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
            logger.error(symbol=sym, session=args.session, cid=cid, stage="SNAPSHOT",
                         error_code="POS_ERR", error_text=str(e))

    except Exception as e:
        logger.error(symbol=sym, session=args.session, stage="MANAGE", error_text=str(e))
        _ERROR_COOLDOWN[sym] = new_york_now() + timedelta(minutes=3)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tickers", type=str, default=None)
    parser.add_argument("--symbols", type=str, default=None)  # NEW: accept comma-separated symbols
    parser.add_argument("--session", type=str, default="AM")
    parser.add_argument("--dry-run", type=int, default=0)     # accepted for compatibility
    parser.add_argument("--force-run", type=int, default=0)   # use 1 to bypass open check after-hours
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = TradeLogger(_log_path(args.config))

    # Resolve creds from env or config
    base_url, key, secret = _resolve_alpaca(cfg)
    if not _require_creds_or_bail(logger, base_url, key, secret):
        print("[live] MISSING Alpaca credentials. Set env or add broker section in config; see README.")
        sys.exit(2)

    # Ensure execution.py uses the same creds (it defaults to env)
    configure_alpaca(base_url, key, secret)

    # Market-open check (skip if --force-run 1 for after-hours testing)
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

    # Determine symbols
    symbols: list[str] = []
    if args.symbols:
        # split comma/space; uppercase & dedupe
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
            until = _ERROR_COOLDOWN.get(sym)
            if until and new_york_now() < until:
                logger.heartbeat(symbol=sym, cooldown_until=until.astimezone(ET).strftime("%H:%M"))
                continue
            manage_symbol(sym, cfg, args, logger, key=key, secret=secret)
        except Exception as e:
            logger.error(symbol=sym, session=args.session, stage="LOOP", error_text=str(e))

if __name__ == "__main__":
    main()
