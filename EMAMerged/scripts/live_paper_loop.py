# === EMAMerged/scripts/live_paper_loop.py ===
from __future__ import annotations
import os, sys, json, time, argparse
import datetime as dt
from typing import Dict, List, Any, Optional

from EMAMerged.src.utils import load_config, read_tickers, new_york_now, round2
from EMAMerged.src.tradelogger import TradeLogger
from EMAMerged.src.filters import attach_verifiers, explain_long_gate, long_ok
from EMAMerged.src.data import (
    fetch_latest_bars,     # expects (symbols, timeframe, history_days, feed) → dict[sym]->DataFrame
    alpaca_market_open,    # (base_url, key, secret) → bool
)
from EMAMerged.src.oco import ensure_oco_for_long

ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

def _utcnow() -> str:
    return dt.datetime.utcnow().strftime(ISO_UTC)

def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v

def _resolve_broker(cfg: Dict) -> Dict[str, str]:
    bk = dict(cfg.get("broker", {}))
    base_url = _env("APCA_BASE_URL", bk.get("base_url", "https://paper-api.alpaca.markets"))
    key_id   = _env("APCA_API_KEY_ID", bk.get("key"))
    secret   = _env("APCA_API_SECRET_KEY", bk.get("secret"))
    # Back-compat aliases if set
    key_id   = _env("ALPACA_KEY", key_id)
    secret   = _env("ALPACA_SECRET", secret)
    return {"base_url": base_url, "key_id": key_id, "secret": secret}

def _trades_path() -> str:
    dstr = dt.datetime.utcnow().strftime("%Y%m%d")
    p = os.path.join(os.path.dirname(__file__), "..", "logs", f"trades_{dstr}.jsonl")
    return os.path.abspath(p)

def _log_signal(log: TradeLogger, row: Dict[str, Any]):
    # row contains: symbol, cross, ref_bar_ts, last_close, adx, ema_slope_pct, rsi (if attached), etc.
    payload = {k: v for k, v in row.items() if k not in ("reasons",)}
    payload.update({"type": "SIGNAL", "ts": _utcnow()})
    log._write(payload)

def _log_gate(log: TradeLogger, symbol: str, cid: str, session: str, decision: str, reasons: List[str]):
    log.gate(symbol=symbol, session=session, cid=cid, decision=decision, reasons=reasons)

def _build_cid(symbol: str) -> str:
    return f"EMA_{symbol}_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

def _calc_tp_sl_from_cfg(cfg: Dict, entry_price: float, atr: Optional[float]) -> Dict[str, float]:
    # Your config supports both ATR multiple and R multiple; keep current behavior
    bcfg = dict(cfg.get("brackets", {}))
    r_mult = float(bcfg.get("take_profit_r", 1.2))
    atr_mult = float(bcfg.get("atr_mult_sl", 1.0))
    # If ATR provided, SL = entry - atr_mult*atr ; TP = entry + r_mult*(entry - SL)
    if atr is not None and atr > 0:
        sl = entry_price - atr_mult * atr
        risk_per_share = max(0.01, entry_price - sl)
        tp = entry_price + r_mult * risk_per_share
    else:
        # Fallback: simple 1R on 1% risk
        sl = entry_price * 0.99
        tp = entry_price * (1 + 0.012)
    return {"tp": round2(tp), "sl": round2(sl)}

def _place_oco_after_fill(
    *,
    symbol: str,
    fill_qty: int,
    intended: Dict[str, Any],
    cid: str,
    log: TradeLogger,
    session: str,
    broker: Dict[str, str],
):
    # Normalize intended tp/sl to floats
    tp = intended.get("tp")
    sl = intended.get("sl")
    if isinstance(tp, dict):  tp = float(tp.get("level") or tp.get("limit") or tp.get("price"))
    else:                     tp = float(tp)
    if isinstance(sl, dict):  sl = float(sl.get("level") or sl.get("stop")  or sl.get("price"))
    else:                     sl = float(sl)

    res = ensure_oco_for_long(
        symbol=symbol,
        intended_qty=fill_qty,
        tp_level=tp,
        sl_level=sl,
        base_url=broker["base_url"],
        key_id=broker["key_id"],
        secret=broker["secret"],
        logger=log,
        cid=cid,
        session=session,
    )
    # Optional summary snapshot
    log.snapshot(symbol=symbol, session=session, cid=cid, stage="OCO_RESULT", result=res.get("status"))

def _maybe_submit_entry(symbol: str, side: str, qty: int, price: float, intended: Dict[str, float],
                        log: TradeLogger, broker: Dict[str, str], cid: str, session: str) -> Optional[Dict[str, Any]]:
    """
    Minimal, conservative entry submit that matches your existing logging.
    Returns fill dict if filled synchronously (paper/live will usually ack then fill quickly),
    otherwise returns None and rely on subsequent polling (kept simple here).
    """
    import requests
    headers = {
        "APCA-API-KEY-ID": broker["key_id"],
        "APCA-API-SECRET-KEY": broker["secret"],
        "Content-Type": "application/json", "Accept": "application/json",
    }
    payload = {
        "symbol": symbol,
        "side": side.lower(),
        "qty": str(int(qty)),
        "type": "market",
        "time_in_force": "day",
        "client_order_id": f"PARENT_{symbol}_{cid[-8:]}",
    }
    log.entry_submit(symbol=symbol, session=session, cid=cid, side=side.upper(),
                     qty=qty, order_type="market", limit_price=None,
                     intended={"tp": intended["tp"], "sl": intended["sl"]},
                     client_order_id=payload["client_order_id"])
    try:
        r = requests.post(f'{broker["base_url"]}/v2/orders', headers=headers, data=json.dumps(payload), timeout=10)
        if not r.ok:
            log.entry_reject(symbol=symbol, session=session, cid=cid,
                             client_order_id=payload["client_order_id"],
                             reason_text=f"alpaca POST /v2/orders -> {r.status_code} {r.text}\n")
            return None
        j = r.json()
        log.entry_ack(symbol=symbol, session=session, cid=cid,
                      client_order_id=payload["client_order_id"],
                      broker_order_id=j.get("id"), status=j.get("status", "pending_new"))
        # Try to fetch immediate fill
        order_id = j.get("id")
        if order_id:
            time.sleep(0.3)  # small wait for fill
            r2 = requests.get(f'{broker["base_url"]}/v2/orders/{order_id}', headers=headers, timeout=8)
            if r2.ok:
                j2 = r2.json()
                if (j2.get("filled_qty") or "0") != "0":
                    fill_qty = int(float(j2.get("filled_qty")))
                    fill_price = float(j2.get("filled_avg_price") or j2.get("limit_price") or price)
                    log.entry_fill(symbol=symbol, session=session, cid=cid,
                                   client_order_id=payload["client_order_id"],
                                   broker_order_id=order_id,
                                   fill_qty=fill_qty, fill_price=fill_price, slippage=0.0)
                    return {"fill_qty": fill_qty, "fill_price": fill_price, "order_id": order_id}
        return None
    except Exception as e:
        log.error(symbol=symbol, session=session, cid=cid, stage="MANAGE",
                  error_text=f"alpaca POST /v2/orders exception: {e}")
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--session", default="AM")
    args, unknown = ap.parse_known_args()

    cfg = load_config(args.config)
    broker = _resolve_broker(cfg)
    SESSION = args.session

    # Guard: market open if force_run not set
    force_run = int(cfg.get("run", {}).get("force_run", int(os.environ.get("FORCE_RUN", "0"))))
    if not force_run:
        if not alpaca_market_open(broker["base_url"], broker["key_id"], broker["secret"]):
            # Exit quietly if market is closed
            print(json.dumps({"session": SESSION, "market_open": False, "type": "HEARTBEAT", "ts": _utcnow()}))
            return

    # Logger
    log_path = _trades_path()
    log = TradeLogger(log_path)
    print(json.dumps({"session": SESSION, "market_open": True, "type": "HEARTBEAT", "ts": _utcnow()}))

    # Inputs
    symbols = read_tickers(args.tickers)

    # Data cadence
    timeframe = cfg.get("timeframe", "15Min")
    history_days = int(cfg.get("history_days", 30))
    feed = cfg.get("feed", "iex")

    # Fetch latest bars
    bars_map = fetch_latest_bars(symbols, timeframe=timeframe, history_days=history_days, feed=feed)

    # Iterate symbols → signal → gate → entry
    qty = int(cfg.get("qty", 1))
    for sym in symbols:
        df = bars_map.get(sym)
        if df is None or df.empty:
            continue

        # Attach verifiers: ADX, RSI, EMA slope, dollar_vol
        df = attach_verifiers(df, cfg)

        # Build simple signal snapshot on last bar
        last = df.iloc[-1]
        cross = 1 if float(last.get("ema_fast", 0)) > float(last.get("ema_slow", 0)) else ( -1 if float(last.get("ema_fast", 0)) < float(last.get("ema_slow", 0)) else 0 )
        row = {
            "symbol": sym,
            "session": SESSION,
            "cid": _build_cid(sym),
            "tf": timeframe,
            "cross": cross,
            "ref_bar_ts": str(df.index[-1]),
            "last_close": float(last.get("close", 0.0)),
            "adx": float(last.get("adx", 0.0)),
            "ema_slope_pct": float(last.get("ema_slope_pct", 0.0)),
            "dollar_vol_avg": float(last.get("dollar_vol_avg", 0.0)),
        }
        # include RSI in the signal snapshot for clarity
        if "rsi" in df.columns:
            row["rsi"] = float(last.get("rsi", 50.0))

        _log_signal(log, row)

        # Gate decision
        ok, reasons = explain_long_gate(last, cfg)
        if not ok:
            _log_gate(log, sym, row["cid"], SESSION, "BLOCK", reasons)
            continue

        # Only go long on cross >= 0 (your logic blocks some cases on cross=0)
        if cross < 0:
            _log_gate(log, sym, row["cid"], SESSION, "BLOCK", ["bearish cross"])
            continue

        # If you keep a "PASS on cross=0" rule, reflect it:
        if cross == 0:
            _log_gate(log, sym, row["cid"], SESSION, "PASS", ["cross=0"])
            continue

        # Entry price = last_close, compute intended brackets
        intended = _calc_tp_sl_from_cfg(cfg, row["last_close"], atr=last.get("atr", None))

        # Submit entry
        filled = _maybe_submit_entry(
            symbol=sym, side="BUY", qty=qty, price=row["last_close"],
            intended=intended, log=log, broker=broker, cid=row["cid"], session=SESSION
        )

        # If filled immediately, place OCO with optional guard & idempotency
        if filled and filled.get("fill_qty", 0) >= 1:
            _place_oco_after_fill(
                symbol=sym,
                fill_qty=int(filled["fill_qty"]),
                intended=intended,
                cid=row["cid"],
                log=log,
                session=SESSION,
                broker=broker,
            )

    # Done
    return

if __name__ == "__main__":
    main()
