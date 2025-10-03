from __future__ import annotations
import os, time, uuid, datetime as dt
from typing import Optional, Dict, Any, Tuple, List
import requests

from EMAMerged.src.trade_logger import TradeLogger

# Reads standard Alpaca envs already used by your code
ALPACA_BASE = os.getenv("APCA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
ALPACA_KEY  = os.getenv("ALPACA_KEY") or os.getenv("APCA_API_KEY_ID") or ""
ALPACA_SEC  = os.getenv("ALPACA_SECRET") or os.getenv("APCA_API_SECRET_KEY") or ""

def _headers() -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SEC,
        "Content-Type": "application/json",
    }

def _now_iso() -> str:
    # tz-aware ISO timestamp (avoids deprecated UTC-naive calls)
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

def _req(method: str, path: str, ok=(200, 201), **kw):
    url = f"{ALPACA_BASE}{path}"
    r = requests.request(method, url, headers=_headers(), timeout=20, **kw)
    if r.status_code not in ok:
        raise RuntimeError(f"alpaca {method} {path} -> {r.status_code} {r.text}")
    return r.json() if r.text else {}

def get_positions(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    if symbol:
        try:
            j = _req("GET", f"/v2/positions/{symbol.upper()}", ok=(200, 404))
            return [j] if isinstance(j, dict) and j.get("symbol") else []
        except Exception:
            return []
    return _req("GET", "/v2/positions") or []

def get_order(order_id: str) -> Dict[str, Any]:
    return _req("GET", f"/v2/orders/{order_id}")

def list_open_orders() -> List[Dict[str, Any]]:
    return _req("GET", "/v2/orders?status=open&nested=true")

def list_activities_fills(after_iso: Optional[str] = None) -> List[Dict[str, Any]]:
    path = "/v2/account/activities/FILL"
    if after_iso:
        path += f"?after={after_iso}"
    return _req("GET", path)

def submit_market_order(symbol: str, side: str, qty: int, client_order_id: str) -> Dict[str, Any]:
    payload = {
        "symbol": symbol.upper(),
        "side": side.lower(),
        "type": "market",
        "time_in_force": "day",
        "qty": qty,
        "client_order_id": client_order_id,
    }
    return _req("POST", "/v2/orders", json=payload)

def submit_limit_order(symbol: str, side: str, qty: int, limit_price: float, client_order_id: str) -> Dict[str, Any]:
    payload = {
        "symbol": symbol.upper(),
        "side": side.lower(),
        "type": "limit",
        "time_in_force": "day",
        "qty": qty,
        "limit_price": float(limit_price),
        "client_order_id": client_order_id,
    }
    return _req("POST", "/v2/orders", json=payload)

def submit_take_profit(symbol: str, qty: int, limit_price: float, client_order_id: str) -> Dict[str, Any]:
    # for LONG exits: SELL limit
    return submit_limit_order(symbol, "sell", qty, limit_price, client_order_id)

def submit_stop_loss(symbol: str, qty: int, stop_price: float, client_order_id: str) -> Dict[str, Any]:
    payload = {
        "symbol": symbol.upper(),
        "side": "sell",
        "type": "stop",
        "time_in_force": "day",
        "qty": qty,
        "stop_price": float(stop_price),
        "client_order_id": client_order_id,
    }
    return _req("POST", "/v2/orders", json=payload)

def wait_for_filled(order_id: str, timeout_sec: int = 90, poll_sec: float = 0.5) -> Tuple[bool, Dict[str, Any]]:
    deadline = time.time() + timeout_sec
    last = {}
    while time.time() < deadline:
        try:
            j = get_order(order_id)
            last = j
            st = (j.get("status") or "").upper()
            if st in ("FILLED", "PARTIALLY_FILLED"):
                return True, j
            if st in ("CANCELED", "EXPIRED", "REJECTED", "SUSPENDED", "DONE_FOR_DAY"):
                return False, j
        except Exception:
            pass
        time.sleep(poll_sec)
    return False, last

def _coid(prefix: str, sym: str) -> str:
    return f"{prefix}_{sym}_{uuid.uuid4().hex[:8]}"

def execute_long_with_oco(
    logger: TradeLogger,
    symbol: str,
    qty: int,
    intended_tp: float,
    intended_sl: float,
    session: str,
    cid: str,
    prefer_limit_entry: bool = False,
    limit_px: Optional[float] = None,
) -> Dict[str, Any]:
    """
    SAFE SEQUENCING:
      1) Submit parent entry (market or limit)
      2) Wait for fill
      3) Submit OCO exits AS TWO CHILD ORDERS (TP limit + SL stop) — robust and explicit
      4) Emit lifecycle logs at each step
    """
    # --- 1) entry submit
    coid_parent = _coid("PARENT", symbol)
    try:
        if prefer_limit_entry and limit_px is not None:
            logger.entry_submit(
                symbol=symbol, session=session, cid=cid, side="BUY",
                qty=qty, order_type="limit", limit_price=limit_px,
                intended={"tp": intended_tp, "sl": intended_sl},
                client_order_id=coid_parent,
            )
            ack = submit_limit_order(symbol, "buy", qty, limit_px, coid_parent)
        else:
            logger.entry_submit(
                symbol=symbol, session=session, cid=cid, side="BUY",
                qty=qty, order_type="market", limit_price=None,
                intended={"tp": intended_tp, "sl": intended_sl},
                client_order_id=coid_parent,
            )
            ack = submit_market_order(symbol, "buy", qty, coid_parent)
        logger.entry_ack(
            symbol=symbol, session=session, cid=cid,
            client_order_id=coid_parent,
            broker_order_id=ack.get("id"), status=ack.get("status"),
        )
    except Exception as e:
        logger.entry_reject(
            symbol=symbol, session=session, cid=cid,
            client_order_id=coid_parent, reason_text=str(e),
        )
        raise

    # --- 2) wait for fill
    ok, fill = wait_for_filled(ack["id"])
    if not ok:
        # Emit error & bail; caller can decide cooldown
        logger.error(
            symbol=symbol, session=session, cid=cid, stage="ENTRY_WAIT",
            error_code="NOT_FILLED", error_text=str(fill),
        )
        return {"status": "entry_not_filled", "detail": fill}

    avg_entry = float(fill.get("filled_avg_price") or fill.get("limit_price") or 0.0)
    fqty = int(fill.get("filled_qty") or fill.get("qty") or qty)
    logger.entry_fill(
        symbol=symbol, session=session, cid=cid,
        client_order_id=coid_parent, broker_order_id=fill.get("id"),
        fill_qty=fqty, fill_price=avg_entry, slippage=0.0,
    )

    # --- 3) submit OCO exits AFTER entry fill
    oco_client = _coid("OCO", symbol)
    logger.oco_submit(
        symbol=symbol, session=session, cid=cid,
        parent_broker_order_id=fill.get("id"),
        tp={"level": intended_tp, "order_type": "limit"},
        sl={"level": intended_sl, "order_type": "stop"},
        oco_client_id=oco_client,
    )
    tp_id = sl_id = None
    try:
        tp_ack = submit_take_profit(symbol, fqty, intended_tp, _coid("TP", symbol))
        sl_ack = submit_stop_loss(symbol, fqty, intended_sl, _coid("SL", symbol))
        tp_id = tp_ack.get("id")
        sl_id = sl_ack.get("id")
        logger.oco_ack(
            symbol=symbol, session=session, cid=cid,
            oco_client_id=oco_client, tp_order_id=tp_id,
            sl_order_id=sl_id, status="ACCEPTED",
        )
    except Exception as e:
        logger.error(
            symbol=symbol, session=session, cid=cid, stage="OCO_SUBMIT",
            error_code="SUBMIT_FAILED", error_text=str(e),
        )

    return {
        "status": "ok",
        "entry_order_id": fill.get("id"),
        "avg_entry": avg_entry,
        "qty": fqty,
        "tp_order_id": tp_id,
        "sl_order_id": sl_id,
    }
