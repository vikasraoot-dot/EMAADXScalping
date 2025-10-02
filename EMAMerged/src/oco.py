# === EMAMerged/src/oco.py ===
from __future__ import annotations
import time, json
from typing import Optional, Dict, Any, Tuple, List
import requests

INSUFFICIENT_QTY_TEXT = "insufficient qty available for order"

def _hdrs(key_id: str, secret: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": key_id,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def _get_position(base_url: str, headers: Dict[str, str], symbol: str) -> Tuple[float, Optional[Dict[str, Any]]]:
    """
    Returns (qty_float, position_json|None). If no position, qty=0, pos=None.
    """
    r = requests.get(f"{base_url}/v2/positions/{symbol}", headers=headers, timeout=8)
    if r.status_code == 404:
        return 0.0, None
    r.raise_for_status()
    pos = r.json()
    # Alpaca returns strings for qty; be robust:
    qty = float(pos.get("qty", pos.get("qty_available", 0)) or 0)
    return qty, pos

def _get_open_orders(base_url: str, headers: Dict[str, str], symbol: str) -> List[Dict[str, Any]]:
    r = requests.get(
        f"{base_url}/v2/orders",
        params={"status": "open", "symbols": symbol},
        headers=headers,
        timeout=8,
    )
    r.raise_for_status()
    data = r.json()
    return data if isinstance(data, list) else (data or [])

def _sum_reduce_only_sells(orders: List[Dict[str, Any]], symbol: str) -> float:
    s = 0.0
    for o in orders:
        if o.get("symbol") == symbol and o.get("side") == "sell" and bool(o.get("reduce_only")):
            try:
                s += float(o.get("qty") or o.get("notional") or 0)
            except Exception:
                pass
    return s

def _post_oco(
    base_url: str,
    headers: Dict[str, str],
    symbol: str,
    qty: float,
    tp: float,
    sl: float,
    tif: str = "gtc",
) -> requests.Response:
    # qty must be a string int for shares
    payload = {
        "symbol": symbol,
        "side": "sell",
        "qty": str(int(qty)),
        "type": "limit",          # required, but TP carries the actual limit price
        "time_in_force": tif,
        "order_class": "oco",
        "take_profit": {"limit_price": float(tp)},
        "stop_loss":   {"stop_price":  float(sl)},
        "reduce_only": True,
    }
    return requests.post(f"{base_url}/v2/orders", headers=headers, data=json.dumps(payload), timeout=8)

def ensure_oco_for_long(
    *,
    symbol: str,
    intended_qty: float,
    tp_level: float,
    sl_level: float,
    base_url: str,
    key_id: str,
    secret: str,
    logger,                 # TradeLogger
    cid: str,
    session: str = "AM",
    backoff_sec: float = 0.5,
) -> Dict[str, Any]:
    """
    Idempotently ensure a single atomic OCO (TP+SL) exists for a long position.
    Optional guard: only attempt if position.qty >= 1 and available >= 1.
    available = position.qty - sum(qty of open reduce-only sell orders)

    Returns dict with keys:
      status: one of {"exists","posted","skip_no_pos","skip_unavailable","error"}
      details: optional extra info
    """
    headers = _hdrs(key_id, secret)

    # 1) Fetch current position
    pos_qty, pos_json = _get_position(base_url, headers, symbol)
    if pos_qty < 1.0:
        logger.snapshot(symbol=symbol, session=session, cid=cid, stage="OCO_CHECK",
                        note="no_position", pos_qty=pos_qty)
        return {"status": "skip_no_pos", "details": {"pos_qty": pos_qty}}

    # 2) Check open reduce-only sells (idempotency + compute availability)
    open_orders = _get_open_orders(base_url, headers, symbol)
    held_reduce = _sum_reduce_only_sells(open_orders, symbol)
    available = pos_qty - held_reduce

    logger.snapshot(symbol=symbol, session=session, cid=cid, stage="OCO_CHECK",
                    pos_qty=pos_qty, held_reduce=held_reduce, available=available)

    # If we already have a reduce-only sell working, we consider OCO "exists"
    if held_reduce >= 1.0:
        logger.oco_ack(symbol=symbol, session=session, cid=cid,
                       note="reduce_only_sell_already_open", held_reduce=held_reduce)
        return {"status": "exists", "details": {"held_reduce": held_reduce, "available": available}}

    # Optional guard: only post if available >= 1 share
    if available < 1.0:
        logger.gate(symbol=symbol, session=session, cid=cid, decision="BLOCK",
                    reasons=[f"available {available:.2f} < 1.0 (reduce-only holds {held_reduce:.2f})"])
        return {"status": "skip_unavailable", "details": {"available": available, "held_reduce": held_reduce}}

    # Determine the qty to bracket (min of intended and available)
    post_qty = max(1, int(min(intended_qty, available)))

    # 3) Post atomic OCO
    logger.oco_submit(symbol=symbol, session=session, cid=cid,
                      tp={"level": tp_level}, sl={"level": sl_level}, qty=post_qty)
    r = _post_oco(base_url, headers, symbol, post_qty, tp_level, sl_level)

    # 403 “insufficient qty available … held_for_orders” → short backoff + retry once
    if r.status_code == 403 and INSUFFICIENT_QTY_TEXT in (r.text or "").lower():
        time.sleep(backoff_sec)
        # Re-check open orders & availability before retry
        open_orders = _get_open_orders(base_url, headers, symbol)
        held_reduce = _sum_reduce_only_sells(open_orders, symbol)
        available = pos_qty - held_reduce
        logger.snapshot(symbol=symbol, session=session, cid=cid, stage="OCO_RETRY_CHECK",
                        pos_qty=pos_qty, held_reduce=held_reduce, available=available)
        if available < 1.0:
            logger.error(symbol=symbol, session=session, cid=cid, stage="OCO_SUBMIT",
                         error_code="SKIP_AFTER_HELD", error_text="available<1 after backoff")
            return {"status": "skip_unavailable", "details": {"available": available}}

        r = _post_oco(base_url, headers, symbol, min(post_qty, int(available)), tp_level, sl_level)

    if r.ok:
        try:
            j = r.json()
        except Exception:
            j = {"raw": r.text}
        logger.oco_ack(symbol=symbol, session=session, cid=cid,
                       broker_order_id=j.get("id"), status=j.get("status"))
        return {"status": "posted", "details": j}

    # Failed
    logger.error(symbol=symbol, session=session, cid=cid, stage="OCO_SUBMIT",
                 error_code="SUBMIT_FAILED", error_text=r.text, http=r.status_code)
    return {"status": "error", "details": {"code": r.status_code, "body": r.text}}
