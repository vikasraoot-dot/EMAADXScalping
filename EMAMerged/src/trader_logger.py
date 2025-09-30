from __future__ import annotations
import os, json, threading, datetime as dt
from typing import Any, Dict

ISO_UTC = "%Y-%m-%dT%H:%M:%SZ"

def utc_now() -> str:
    return dt.datetime.utcnow().strftime(ISO_UTC)

class TradeLogger:
    """
    JSONL logger for full trade lifecycle & telemetry.
    One event per line; safe for long-running processes (line-buffered).
    """
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # line-buffered, UTF-8
        self._fp = open(path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def _emit(self, obj: Dict[str, Any]):
        if "ts" not in obj:
            obj["ts"] = utc_now()
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._fp.write(line + "\n")

    # --- lifecycle ---
    def signal(self, **kw):        kw.setdefault("type","SIGNAL");        self._emit(kw)
    def entry_submit(self, **kw):  kw.setdefault("type","ENTRY_SUBMIT");  self._emit(kw)
    def entry_ack(self, **kw):     kw.setdefault("type","ENTRY_ACK");     self._emit(kw)
    def entry_reject(self, **kw):  kw.setdefault("type","ENTRY_REJECT");  self._emit(kw)
    def entry_fill(self, **kw):    kw.setdefault("type","ENTRY_FILL");    self._emit(kw)
    def oco_submit(self, **kw):    kw.setdefault("type","OCO_SUBMIT");    self._emit(kw)
    def oco_ack(self, **kw):       kw.setdefault("type","OCO_ACK");       self._emit(kw)
    def exit_fill(self, **kw):     kw.setdefault("type","EXIT_FILL");     self._emit(kw)
    def cancel(self, **kw):        kw.setdefault("type","CANCEL");        self._emit(kw)
    # --- telemetry ---
    def snapshot(self, **kw):      kw.setdefault("type","SNAPSHOT");      self._emit(kw)
    def error(self, **kw):         kw.setdefault("type","ERROR");         self._emit(kw)
    def eod_summary(self, **kw):   kw.setdefault("type","EOD_SUMMARY");   self._emit(kw)

    def close(self):
        try: self._fp.close()
        except Exception: pass
