from __future__ import annotations
import os, json, threading, datetime as dt
from typing import Any, Dict, Optional

# ISO timestamp in UTC for stable ordering across machines
_ISO = "%Y-%m-%dT%H:%M:%SZ"

def _ts() -> str:
    return dt.datetime.now(dt.UTC).strftime(_ISO)

class TradeLogger:
    """
    Line-buffered JSONL logger with a compact, auditable schema.
    One event per line. Designed to be safe to call from multiple places.

    Enhancement: mirror each JSONL line to stdout so CI (GitHub Actions) shows
    live progress, without changing the on-disk audit format.
    """
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # line-buffered
        self._fp = open(path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def _write(self, obj: Dict[str, Any]):
        obj.setdefault("ts", _ts())
        line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            # Always try to write the file log first (audit source of truth)
            try:
                self._fp.write(line + "\n")
            except Exception:
                # Never let logging crash trading; still try to emit to console
                pass
            # Mirror to console so it appears in GitHub Actions live logs
            try:
                print(line, flush=True)
            except Exception:
                # Ignore console failures to keep trading resilient
                pass

    # ---- Emitters (normalized "type" values) ----
    def heartbeat(self, **kw): kw.setdefault("type","HEARTBEAT"); self._write(kw)
    def signal(self, **kw): kw.setdefault("type","SIGNAL"); self._write(kw)
    def gate(self, **kw): kw.setdefault("type","GATE"); self._write(kw)

    def entry_submit(self, **kw): kw.setdefault("type","ENTRY_SUBMIT"); self._write(kw)
    def entry_ack(self, **kw): kw.setdefault("type","ENTRY_ACK"); self._write(kw)
    def entry_reject(self, **kw): kw.setdefault("type","ENTRY_REJECT"); self._write(kw)
    def entry_fill(self, **kw): kw.setdefault("type","ENTRY_FILL"); self._write(kw)

    def oco_submit(self, **kw): kw.setdefault("type","OCO_SUBMIT"); self._write(kw)
    def oco_ack(self, **kw): kw.setdefault("type","OCO_ACK"); self._write(kw)
    def exit_fill(self, **kw): kw.setdefault("type","EXIT_FILL"); self._write(kw)
    def cancel(self, **kw): kw.setdefault("type","CANCEL"); self._write(kw)

    def error(self, **kw): kw.setdefault("type","ERROR"); self._write(kw)
    def snapshot(self, **kw): kw.setdefault("type","SNAPSHOT"); self._write(kw)

    def eod_summary(self, **kw): kw.setdefault("type","EOD_SUMMARY"); self._write(kw)

    def close(self):
        try:
            self._fp.close()
        except Exception:
            pass
