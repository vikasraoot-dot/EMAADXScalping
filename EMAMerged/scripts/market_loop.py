# EMAMerged/scripts/market_loop.py
from __future__ import annotations
import os, sys, time, subprocess
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    # py<3.9 fallback (not expected for your runner, but harmless)
    from backports.zoneinfo import ZoneInfo  # type: ignore

ET  = ZoneInfo("US/Eastern")
UTC = ZoneInfo("UTC")

OPEN_ET  = (9, 30)   # 09:30
CLOSE_ET = (16, 0)   # 16:00

def now_et() -> datetime:
    return datetime.now(ET)

def is_weekday(d: datetime | None = None) -> bool:
    d = d or now_et()
    return d.weekday() < 5

def market_open_et(d: datetime | None = None) -> datetime:
    d = d or now_et()
    return d.replace(hour=OPEN_ET[0], minute=OPEN_ET[1], second=0, microsecond=0)

def market_close_et(d: datetime | None = None) -> datetime:
    d = d or now_et()
    return d.replace(hour=CLOSE_ET[0], minute=CLOSE_ET[1], second=0, microsecond=0)

def seconds_until_next_5m(d_utc: datetime | None = None) -> int:
    """
    Sleep to the next 5-minute boundary in UTC. This aligns loops to :00/:05/:10…
    Min 1s to keep loops tight if called right after a boundary.
    """
    d = (d_utc or datetime.now(UTC)).replace(tzinfo=UTC)
    mins = d.minute
    next_min = (mins - (mins % 5)) + 5
    carry = 0
    if next_min >= 60:
        next_min -= 60
        carry = 1
    target = d.replace(minute=next_min, second=0, microsecond=0) + timedelta(hours=carry)
    secs = int((target - d).total_seconds())
    return max(secs, 1)

def run_once():
    """
    Invoke the existing single-shot runner with the same interpreter.
    We prefer the tickers file by default; symbols can still be passed via env.
    """
    py = sys.executable
    repo_root = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
    cfg = os.environ.get("EMA_CONFIG", "EMAMerged/config.yaml")
    tickers = os.environ.get("EMA_TICKERS_FILE", "EMAMerged/tickers.txt")
    symbols = os.environ.get("SYMBOLS", "").strip()
    dry = os.environ.get("DRY_RUN", "1")
    force = os.environ.get("FORCE_RUN", "0")

    args = [py, "-X", "utf8", "-u", "EMAMerged/scripts/live_paper_loop.py", "--config", cfg]
    if symbols:
        args += ["--symbols", symbols]
    else:
        args += ["--tickers", tickers]
    args += ["--dry-run", dry, "--force-run", force]

    print(f"[loop] launching live_paper_loop (symbols={'custom' if symbols else 'file'}, dry={dry}, force={force})", flush=True)
    proc = subprocess.run(args, cwd=repo_root)
    print(f"[loop] live_paper_loop exit code={proc.returncode}", flush=True)

def main():
    fast_loop = os.environ.get("FAST_LOOP", "0") == "1"

    print("[loop] starting EMAADXScalping AM loop (15m strategy; check every 5m)", flush=True)
    while True:
        d = now_et()
        if not is_weekday(d):
            # weekend: sleep to next 5m boundary, keep logs flowing
            secs = seconds_until_next_5m()
            print(f"[loop] weekend; sleep {secs}s", flush=True)
            time.sleep(secs)
            continue

        mo, mc = market_open_et(d), market_close_et(d)
        if d < mo:
            # Pre-open: sleep until the next 5m tick, or to open if closer
            secs_to_open = int((mo - d).total_seconds())
            secs_5m = seconds_until_next_5m()
            secs = min(secs_to_open, secs_5m)
            print(f"[loop] pre-open {d.strftime('%H:%M:%S %Z')}; sleep {secs}s", flush=True)
            time.sleep(secs)
            continue

        if d >= mc:
            print(f"[loop] reached market close {mc.strftime('%H:%M %Z')} → exiting loop", flush=True)
            break

        # In RTH: run once, then sleep to the next 5m boundary (or 1s if FAST_LOOP)
        try:
            run_once()
        except Exception as e:
            print(f"[loop] ERROR: {e}", flush=True)

        secs = 1 if fast_loop else seconds_until_next_5m()
        print(f"[loop] sleep {secs}s → {'FAST' if fast_loop else 'next 5m boundary'}", flush=True)
        time.sleep(secs)

if __name__ == "__main__":
    # ensure immediate flushing
    try:
        import functools as _f
        print = _f.partial(print, flush=True)  # type: ignore
    except Exception:
        pass
    main()
