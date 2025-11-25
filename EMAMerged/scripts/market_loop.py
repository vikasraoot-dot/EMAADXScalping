# EMAMerged/scripts/market_loop.py
from __future__ import annotations
import os, sys, time, subprocess
from datetime import datetime, timedelta
try:
    from zoneinfo import ZoneInfo
except Exception:
    from backports.zoneinfo import ZoneInfo  # type: ignore
import yaml  # added

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
    Sleep to the next 5-minute boundary in UTC. Aligns loops to :00/:05/:10…
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

    print(f"[loop] invoking: {py} {' '.join(args[1:])}", flush=True)
    rc = subprocess.run(args, check=False).returncode
    print(f"[loop] live_paper_loop exit code={rc}", flush=True)

def _load_cfg(path="EMAMerged/config.yaml"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def main():
    fast_loop = os.environ.get("FAST_LOOP", "0") == "1"
    cfg_path = os.environ.get("EMA_CONFIG", "EMAMerged/config.yaml")
    cfg = _load_cfg(cfg_path)
    flatten_mins = int(cfg.get("flatten_minutes_before_close", 5))
    flattened_today = False

    print("[loop] starting EMAADXScalping AM loop (15m strategy; check every 5m)", flush=True)
    while True:
        d = now_et()
        if not is_weekday(d):
            secs = seconds_until_next_5m()
            print(f"[loop] weekend; sleep {secs}s", flush=True)
            time.sleep(secs)
            continue

        mo, mc = market_open_et(d), market_close_et(d)
        if d < mo:
            secs_to_open = int((mo - d).total_seconds())
            secs_5m = seconds_until_next_5m()
            secs = min(secs_to_open, secs_5m)
            print(f"[loop] pre-open {d.strftime('%H:%M:%S %Z')}; sleep {secs}s", flush=True)
            time.sleep(secs)
            continue

        # EOD flatten once flatten_mins before close
        if not flattened_today:
            mins_to_close = (mc - d).total_seconds() / 60.0
            if mins_to_close <= flatten_mins:
                print(f"[loop] within {flatten_mins}m of close → EOD flatten", flush=True)
                try:
                    subprocess.run([sys.executable, "-u", "EMAMerged/scripts/flat_eod.py"], check=False)
                    # NEW: Run PnL Report after flattening
                    print("[loop] Generating Daily PnL Report...", flush=True)
                    subprocess.run([sys.executable, "-u", "EMAMerged/scripts/report_pnl.py"], check=False)
                    flattened_today = True
                except Exception as e:
                    print(f"[loop] EOD flatten/report error: {e}", flush=True)

        if d >= mc:
            print(f"[loop] reached market close {mc.strftime('%H:%M %Z')} → exiting loop", flush=True)
            break

        try:
            run_once()
        except Exception as e:
            print(f"[loop] ERROR: {e}", flush=True)

        secs = 1 if fast_loop else seconds_until_next_5m()
        print(f"[loop] sleep {secs}s → {'FAST' if fast_loop else 'next 5m boundary'}", flush=True)
        time.sleep(secs)

if __name__ == "__main__":
    try:
        import functools as _f
        print = _f.partial(print, flush=True)  # type: ignore
    except Exception:
        pass
    main()
