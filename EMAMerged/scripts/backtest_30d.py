#!/usr/bin/env python3
from __future__ import annotations
import os, sys, argparse
import pandas as pd

# --- Add repo root to sys.path so "EMAMerged" imports resolve when run as a script ---
_THIS = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from EMAMerged.src.utils import load_config, read_tickers
from EMAMerged.src.data import (
    get_alpaca_bars, filter_rth, drop_unclosed_last_bar,
)
from EMAMerged.src.strategy import compute_indicators, crossover
from EMAMerged.src.filters import long_ok

def _env(name: str, default: str = "") -> str:
    v = os.environ.get(name)
    return v if v is not None else default

def _alpaca_creds():
    key = _env("ALPACA_API_KEY") or _env("ALPACA_KEY")
    secret = _env("ALPACA_API_SECRET") or _env("ALPACA_SECRET")
    base = (_env("APCA_API_BASE_URL") or _env("ALPACA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
    if not key or not secret:
        raise RuntimeError("Missing ALPACA_API_KEY/ALPACA_API_SECRET")
    return base, key, secret

def load_bars_for_symbol(
    sym: str,
    cfg: dict,
    days: int,
    timeframe_override: str | None = None,
    limit_override: int | None = None,
    rth_only_override: bool | None = None,
) -> pd.DataFrame:
    """
    Fetch bars from Alpaca, optionally overriding timeframe/limit and enforcing RTH.
    """
    # creds are read only to keep parity with live; get_alpaca_bars uses key/secret
    base, key, secret = _alpaca_creds()
    tf = timeframe_override or cfg.get("timeframe", "5Min")
    limit = int(limit_override or cfg.get("bar_limit", 500))
    feed = cfg.get("feed", "iex")

    df = get_alpaca_bars(
        key, secret, sym,
        timeframe=tf,
        history_days=days,
        bar_limit=limit,
        feed=feed,
    )
    if df.empty:
        return df

    # RTH filter (force on by default for backtests)
    if rth_only_override is None:
        rth_flag = bool(cfg.get("rth_only", True))
    else:
        rth_flag = bool(rth_only_override)

    if rth_flag:
        tz = cfg.get("timezone", "US/Eastern")
        df = filter_rth(
            df,
            tz_name=tz,
            rth_start=cfg.get("rth_start", "09:30"),
            rth_end=cfg.get("rth_end", "15:55"),
            allowed_windows=cfg.get("entry_windows"),  # respect windows if present
        )

    # Drop currently-forming last bar to match live logic
    df = drop_unclosed_last_bar(df, timeframe=tf)
    return df

def simulate_long_only(df: pd.DataFrame, cfg: dict, start_cash=10_000.0, risk_per_trade=0.01):
    """
    Simple, deterministic long-only backtest:
    - Entry: previous bar signals cross==1 AND long_ok(prev, cfg) → enter at next bar OPEN.
    - Exit:  cross==-1 on previous bar → exit next OPEN
             plus bracket-style ATR SL/TP if present in cfg['brackets'].
    - Sizing: risk% of equity, risk-per-share = max(ATR, 1% of price).
    """
    if df is None or df.empty or len(df) < 30:
        return dict(trades=0, gross=0.0, net=0.0, win_rate=0.0, ret_pct=0.0, equity=start_cash)

    # Build indicators exactly like live (adds ema_fast/slow, atr, rsi, adx, etc.)
    df = co
