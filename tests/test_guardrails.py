# tests/test_guardrails.py
# Exercises: EOD flatten, 422 cooldown, and breakeven bump.
# All network/Alpaca calls are mocked. Safe to run in CI.

from __future__ import annotations
import types
import pandas as pd
import pytest

#
# ---------- helpers ----------
#

def _utc_now():
    return pd.Timestamp.utcnow().tz_localize("UTC")

def _make_bars(periods=3, tf_minutes=15, close=100.0, atr=0.8):
    """Small 15m bar DataFrame with ATR column present."""
    end = _utc_now()
    idx = pd.date_range(end=end, periods=periods, freq=f"{tf_minutes}min", tz="UTC")
    df = pd.DataFrame(
        {
            "open":  [close-0.5]*periods,
            "high":  [close+0.5]*periods,
            "low":   [close-1.0]*periods,
            "close": [close]*periods,
            "volume":[10000]*periods,
            "atr":   [atr]*(periods-1) + [atr],
        },
        index=idx
    )
    return df

#
# ---------- Test 1: EOD flatten calls ----------
#

def test_guardrails_eod_flatten_invokes_cancel_and_close(monkeypatch):
    # Import module under test
    import EMAMerged.scripts.flat_eod as flat

    called = {"cancel": 0, "close": 0}

    def fake_cancel(base, key, sec):
        called["cancel"] += 1
        return {}

    def fake_close(base, key, sec):
        called["close"] += 1
        return {}

    # monkeypatch credentials
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Patch at the import site (flat_eod module)
    monkeypatch.setattr(flat, "cancel_all_orders", fake_cancel)
    monkeypatch.setattr(flat, "close_all_positions", fake_close)

    flat.main()

    assert called["cancel"] == 1, "EOD flatten should cancel all open orders once"
    assert called["close"] == 1,  "EOD flatten should close all positions once"

#
# ---------- Test 2: 422 cooldown behavior ----------
#

def test_guardrails_422_cooldown_sets_and_skips(monkeypatch, capsys):
    """
    Force a 422 on bracket submit.
    Verify the symbol is put on a 15m cooldown and subsequent attempt skips.
    """
    import EMAMerged.scripts.live_paper_loop as live

    sym = "TEST"

    cfg = {
        "timezone": "US/Eastern",
        "timeframe": "15Min",
        # disable RTH path; orthogonal to this test
        "rth_only": False,
        "rth_start": "09:30",
        "rth_end": "15:55",
        "brackets": {"enabled": True, "atr_mult_sl": 1.0, "take_profit_r": 1.2},
        "entry_cutoff_min": 0,   # disable cutoff for this test
        "history_days": 1,
        "bar_limit": 500,
        "feed": "iex",
        "qty": 1,
    }

    # --- Minimal mocks to jump straight to order placement ---
    # Bypass bar-prep entirely (avoids tz-localize paths)
    monkeypatch.setattr(live, "get_alpaca_bars", lambda *a, **kw: pd.DataFrame())
    # Keep these as no-ops (not used because df is empty, but safe)
    monkeypatch.setattr(live, "filter_rth", lambda df, **kw: df)
    monkeypatch.setattr(live, "drop_unclosed_last_bar", lambda df, tf: df)

    # Pretend signal & gating are always long/ok
    monkeypatch.setattr(live, "compute_indicators", lambda df, cfg: df)
    monkeypatch.setattr(live, "crossover", lambda df: 1)
    monkeypatch.setattr(live, "long_ok", lambda df, cfg: True)
    monkeypatch.setattr(live, "explain_long_gate", lambda df, cfg: {"ok": True})

    # No existing positions/orders
    monkeypatch.setattr(live, "get_positions", lambda *a, **kw: {})
    monkeypatch.setattr(live, "get_open_orders", lambda *a, **kw: [])

    # creds env for _alpaca_creds
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Force a 422 on submit
    def boom(*a, **kw): raise Exception("422 Unprocessable Entity")
    monkeypatch.setattr(live, "submit_bracket_order", boom)

    args = types.SimpleNamespace(dry_run=0)
    live.manage_symbol(sym, cfg, args)

    assert sym in live._ERROR_COOLDOWN, "422 should place symbol on cooldown"

    # second call should skip due to cooldown
    live.manage_symbol(sym, cfg, args)
    out = capsys.readouterr().out
    assert "on error cooldown" in out

#
# ---------- Test 3: Breakeven stop bump ----------
#

def test_guardrails_breakeven_bump_updates_stop(monkeypatch):
    """
    At ≥0.5R profit, bump stop to entry + 2 ticks (0.02 by default).
    """
    import EMAMerged.scripts.live_paper_loop as live

    # Provide creds (for _alpaca_creds in helper)
    monkeypatch.setenv("ALPACA_API_KEY", "k")
    monkeypatch.setenv("ALPACA_API_SECRET", "s")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

    # Mock list_open_orders & patch_order AT THE IMPORT SITE
    stop_order = {"id": "abc123", "symbol": "XYZ", "side": "sell", "type": "stop", "stop_price": 100.00}

    def fake_list(base, key, secret, symbols=None):
        return [stop_order]

    patched = {"patched": False, "stop_price": None}

    def fake_patch(base, key, secret, order_id, **fields):
        patched["patched"] = True
        patched["stop_price"] = fields.get("stop_price")
        return {"id": order_id, **fields}

    monkeypatch.setattr(live, "list_open_orders", fake_list)
    monkeypatch.setattr(live, "patch_order", fake_patch)

    # Config for breakeven
    cfg = {
        "breakeven": {"enabled": True, "trigger_r": 0.5, "bump_ticks": 2, "tick_size": 0.01}
    }

    # Entry/last so that r_mult = (last-entry)/r_dist >= 0.5
    entry = 100.00
    r_dist = 0.08  # so +0.5R = +0.04
    last = 100.10  # >= entry + 0.5R ⇒ triggers bump

    # Call the internal helper directly
    live._maybe_breakeven_bump("XYZ", cfg, entry, r_dist, last)

    assert patched["patched"] is True, "patch_order should be called once"
    assert abs(patched["stop_price"] - (entry + 0.02)) < 1e-9, "stop should bump to entry + 2 ticks (0.02)"
