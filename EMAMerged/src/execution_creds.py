# EMAMerged/src/execution_creds.py
from __future__ import annotations
import os
import requests

# Treat these as placeholders (don't write them into env)
_PLACEHOLDERS = {"", None, "YOUR_KEY_ID", "YOUR_SECRET"}

def _set_env_if_absent(name: str, value: str | None) -> None:
    """Do not clobber env if already set via GitHub Secrets."""
    if os.getenv(name):
        return
    if value not in _PLACEHOLDERS:
        os.environ[name] = value  # type: ignore[arg-type]

def configure_alpaca(base_url: str | None, key: str | None, secret: str | None, do_check: bool = True) -> None:
    """
    Env-first shim:
      - If GitHub Actions exported APCA_* envs, we KEEP them.
      - If not, we fall back to values from config.yaml (if provided).
      - Then we run a quick /v2/account check to fail fast on bad creds.
    """
    _set_env_if_absent("APCA_BASE_URL", base_url)
    _set_env_if_absent("APCA_API_KEY_ID", key)
    _set_env_if_absent("APCA_API_SECRET_KEY", secret)
    # common aliases some code paths may read
    _set_env_if_absent("ALPACA_KEY", key)
    _set_env_if_absent("ALPACA_SECRET", secret)

    if do_check:
        _quick_account_check()

def _quick_account_check(timeout: int = 8) -> None:
    base = (os.getenv("APCA_BASE_URL") or "https://paper-api.alpaca.markets").rstrip("/")
    key  = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_KEY")
    sec  = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET")

    if not key or not sec:
        raise SystemExit(
            "Alpaca credentials missing: set APCA_API_KEY_ID and APCA_API_SECRET_KEY via GitHub Secrets."
        )

    try:
        r = requests.get(
            f"{base}/v2/account",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
            timeout=timeout,
        )
    except Exception as e:
        raise SystemExit(f"Alpaca account check failed: {e}")

    if r.status_code == 401:
        raise SystemExit("Alpaca returned 401 Unauthorized — check APCA_* secrets and that BASE_URL matches paper vs live.")
    if r.status_code >= 400:
        raise SystemExit(f"Alpaca account check error {r.status_code}: {r.text[:300]}")
