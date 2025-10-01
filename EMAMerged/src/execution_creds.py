# EMAMerged/src/execution_creds.py
from __future__ import annotations
import os

def configure_alpaca(base_url: str, key: str, secret: str) -> None:
    """
    Minimal, surgical shim:
    - Set standard Alpaca env vars so EMAMerged.src.execution (which already
      reads env) picks up credentials without touching that stable code.
    """
    if base_url:
        os.environ["APCA_BASE_URL"] = base_url
    if key:
        os.environ["APCA_API_KEY_ID"] = key
        # Common aliases in some environments
        os.environ["ALPACA_KEY"] = key
        os.environ["ALPACA_API_KEY"] = key
    if secret:
        os.environ["APCA_API_SECRET_KEY"] = secret
        os.environ["ALPACA_SECRET"] = secret
        os.environ["ALPACA_API_SECRET"] = secret
