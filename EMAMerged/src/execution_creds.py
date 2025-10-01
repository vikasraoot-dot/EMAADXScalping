# EMAMerged/src/execution_creds.py
from __future__ import annotations
import os

def configure_alpaca(base_url: str, key: str, secret: str) -> None:
    """
    Minimal, surgical shim:
    - Puts creds into standard Alpaca env vars so EMAMerged.src.execution
      (which already reads env) sees them with zero changes.
    - This avoids touching the stable execution layer.
    """
    if base_url:
        os.environ["APCA_BASE_URL"] = base_url
    if key:
        os.environ["APCA_API_KEY_ID"] = key
        # Also set common aliases used in some environments
        os.environ["ALPACA_KEY"] = key
        os.environ["ALPACA_API_KEY"] = key
    if secret:
        os.environ["APCA_API_SECRET_KEY"] = secret
        # Common aliases
        os.environ["ALPACA_SECRET"] = secret
        os.environ["ALPACA_API_SECRET"] = secret
