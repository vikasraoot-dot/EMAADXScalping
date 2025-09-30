from __future__ import annotations
from typing import Dict, Any, List, Optional
import requests, time

RETRY_STATUS = {429, 500, 502, 503, 504}

def _req_with_retry(method: str, url: str, headers: dict, timeout: int = 20,
                    max_retries: int = 5, backoff_base: float = 0.7, **kwargs) -> requests.Response:
    attempt = 0
    while True:
        try:
            r = requests.request(method, url, headers=headers, timeout=timeout, **kwargs)
            if r.status_code in RETRY_STATUS and attempt < max_retries:
                attempt += 1
                time.sleep(backoff_base * (2 ** (attempt - 1)))
                continue
            r.raise_for_status()
            return r
        except requests.HTTPError:
            if attempt >= max_retries: raise
            attempt += 1
            time.sleep(backoff_base * (2 ** (attempt - 1)))
        except requests.RequestException:
            if attempt >= max_retries: raise
            attempt += 1
            time.sleep(backoff_base * (2 ** (attempt - 1)))

def _headers(key: str, secret: str) -> dict:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

def get_order(base_url: str, key: str, secret: str, order_id: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v2/orders/{order_id}"
    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=15)
    return r.json() if r.text else {}

def get_activities(base_url: str, key: str, secret: str,
                   activity_types: Optional[str] = None,
                   date: Optional[str] = None, after: Optional[str] = None, until: Optional[str] = None,
                   direction: Optional[str] = None, page_size: int = 100) -> List[Dict[str, Any]]:
    """
    Wraps GET /v2/account/activities.
    - activity_types: e.g., "FILL"
    - date: YYYY-MM-DD, or use after/until iso8601
    - direction: "asc" or "desc"
    """
    url = f"{base_url.rstrip('/')}/v2/account/activities"
    params = {"page_size": page_size}
    if activity_types: params["activity_types"] = activity_types
    if date:           params["date"] = date
    if after:          params["after"] = after
    if until:          params["until"] = until
    if direction:      params["direction"] = direction

    r = _req_with_retry("GET", url, headers=_headers(key, secret), timeout=20, params=params)
    return r.json() if r.text else []
