"""
Resilient HTTP fetch with retries and exponential backoff.

Public data endpoints (FRED, ECB, Dartmouth) are occasionally slow or flaky.
A single 30s attempt that gives up is fragile; a few retries with backoff turn
a transient timeout into a non-event. This is shared by all web ingesters.
"""

from __future__ import annotations

import time

import requests
from loguru import logger


def get_with_retry(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 60,
    max_retries: int = 4,
    backoff: float = 2.0,
) -> requests.Response:
    """
    GET a URL with retries on timeout / connection / 5xx errors.

    timeout     : per-attempt read timeout in seconds (generous default 60)
    max_retries : total attempts before giving up
    backoff     : wait grows as backoff ** attempt (2s, 4s, 8s, ...)
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            # Retry transient server errors; raise immediately on client errors
            if resp.status_code >= 500:
                raise requests.HTTPError(f"server error {resp.status_code}")
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_error = e
            if attempt < max_retries - 1:
                wait = backoff ** attempt
                logger.warning(
                    f"Request to {url} failed ({e}); "
                    f"retry {attempt + 1}/{max_retries - 1} in {wait:.0f}s"
                )
                time.sleep(wait)

    raise RuntimeError(
        f"Request to {url} failed after {max_retries} attempts: {last_error}"
    )
