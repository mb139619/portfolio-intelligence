"""
Resilient HTTP fetch with retries and a constant inter-retry delay.

Public data endpoints (FRED, ECB, Dartmouth) are occasionally slow or flaky, so
a handful of retries turns a transient failure into a non-event. Two important
details learned the hard way:

  - A realistic **User-Agent** is sent by default. Several of these endpoints sit
    behind a WAF/CDN that silently DROPS requests whose UA is the bare
    `python-requests/x.y` string — the request then hangs until timeout rather
    than returning 403. A browser-like UA fixes the most common "works in the
    browser, times out from Python" failure.
  - The delay between retries is **constant**, not exponential — a few quick
    retries instead of a long, escalating wait.
"""

from __future__ import annotations

import time

import requests
from loguru import logger

# A realistic browser User-Agent — avoids WAFs that drop bare python-requests UAs.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def get_with_retry(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 1.5,
) -> requests.Response:
    """
    GET a URL with retries on timeout / connection / 5xx errors.

    timeout      : per-attempt read timeout in seconds
    max_retries  : total attempts before giving up
    retry_delay  : CONSTANT seconds to wait between attempts (no escalation)
    """
    # Merge caller headers over the realistic defaults
    merged_headers = {**DEFAULT_HEADERS, **(headers or {})}
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=merged_headers, timeout=timeout)
            if resp.status_code >= 500:
                raise requests.HTTPError(f"server error {resp.status_code}")
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as e:
            last_error = e
            if attempt < max_retries - 1:
                logger.warning(
                    f"Request to {url} failed ({e}); "
                    f"retry {attempt + 1}/{max_retries - 1} in {retry_delay:.1f}s"
                )
                time.sleep(retry_delay)

    raise RuntimeError(
        f"Request to {url} failed after {max_retries} attempts: {last_error}"
    )
