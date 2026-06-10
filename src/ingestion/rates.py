"""
Rates ingestion — unified interface over multiple central-bank sources.

Key design decision (per the architecture discussion):
  We do NOT have a "FRED ingester" and an "ECB ingester" as separate concerns.
  We have rate SERIES, each of which knows which backend to pull from.
  The rest of the system asks for a logical rate (e.g. "EUR_3M") and never
  cares whether it came from FRED or the ECB Data Portal.

Backends:
  - FRED  : via pandas-datareader, no API key needed for most series
  - ECB   : via the ECB Data Portal REST API (SDMX), no key needed

Output schema (long format, same for every backend):
  date | series_id | value | currency | source
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import polars as pl
import requests
from loguru import logger

from src.ingestion.base import BaseIngester
from src.ingestion.http import get_with_retry


# ──────────────────────────────────────────────────────────────
# Series registry — the single place that maps a logical rate
# to its backend and the backend-specific code.
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RateSeries:
    series_id: str       # our logical id, e.g. "USD_FEDFUNDS"
    backend: str         # "fred" or "ecb"
    code: str            # backend-specific code
    currency: str
    description: str


RATE_REGISTRY: dict[str, RateSeries] = {
    # --- USD (FRED) ---
    "USD_FEDFUNDS": RateSeries("USD_FEDFUNDS", "fred", "DFF", "USD", "Effective Fed Funds Rate"),
    "USD_3M":       RateSeries("USD_3M", "fred", "DGS3MO", "USD", "3-Month Treasury"),
    "USD_2Y":       RateSeries("USD_2Y", "fred", "DGS2", "USD", "2-Year Treasury"),
    "USD_10Y":      RateSeries("USD_10Y", "fred", "DGS10", "USD", "10-Year Treasury"),
    "USD_CPI":      RateSeries("USD_CPI", "fred", "CPIAUCSL", "USD", "US CPI"),
    "USD_HY_OAS":   RateSeries("USD_HY_OAS", "fred", "BAMLH0A0HYM2", "USD", "US High Yield OAS"),

    # --- EUR (ECB Data Portal) ---
    # ECB key format: <dataflow>.<series key>
    "EUR_DFR":      RateSeries("EUR_DFR", "ecb", "FM.D.U2.EUR.4F.KR.DFR.LEV", "EUR", "ECB Deposit Facility Rate"),
    "EUR_MRO":      RateSeries("EUR_MRO", "ecb", "FM.D.U2.EUR.4F.KR.MRR_FR.LEV", "EUR", "ECB Main Refinancing Rate"),
    "EUR_ESTR":     RateSeries("EUR_ESTR", "ecb", "EST.B.EU000A2X2A25.WT", "EUR", "Euro Short-Term Rate"),
}


# ──────────────────────────────────────────────────────────────
# Backends
# ──────────────────────────────────────────────────────────────

class _FredBackend:
    """
    Fetches FRED series via the public CSV endpoint (no API key, no
    pandas-datareader). URL form:
      https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd=...&coed=...
    Missing observations are encoded as '.'.
    """
    BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

    def fetch(self, code: str, start: str, end: Optional[str]) -> pl.DataFrame:
        params = {"id": code, "cosd": start}
        if end:
            params["coed"] = end
        resp = get_with_retry(self.BASE, params=params, timeout=60)
        return _parse_fred_csv(resp.text)


def _parse_fred_csv(text: str) -> pl.DataFrame:
    """Pure parser for the fredgraph CSV. Returns date | value."""
    df = pl.read_csv(io.StringIO(text), null_values=["."])
    # First column is the date ('observation_date' or 'DATE'), second the value
    date_col, value_col = df.columns[0], df.columns[1]
    return (
        df.select([
            pl.col(date_col).cast(pl.Date).alias("date"),
            pl.col(value_col).cast(pl.Float64, strict=False).alias("value"),
        ])
        .drop_nulls()
        .sort("date")
    )


class _ECBBackend:
    BASE = "https://data-api.ecb.europa.eu/service/data"

    def fetch(self, code: str, start: str, end: Optional[str]) -> pl.DataFrame:
        dataflow, _, key = code.partition(".")
        params = {"format": "csvdata", "startPeriod": start}
        if end:
            params["endPeriod"] = end
        url = f"{self.BASE}/{dataflow}/{key}"
        resp = get_with_retry(url, params=params, timeout=60,
                              headers={"Accept": "text/csv"})
        df = pl.read_csv(io.StringIO(resp.text))
        # ECB CSV uses TIME_PERIOD and OBS_VALUE
        return (
            df.select([
                pl.col("TIME_PERIOD").str.to_date().alias("date"),
                pl.col("OBS_VALUE").cast(pl.Float64).alias("value"),
            ])
            .sort("date")
        )


_BACKENDS = {"fred": _FredBackend(), "ecb": _ECBBackend()}


# ──────────────────────────────────────────────────────────────
# Unified ingester
# ──────────────────────────────────────────────────────────────

class RatesIngester(BaseIngester):
    source_name = "rates"

    def fetch(self, identifier: str, start: str, end: Optional[str] = None) -> pl.DataFrame:
        """identifier is a logical series id from RATE_REGISTRY (e.g. 'EUR_DFR')."""
        if identifier not in RATE_REGISTRY:
            raise ValueError(
                f"Unknown rate series '{identifier}'. "
                f"Available: {sorted(RATE_REGISTRY)}"
            )
        spec = RATE_REGISTRY[identifier]
        backend = _BACKENDS[spec.backend]
        logger.info(f"Rate {identifier} -> {spec.backend}:{spec.code}")

        raw = backend.fetch(spec.code, start, end)
        return raw.with_columns([
            pl.lit(spec.series_id).alias("series_id"),
            pl.lit(spec.currency).alias("currency"),
            pl.lit(spec.backend).alias("source"),
        ]).select(["date", "series_id", "value", "currency", "source"])

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("value").is_not_null())


def available_rates(currency: Optional[str] = None) -> list[str]:
    """List logical rate ids, optionally filtered by currency."""
    if currency:
        return sorted(s for s, r in RATE_REGISTRY.items() if r.currency == currency)
    return sorted(RATE_REGISTRY)
