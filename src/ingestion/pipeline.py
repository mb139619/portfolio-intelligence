"""
Ingestion pipeline — orchestrates ingesters + store.

This is the entry point you call from a notebook or a CLI/cron:
    pipeline = IngestionPipeline(store)
    pipeline.update_prices(["SPY", "TLT", "GLD"])
    pipeline.update_rates(["USD_FEDFUNDS", "EUR_DFR"])
    pipeline.update_factors(["FF5", "MOM"])

It handles incremental updates: for prices, it only fetches from the last
stored date onward, so re-running is cheap.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Optional

from loguru import logger

from src.config import settings
from src.store.parquet_store import ParquetStore
from src.ingestion.base import IngestionResult
from src.ingestion.yahoo import YahooIngester
from src.ingestion.rates import RatesIngester
from src.ingestion.french import FrenchIngester


class IngestionPipeline:
    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self.yahoo = YahooIngester()
        self.rates = RatesIngester()
        self.french = FrenchIngester()

    # --- prices ---

    def update_prices(
        self,
        tickers: list[str],
        start: Optional[str] = None,
        incremental: bool = True,
    ) -> list[IngestionResult]:
        results = []
        for t in tickers:
            fetch_start = start or settings.default_start
            if incremental:
                last = self.store.last_date(t)
                if last is not None:
                    fetch_start = (last + timedelta(days=1)).isoformat()
                    if fetch_start >= str(__import__("datetime").date.today()):
                        logger.info(f"{t} already up to date")
                        results.append(IngestionResult("yahoo", t, 0, None, None, True))
                        continue

            df, res = self.yahoo.ingest(t, fetch_start)
            if res.success and res.rows > 0:
                self.store.write_prices(t, df, upsert=True)
            results.append(res)
        return results

    # --- rates ---

    def update_rates(
        self,
        series_ids: list[str],
        start: Optional[str] = None,
    ) -> list[IngestionResult]:
        results = []
        frames = []
        for sid in series_ids:
            df, res = self.rates.ingest(sid, start or settings.default_start)
            if res.success and res.rows > 0:
                frames.append(df)
            results.append(res)
        if frames:
            import polars as pl
            combined = pl.concat(frames, how="vertical_relaxed")
            # Merge, don't overwrite: a failed series (e.g. FRED timing out) must
            # not wipe the rows we already have for the series that did succeed.
            self.store.write_series(
                combined, "rates", "macro", upsert_keys=["date", "series_id"]
            )
        return results

    # --- factors ---

    def update_factors(
        self,
        datasets: list[str],
        start: Optional[str] = None,
    ) -> list[IngestionResult]:
        results = []
        frames = []
        for ds in datasets:
            df, res = self.french.ingest(ds, start or settings.default_start)
            if res.success and res.rows > 0:
                frames.append(df)
            results.append(res)
        if frames:
            import polars as pl
            combined = pl.concat(frames, how="vertical_relaxed").unique(
                subset=["date", "factor"], keep="last"
            )
            self.store.write_series(
                combined, "factors", "factors", upsert_keys=["date", "factor"]
            )
        return results
