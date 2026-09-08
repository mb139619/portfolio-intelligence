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

from loguru import logger

from src.config import settings
from src.ingestion.base import IngestionResult
from src.ingestion.french import FrenchIngester
from src.ingestion.prices import PricesIngester
from src.ingestion.prices import resolve as resolve_price_source
from src.ingestion.rates import RatesIngester
from src.store.parquet_store import ParquetStore


class IngestionPipeline:
    def __init__(self, store: ParquetStore) -> None:
        self.store = store
        self.prices = PricesIngester()
        self.rates = RatesIngester()
        self.french = FrenchIngester()

    # --- prices ---

    def update_prices(
        self,
        tickers: list[str],
        start: str | None = None,
        incremental: bool = True,
        asset_classes: dict[str, str] | None = None,
    ) -> list[IngestionResult]:
        """
        Fetch prices for any ticker the price registry can route.

        `asset_classes` optionally overrides the registry's guess per ticker —
        the portfolio spec knows that TLT is fixed income where the registry
        only knows it is "some Yahoo symbol", and the data quality outlier
        bands depend on getting that right.
        """
        asset_classes = asset_classes or {}
        results = []
        for t in tickers:
            source = resolve_price_source(t)
            fetch_start = start or settings.default_start
            if incremental:
                last = self.store.last_date(t)
                if last is not None:
                    fetch_start = (last + timedelta(days=1)).isoformat()
                    if fetch_start >= str(__import__("datetime").date.today()):
                        logger.info(f"{t} already up to date")
                        results.append(
                            IngestionResult(source.backend, t, 0, None, None, True)
                        )
                        # Still record the metadata: a ticker already on disk
                        # from before the registry existed has no calendar
                        # recorded, and skipping it here would leave it stuck
                        # on the default forever.
                        self._record_meta(t, source, asset_classes)
                        continue

            df, res = self.prices.ingest(t, fetch_start)
            if res.success and res.rows > 0:
                self.store.write_prices(t, df, upsert=True)
                self._record_meta(t, source, asset_classes)
            results.append(res)
        return results

    def _record_meta(self, ticker: str, source, asset_classes: dict[str, str]) -> None:
        self.store.write_asset_meta(
            ticker,
            calendar=source.calendar,
            asset_class=asset_classes.get(ticker, source.asset_class),
            source=f"{source.backend}:{source.exchange}" if source.exchange
            else source.backend,
        )

    # --- rates ---

    def update_rates(
        self,
        series_ids: list[str],
        start: str | None = None,
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
        start: str | None = None,
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
