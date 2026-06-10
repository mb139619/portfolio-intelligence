"""
Abstract base class for all data sources.
Every ingester returns a clean Polars DataFrame in a documented schema.
The rest of the system never knows where data came from.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional

import polars as pl
from loguru import logger


@dataclass
class IngestionResult:
    source: str
    identifier: str
    rows: int
    start: Optional[date]
    end: Optional[date]
    success: bool
    error: Optional[str] = None

    def __str__(self) -> str:
        if self.success:
            return f"[OK] {self.source}/{self.identifier}: {self.rows} rows ({self.start} -> {self.end})"
        return f"[ERR] {self.source}/{self.identifier}: {self.error}"


class BaseIngester(ABC):
    source_name: str = "unknown"

    @abstractmethod
    def fetch(self, identifier: str, start: str, end: Optional[str] = None) -> pl.DataFrame:
        """Download raw data. Must return a DataFrame with a 'date' column."""
        ...

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Default: drop rows where all non-date columns are null. Override per source."""
        cols = [c for c in df.columns if c != "date"]
        if not cols:
            return df
        return df.filter(~pl.all_horizontal([pl.col(c).is_null() for c in cols]))

    def ingest(self, identifier: str, start: str,
               end: Optional[str] = None) -> tuple[pl.DataFrame, IngestionResult]:
        try:
            logger.info(f"Fetching {self.source_name}/{identifier} from {start}")
            clean = self.validate(self.fetch(identifier, start, end))
            n = len(clean)
            res = IngestionResult(
                source=self.source_name, identifier=identifier, rows=n,
                start=clean["date"].min() if n else None,
                end=clean["date"].max() if n else None, success=True,
            )
            logger.info(str(res))
            return clean, res
        except Exception as e:
            logger.error(f"Ingestion failed: {self.source_name}/{identifier}: {e}")
            return pl.DataFrame(), IngestionResult(
                source=self.source_name, identifier=identifier, rows=0,
                start=None, end=None, success=False, error=str(e),
            )
