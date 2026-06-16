"""
ParquetStore — Parquet-first storage.

Design philosophy:
  - Data lives in Parquet files on disk. That is the source of truth.
  - There is NO always-on database, no schema, no migrations.
  - DuckDB is used as a *stateless query engine* that reads the Parquet
    files on demand. Open it, query, throw it away.

Layout:
  data/raw/prices/{ticker}.parquet      one file per ticker (long format)
  data/raw/macro/rates.parquet          all rate series (long format)
  data/raw/factors/factors.parquet      all factor series (long format)

Two access patterns:
  1. Typed reads → return domain objects (ReturnSeries). Used by analytics.
  2. .sql(query) → ad-hoc DuckDB SQL over the Parquet files. Used in notebooks.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import polars as pl
from loguru import logger

from src.domain.returns import ReturnSeries


class ParquetStore:
    def __init__(self, base_dir: Path) -> None:
        self.base = Path(base_dir)
        self.prices_dir = self.base / "raw" / "prices"
        self.macro_dir = self.base / "raw" / "macro"
        self.factors_dir = self.base / "raw" / "factors"
        for d in (self.prices_dir, self.macro_dir, self.factors_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Ad-hoc SQL — DuckDB as a query engine over Parquet
    # ------------------------------------------------------------------

    def sql(self, query: str) -> pl.DataFrame:
        """
        Run arbitrary DuckDB SQL. Use the table-function helpers below,
        or read_parquet() directly. A fresh connection per call — stateless.

        Example (in a notebook):
            store.sql("SELECT ticker, count(*) FROM prices() GROUP BY ticker")
        """
        con = duckdb.connect(":memory:")
        try:
            # Register convenient views over the parquet globs
            con.execute(f"""
                CREATE OR REPLACE MACRO prices() AS TABLE
                    SELECT * FROM read_parquet('{self.prices_dir}/*.parquet');
            """)
            macro_glob = self.macro_dir / "*.parquet"
            factors_glob = self.factors_dir / "*.parquet"
            if list(self.macro_dir.glob("*.parquet")):
                con.execute(f"""
                    CREATE OR REPLACE MACRO rates() AS TABLE
                        SELECT * FROM read_parquet('{macro_glob}');
                """)
            if list(self.factors_dir.glob("*.parquet")):
                con.execute(f"""
                    CREATE OR REPLACE MACRO factors() AS TABLE
                        SELECT * FROM read_parquet('{factors_glob}');
                """)
            return con.execute(query).pl()
        finally:
            con.close()

    # ------------------------------------------------------------------
    # Prices — write
    # ------------------------------------------------------------------

    def write_prices(self, ticker: str, df: pl.DataFrame, upsert: bool = True) -> int:
        """
        Write a single ticker's prices to its Parquet file.
        df: long format with columns date, ticker, open, high, low, close, adj_close, volume.

        If upsert=True and a file exists, merge on date (new rows win) so that
        incremental updates don't lose history.
        """
        path = self.prices_dir / f"{ticker}.parquet"

        if upsert and path.exists():
            existing = pl.read_parquet(path)
            combined = (
                pl.concat([existing, df], how="vertical_relaxed")
                .unique(subset=["date"], keep="last")
                .sort("date")
            )
        else:
            combined = df.sort("date")

        combined.write_parquet(path)
        logger.debug(f"Wrote {len(combined)} rows → {path.name}")
        return len(combined)

    # ------------------------------------------------------------------
    # Prices — read (typed)
    # ------------------------------------------------------------------

    def read_prices(
        self,
        tickers: list[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        column: str = "adj_close",
    ) -> pl.DataFrame:
        """
        Read prices for the given tickers as a WIDE DataFrame:
        date | ticker_1 | ticker_2 | ...
        Reads each ticker's parquet directly with Polars (no DuckDB needed here).
        """
        frames = []
        for t in tickers:
            path = self.prices_dir / f"{t}.parquet"
            if not path.exists():
                logger.warning(f"No price file for {t}, skipping")
                continue
            df = pl.read_parquet(path).select(["date", column]).rename({column: t})
            frames.append(df)

        if not frames:
            return pl.DataFrame()

        # Join all on date (outer → align calendars)
        wide = frames[0]
        for f in frames[1:]:
            wide = wide.join(f, on="date", how="full", coalesce=True)
        wide = wide.sort("date")

        if start:
            wide = wide.filter(pl.col("date") >= pl.lit(start).str.to_date())
        if end:
            wide = wide.filter(pl.col("date") <= pl.lit(end).str.to_date())

        return wide

    def read_returns(
        self,
        tickers: list[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
        kind: str = "simple",
    ) -> ReturnSeries:
        """Read prices and convert to a ReturnSeries (drops rows with any null)."""
        prices = self.read_prices(tickers, start, end)
        if prices.is_empty():
            raise ValueError(f"No price data for {tickers}")
        # Drop rows where any ticker is null so returns are aligned
        prices = prices.drop_nulls()
        if kind == "log":
            return ReturnSeries.from_log_prices(prices)
        return ReturnSeries.from_prices(prices)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def last_date(self, ticker: str) -> Optional[date]:
        path = self.prices_dir / f"{ticker}.parquet"
        if not path.exists():
            return None
        return pl.read_parquet(path, columns=["date"])["date"].max()

    def available_tickers(self) -> list[str]:
        return sorted(p.stem for p in self.prices_dir.glob("*.parquet"))

    # ------------------------------------------------------------------
    # Generic long-format series (rates, factors)
    # ------------------------------------------------------------------

    def write_series(
        self,
        df: pl.DataFrame,
        name: str,
        subdir: str,
        upsert_keys: Optional[list[str]] = None,
    ) -> int:
        """
        Write a long-format series file (e.g. rates, factors).

        If ``upsert_keys`` is given and the file already exists, the new rows are
        MERGED into the existing file (new rows win on key collision) instead of
        overwriting it. This makes a partial-failure run non-destructive: if one
        source (e.g. a flaky FRED series) fails, the previously stored rows for
        the missing series survive instead of being silently wiped.
        """
        target = (self.base / "raw" / subdir)
        target.mkdir(parents=True, exist_ok=True)
        path = target / f"{name}.parquet"
        if upsert_keys and path.exists():
            existing = pl.read_parquet(path)
            # New data is concatenated last, so keep="last" lets fresh rows win.
            df = pl.concat([existing, df], how="vertical_relaxed").unique(
                subset=upsert_keys, keep="last", maintain_order=True
            )
        df.sort("date").write_parquet(path)
        logger.debug(f"Wrote {len(df)} rows → {subdir}/{name}.parquet")
        return len(df)

    def read_factors(
        self,
        factor_names: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Read factor series as a WIDE DataFrame: date | Mkt-RF | SMB | ... | RF
        If factor_names is None, returns all available factors.
        """
        path = self.factors_dir / "factors.parquet"
        if not path.exists():
            return pl.DataFrame()
        long = pl.read_parquet(path)
        if factor_names:
            long = long.filter(pl.col("factor").is_in(factor_names))
        wide = long.pivot(index="date", on="factor", values="value").sort("date")
        if start:
            wide = wide.filter(pl.col("date") >= pl.lit(start).str.to_date())
        if end:
            wide = wide.filter(pl.col("date") <= pl.lit(end).str.to_date())
        return wide

    def read_rates(
        self,
        series_ids: Optional[list[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pl.DataFrame:
        """Read rate series as a WIDE DataFrame: date | series_1 | series_2 | ..."""
        path = self.macro_dir / "rates.parquet"
        if not path.exists():
            return pl.DataFrame()
        long = pl.read_parquet(path)
        if series_ids:
            long = long.filter(pl.col("series_id").is_in(series_ids))
        wide = long.pivot(index="date", on="series_id", values="value").sort("date")
        if start:
            wide = wide.filter(pl.col("date") >= pl.lit(start).str.to_date())
        if end:
            wide = wide.filter(pl.col("date") <= pl.lit(end).str.to_date())
        return wide
