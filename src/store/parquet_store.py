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

import duckdb
import polars as pl
from loguru import logger

from src.domain.calendar import Calendar
from src.domain.calendar import resolve as resolve_calendar
from src.domain.returns import ReturnSeries


class ParquetStore:
    def __init__(self, base_dir: Path) -> None:
        self.base = Path(base_dir)
        self.prices_dir = self.base / "raw" / "prices"
        self.macro_dir = self.base / "raw" / "macro"
        self.factors_dir = self.base / "raw" / "factors"
        # Per-ticker metadata (calendar, asset class, source). Kept OUTSIDE
        # prices_dir on purpose: the DuckDB `prices()` macro globs that
        # directory, and a file with a different schema would break the union.
        self.assets_path = self.base / "raw" / "_assets.parquet"
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
        df: long format with columns date, ticker, open, high, low, close,
            adj_close, volume.

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
        start: str | None = None,
        end: str | None = None,
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
        start: str | None = None,
        end: str | None = None,
        kind: str = "simple",
    ) -> ReturnSeries:
        """
        Read prices and convert to a ReturnSeries.

        Calendar handling lives here, and it is the whole ballgame when asset
        classes are mixed. The `drop_nulls` below is an INTERSECTION: only dates
        on which every requested asset actually traded survive. Nothing is ever
        forward-filled, so a crypto series joined to equities loses its weekends
        rather than lending equities two fake flat days a week.

        The resulting series is annualised on the most restrictive calendar of
        those it contains — 252 for any mix involving an exchange-traded asset,
        365 only when everything trades continuously. Annualising an
        intersected series at 365 would inflate volatility by about 20%.
        """
        prices = self.read_prices(tickers, start, end)
        if prices.is_empty():
            raise ValueError(f"No price data for {tickers}")

        present = [c for c in prices.columns if c != "date"]
        n_before = len(prices)
        prices = prices.drop_nulls()

        calendars = [self.calendar_for(t) for t in present]
        calendar = resolve_calendar(calendars)

        distinct = {str(c) for c in calendars}
        if len(distinct) > 1:
            dropped = n_before - len(prices)
            logger.info(
                f"Mixed calendars across {present} ({', '.join(sorted(distinct))}). "
                f"Intersected to {len(prices)} common dates, dropping {dropped}; "
                f"annualising at {calendar.periods_per_year}/yr ({calendar}). "
                f"No prices were forward-filled."
            )

        if kind == "log":
            return ReturnSeries.from_log_prices(prices, calendar=calendar)
        return ReturnSeries.from_prices(prices, calendar=calendar)

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Asset metadata — the calendar travels with the data
    # ------------------------------------------------------------------

    def write_asset_meta(
        self, ticker: str, calendar: Calendar,
        asset_class: str = "unknown", source: str = "",
    ) -> None:
        """
        Record how a ticker behaves. Written by ingestion, read by everything
        that needs to annualise or judge a gap.

        This exists so the store never has to import the ingestion layer to
        find out that BTC trades on weekends — the fact is stored alongside the
        prices instead of being re-derived from whoever fetched them.
        """
        row = pl.DataFrame({
            "ticker": [ticker],
            "calendar": [str(calendar)],
            "asset_class": [asset_class],
            "source": [source],
        })
        if self.assets_path.exists():
            existing = pl.read_parquet(self.assets_path)
            row = pl.concat([existing, row], how="vertical_relaxed").unique(
                subset=["ticker"], keep="last", maintain_order=True
            )
        row.sort("ticker").write_parquet(self.assets_path)

    def read_asset_meta(self) -> dict[str, dict]:
        """{ticker: {calendar, asset_class, source}}. Empty if never written."""
        if not self.assets_path.exists():
            return {}
        df = pl.read_parquet(self.assets_path)
        return {r["ticker"]: r for r in df.to_dicts()}

    def calendar_for(self, ticker: str) -> Calendar:
        """
        A ticker's calendar, defaulting to TRADING_DAYS when unknown.

        The default is what keeps this change backward compatible: price files
        written before the metadata existed carry no entry, and every one of
        them is an exchange-traded instrument, so they resolve exactly as they
        did before.
        """
        meta = self.read_asset_meta().get(ticker)
        if not meta:
            return Calendar.TRADING_DAYS
        try:
            return Calendar(meta["calendar"])
        except ValueError:
            logger.warning(
                f"Unknown calendar {meta['calendar']!r} recorded for {ticker}; "
                f"falling back to {Calendar.TRADING_DAYS}"
            )
            return Calendar.TRADING_DAYS

    def last_date(self, ticker: str) -> date | None:
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
        upsert_keys: list[str] | None = None,
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
        factor_names: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
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
        series_ids: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
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
