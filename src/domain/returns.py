"""
ReturnSeries — typed wrapper around a Polars DataFrame of asset returns.
Encapsulates alignment, validation, and common transformations.
Pure: depends only on polars and numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

from src.domain.calendar import Calendar


@dataclass
class ReturnSeries:
    data: pl.DataFrame          # columns: date + one per ticker
    tickers: list[str]
    is_log: bool = False
    # The calendar the ROWS sit on, not the calendar of any one asset. A crypto
    # pair read on its own is CONTINUOUS; the same pair joined to an equity is
    # TRADING_DAYS, because the join keeps only the days both actually traded.
    # Analytics annualise from this instead of assuming 252.
    calendar: Calendar = Calendar.TRADING_DAYS

    def __post_init__(self) -> None:
        if "date" not in self.data.columns:
            raise ValueError("ReturnSeries must have a 'date' column")
        missing = [t for t in self.tickers if t not in self.data.columns]
        if missing:
            raise ValueError(f"Tickers missing from DataFrame: {missing}")

    # --- constructors ---

    @classmethod
    def from_prices(
        cls, prices: pl.DataFrame, calendar: Calendar = Calendar.TRADING_DAYS
    ) -> "ReturnSeries":
        tickers = [c for c in prices.columns if c != "date"]
        returns = prices.select(
            [pl.col("date")]
            + [(pl.col(t) / pl.col(t).shift(1) - 1).alias(t) for t in tickers]
        ).drop_nulls()
        return cls(data=returns, tickers=tickers, is_log=False, calendar=calendar)

    @classmethod
    def from_log_prices(
        cls, prices: pl.DataFrame, calendar: Calendar = Calendar.TRADING_DAYS
    ) -> "ReturnSeries":
        tickers = [c for c in prices.columns if c != "date"]
        returns = prices.select(
            [pl.col("date")]
            + [(pl.col(t) / pl.col(t).shift(1)).log().alias(t) for t in tickers]
        ).drop_nulls()
        return cls(data=returns, tickers=tickers, is_log=True, calendar=calendar)

    # --- accessors ---

    def to_numpy(self) -> np.ndarray:
        return self.data.select(self.tickers).to_numpy()

    def to_numpy_series(self, ticker: str) -> np.ndarray:
        return self.data[ticker].to_numpy()

    @property
    def dates(self) -> pl.Series:
        return self.data["date"]

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def n_obs(self) -> int:
        return len(self.data)

    @property
    def periods_per_year(self) -> int:
        """Annualisation factor implied by the calendar these rows sit on."""
        return self.calendar.periods_per_year

    def select(self, tickers: list[str]) -> "ReturnSeries":
        missing = [t for t in tickers if t not in self.tickers]
        if missing:
            raise ValueError(f"Tickers not in series: {missing}")
        return ReturnSeries(
            data=self.data.select(["date"] + tickers),
            tickers=tickers,
            is_log=self.is_log,
            calendar=self.calendar,
        )

    def trim(self, start: Optional[str] = None, end: Optional[str] = None) -> "ReturnSeries":
        df = self.data
        if start:
            df = df.filter(pl.col("date") >= pl.lit(start).str.to_date())
        if end:
            df = df.filter(pl.col("date") <= pl.lit(end).str.to_date())
        return ReturnSeries(data=df, tickers=self.tickers, is_log=self.is_log,
                            calendar=self.calendar)

    def portfolio_returns(self, weights: dict[str, float]) -> pl.Series:
        w = np.array([weights[t] for t in self.tickers])
        return pl.Series("portfolio", self.to_numpy() @ w)

    def __repr__(self) -> str:
        return (
            f"ReturnSeries(tickers={self.tickers}, obs={self.n_obs}, "
            f"from={self.dates[0]}, to={self.dates[-1]}, log={self.is_log}, "
            f"calendar={self.calendar}, ppy={self.periods_per_year})"
        )
