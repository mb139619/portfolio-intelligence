"""Yahoo Finance ingester via yfinance. Prices + OHLCV with adj_close."""

from __future__ import annotations

from typing import Optional

import polars as pl
import yfinance as yf

from src.ingestion.base import BaseIngester


class YahooIngester(BaseIngester):
    source_name = "yahoo"

    def fetch(self, identifier: str, start: str, end: Optional[str] = None) -> pl.DataFrame:
        raw = yf.Ticker(identifier).history(
            start=start, end=end, auto_adjust=False, actions=False
        )
        if raw.empty:
            raise ValueError(f"No data returned for {identifier}")

        raw = raw.reset_index()
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

        # Normalise the date column (strip timezone, keep date only)
        if raw["date"].dt.tz is not None:
            raw["date"] = raw["date"].dt.tz_localize(None)
        raw["date"] = raw["date"].dt.date
        raw["ticker"] = identifier

        cols = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        present = [c for c in cols if c in raw.columns]
        return (
            pl.from_pandas(raw[present])
            .with_columns(pl.col("date").cast(pl.Date))
            .sort("date")
        )

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("adj_close").is_not_null() & (pl.col("adj_close") > 0))
