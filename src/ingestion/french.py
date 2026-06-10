"""
Kenneth French Data Library — Fama-French factors.

Fetches the official zipped CSV files directly from the Dartmouth site (no
pandas-datareader, which is unmaintained and breaks against current pandas).

The French CSV files have a free-text preamble, then a header row, then daily
rows shaped `YYYYMMDD, val, val, ...`, and sometimes trailing annual blocks.
The parser keeps only the daily block and converts percentages to decimals.
"""

from __future__ import annotations

import io
import re
import zipfile
from typing import Optional

import polars as pl
import requests
from loguru import logger

from src.ingestion.base import BaseIngester
from src.ingestion.http import get_with_retry

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

DATASET_FILES = {
    # --- US factors ---
    "FF3": "F-F_Research_Data_Factors_daily_CSV.zip",
    "FF5": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "MOM": "F-F_Momentum_Factor_daily_CSV.zip",
    # --- Developed / global factors ---
    # Use these for portfolios with material non-US exposure (e.g. EFA/EAFE):
    # regressing a global allocation on US-only factors is a defensible
    # approximation (global equity is highly correlated with US Mkt) but the
    # benchmark is conceptually US. These give the right opportunity set.
    # NOTE: the exact international filenames change occasionally on the French
    # site — verify against the data library directory if a download 404s.
    "FF5_DEV": "Developed_5_Factors_Daily_CSV.zip",
    "FF5_DEV_EX_US": "Developed_ex_US_5_Factors_Daily_CSV.zip",
    "FF5_EUROPE": "Europe_5_Factors_Daily_CSV.zip",
}

_DATE_ROW = re.compile(r"^\s*(\d{8})\s*,")


class FrenchIngester(BaseIngester):
    source_name = "french"

    def fetch(self, identifier: str, start: str, end: Optional[str] = None) -> pl.DataFrame:
        if identifier not in DATASET_FILES:
            raise ValueError(f"Unknown French dataset '{identifier}'. Choose {list(DATASET_FILES)}")
        url = BASE + DATASET_FILES[identifier]
        logger.info(f"Downloading {url}")
        resp = get_with_retry(url, timeout=60)

        # The zip contains a single CSV
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_name = zf.namelist()[0]
            text = zf.read(csv_name).decode("utf-8", errors="replace")

        df = _parse_french_csv(text)

        if start:
            df = df.filter(pl.col("date") >= pl.lit(start).str.to_date())
        if end:
            df = df.filter(pl.col("date") <= pl.lit(end).str.to_date())
        return df

    def validate(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("value").is_not_null())


def _parse_french_csv(text: str) -> pl.DataFrame:
    """
    Pure parser for a French daily CSV. Returns long format:
    date | factor | value | source  (values already in decimal).
    """
    lines = text.splitlines()

    # Find the first daily data row (8-digit date), and use the line above it
    # (the comma-led header) for the factor names.
    first_data_idx = None
    for i, line in enumerate(lines):
        if _DATE_ROW.match(line):
            first_data_idx = i
            break
    if first_data_idx is None:
        raise ValueError("No daily data rows found in French CSV")

    # Header: nearest preceding non-empty line
    header_idx = first_data_idx - 1
    while header_idx >= 0 and not lines[header_idx].strip():
        header_idx -= 1
    header_tokens = [t.strip() for t in lines[header_idx].split(",")]
    factor_names = [t for t in header_tokens[1:] if t]  # drop leading empty col

    # Collect the contiguous daily block
    data_rows = []
    for line in lines[first_data_idx:]:
        if not _DATE_ROW.match(line):
            # stop at the first non-daily row after the block starts
            if data_rows:
                break
            continue
        parts = [p.strip() for p in line.split(",")]
        date_str = parts[0]
        values = parts[1:1 + len(factor_names)]
        data_rows.append((date_str, values))

    records = []
    for date_str, values in data_rows:
        date = pl.lit(date_str).str.to_date("%Y%m%d")
        for fname, v in zip(factor_names, values):
            try:
                val = float(v) / 100.0   # percent -> decimal
            except ValueError:
                val = None
            records.append({"date_str": date_str, "factor": fname, "value": val})

    long = pl.DataFrame(records)
    return (
        long
        .with_columns([
            pl.col("date_str").str.to_date("%Y%m%d").alias("date"),
            pl.lit("french").alias("source"),
        ])
        .select(["date", "factor", "value", "source"])
        .sort(["date", "factor"])
    )
