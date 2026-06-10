"""
Rolling / dynamic correlation through time.

Two views:
  - pairwise rolling correlation: how the link between two specific assets
    evolves (e.g. stocks vs bonds — the sign flip is a regime tell).
  - average correlation: the mean off-diagonal correlation of the whole
    universe at each point in time. A single "systemic correlation" series:
    spikes toward 1 in crises (everything moves together), falls in calm
    regimes. This is one of the most useful early-warning indicators.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries


def rolling_pairwise_correlation(
    rs: ReturnSeries,
    ticker_a: str,
    ticker_b: str,
    window: int = 63,
) -> pl.DataFrame:
    """Rolling Pearson correlation between two assets. Returns date | correlation."""
    a = rs.to_numpy_series(ticker_a)
    b = rs.to_numpy_series(ticker_b)
    dates = rs.dates.to_list()

    rows = []
    for i in range(window, len(a) + 1):
        wa, wb = a[i - window:i], b[i - window:i]
        c = np.corrcoef(wa, wb)[0, 1]
        rows.append({"date": dates[i - 1], "correlation": float(c)})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def average_correlation(
    rs: ReturnSeries,
    window: int = 63,
) -> pl.DataFrame:
    """
    Mean off-diagonal correlation of the whole universe over time.
    Returns date | avg_correlation | min_correlation | max_correlation.
    """
    R = rs.to_numpy()
    dates = rs.dates.to_list()
    n_assets = R.shape[1]
    iu = np.triu_indices(n_assets, k=1)   # upper triangle, excl. diagonal

    rows = []
    for i in range(window, R.shape[0] + 1):
        corr = np.corrcoef(R[i - window:i], rowvar=False)
        off = corr[iu]
        rows.append({
            "date": dates[i - 1],
            "avg_correlation": float(np.mean(off)),
            "min_correlation": float(np.min(off)),
            "max_correlation": float(np.max(off)),
        })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
