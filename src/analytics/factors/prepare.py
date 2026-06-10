"""
Factor data preparation.

Bridges two data worlds with different calendars:
  - asset returns (from Yahoo, via ReturnSeries)
  - factor returns + risk-free (from French library)

The regression r_excess = alpha + B*f + eps requires:
  - LHS: asset returns in EXCESS of the risk-free rate
  - RHS: the factor returns (Mkt-RF, SMB, ... — already excess by construction)

This module produces aligned, ready-to-regress numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from loguru import logger

from src.domain.returns import ReturnSeries


# Standard Fama-French factor names (RF is handled separately, not a regressor)
FF5_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
FF3_FACTORS = ["Mkt-RF", "SMB", "HML"]
RF_COLUMN = "RF"
MOM_FACTOR = "Mom"


@dataclass
class AlignedFactorData:
    """Asset/portfolio excess returns aligned with factor returns on common dates."""
    dates: list
    excess_returns: np.ndarray       # (T,) — single series, already excess of RF
    factors: np.ndarray              # (T, K)
    factor_names: list[str]
    rf: np.ndarray                   # (T,) — daily risk-free used

    @property
    def n_obs(self) -> int:
        return len(self.dates)

    @property
    def coverage(self) -> str:
        """Human-readable window the factor model is actually fit on."""
        if not self.dates:
            return "empty"
        return f"{self.dates[0]} → {self.dates[-1]} ({self.n_obs} obs)"


def align_factors(
    asset_returns: pl.Series | np.ndarray,
    return_dates: pl.Series,
    factor_wide: pl.DataFrame,
    factor_names: list[str],
    rf_column: str = RF_COLUMN,
) -> AlignedFactorData:
    """
    Inner-join a single return series with factor data on date, and convert
    the asset returns to excess returns (asset - RF).

    asset_returns : the return series (one column), simple returns
    return_dates  : matching dates (pl.Series of Date)
    factor_wide   : wide factor DataFrame (date | Mkt-RF | ... | RF)
    factor_names  : which columns to use as regressors (excludes RF)
    """
    if isinstance(asset_returns, np.ndarray):
        asset_returns = pl.Series("ret", asset_returns)

    ret_df = pl.DataFrame({"date": return_dates, "ret": asset_returns})

    needed = factor_names + [rf_column]
    missing = [c for c in needed if c not in factor_wide.columns]
    if missing:
        raise ValueError(f"Factor columns missing from store: {missing}")

    merged = ret_df.join(
        factor_wide.select(["date"] + needed), on="date", how="inner"
    ).drop_nulls().sort("date")

    if merged.is_empty():
        raise ValueError("No overlapping dates between returns and factors")

    # --- Make the date-coverage gap VISIBLE rather than silent ---
    # The French library publishes with a lag, so factor data typically ends
    # weeks before the price data. The inner-join silently drops those dates;
    # we surface it so the factor analysis window is never misread.
    n_returns = len(ret_df)
    n_aligned = len(merged)
    dropped = n_returns - n_aligned
    if dropped > 0:
        ret_end = ret_df["date"].max()
        fac_end = factor_wide["date"].max()
        pct = dropped / n_returns
        logger.warning(
            f"Factor alignment dropped {dropped}/{n_returns} observations "
            f"({pct:.1%}). Returns end {ret_end}, factor data ends {fac_end}. "
            f"Factor analysis covers {merged['date'].min()} → {merged['date'].max()}; "
            f"the most recent {dropped} return days are NOT in the factor model."
        )

    excess = (merged["ret"] - merged[rf_column]).to_numpy()
    factors = merged.select(factor_names).to_numpy()
    rf = merged[rf_column].to_numpy()

    return AlignedFactorData(
        dates=merged["date"].to_list(),
        excess_returns=excess,
        factors=factors,
        factor_names=factor_names,
        rf=rf,
    )
