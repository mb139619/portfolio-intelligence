"""
Risk Decomposition.

For portfolio weights w and annualised covariance Sigma:
  variance: sigma2_p = w' Sigma w
  MCR_i = (Sigma w)_i / sigma_p          marginal contribution to risk
  RC_i  = w_i * MCR_i                     risk contribution
  %RC_i = RC_i / sigma_p                  percent contribution (sums to 1)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries


@dataclass
class RiskDecomposition:
    tickers: list[str]
    weights: np.ndarray
    portfolio_volatility: float
    marginal_contribution: np.ndarray
    risk_contribution: np.ndarray
    percent_contribution: np.ndarray

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({
            "ticker": self.tickers,
            "weight": self.weights.tolist(),
            "mcr": self.marginal_contribution.tolist(),
            "risk_contribution": self.risk_contribution.tolist(),
            "pct_risk_contribution": self.percent_contribution.tolist(),
        })

    def summary(self) -> str:
        df = self.to_dataframe().sort("pct_risk_contribution", descending=True)
        lines = [
            f"-- Risk Decomposition (sigma_p = {self.portfolio_volatility:.2%}) --",
            f"  {'Ticker':<8} {'Weight':>8} {'%RC':>8}  {'concentration':>12}",
        ]
        for row in df.iter_rows(named=True):
            flag = " <-- " if row["pct_risk_contribution"] > 1.5 * row["weight"] else ""
            lines.append(
                f"  {row['ticker']:<8} {row['weight']:>8.1%} "
                f"{row['pct_risk_contribution']:>8.1%}{flag}"
            )
        return "\n".join(lines)


def covariance_matrix(
    rs: ReturnSeries, method: str = "sample", ppy: int = 252
) -> np.ndarray:
    """
    Annualised covariance matrix. Thin wrapper over the canonical estimator in
    risk.covariance so there is a single source of truth; supports the same
    methods plus "ledoit_wolf_cc" (constant-correlation shrinkage, recommended).
    """
    from src.analytics.risk.covariance import estimate_covariance
    return estimate_covariance(rs, method=method, ppy=ppy, annualize=True).matrix


def decompose_risk(weights: dict[str, float], rs: ReturnSeries,
                   cov_method: str = "sample", ppy: int = 252) -> RiskDecomposition:
    tickers = list(weights.keys())
    w = np.array([weights[t] for t in tickers])
    aligned = rs.select(tickers)
    cov = covariance_matrix(aligned, method=cov_method, ppy=ppy)
    port_vol = float(np.sqrt(w @ cov @ w))
    if port_vol == 0:
        raise ValueError("Portfolio volatility is zero")
    mcr = (cov @ w) / port_vol
    rc = w * mcr
    return RiskDecomposition(
        tickers=tickers, weights=w, portfolio_volatility=port_vol,
        marginal_contribution=mcr, risk_contribution=rc,
        percent_contribution=rc / port_vol,
    )


def rolling_risk_contribution(weights: dict[str, float], rs: ReturnSeries,
                              window: int = 63, cov_method: str = "ewma",
                              ppy: int = 252) -> pl.DataFrame:
    tickers = list(weights.keys())
    aligned = rs.select(tickers)
    dates = aligned.dates.to_list()
    rows = []
    for i in range(window, aligned.n_obs + 1):
        window_rs = ReturnSeries(
            data=aligned.data.slice(i - window, window), tickers=tickers
        )
        try:
            d = decompose_risk(weights, window_rs, cov_method, ppy)
            for t, pct in zip(tickers, d.percent_contribution):
                rows.append({"date": dates[i - 1], "ticker": t,
                             "pct_rc": float(pct),
                             "portfolio_vol": d.portfolio_volatility})
        except Exception:
            pass
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
