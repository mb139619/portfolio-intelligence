"""
Hidden Concentration Detector.

A portfolio can hold many names with well-spread weights and still be, in
risk terms, a single bet. This module measures that gap rigorously.

Method (Meucci, "Managing Diversification"):
  Decompose portfolio variance along the principal components. With covariance
  Σ = V Λ V', weights w, and PC exposures e = V'w:

      σ²_p = w'Σw = Σ_i λ_i e_i²

  Let p_i = λ_i e_i² / σ²_p  be the fraction of portfolio variance coming from
  principal component i. Then the Effective Number of Bets is the exponential
  of the entropy of {p_i}:

      ENB = exp( − Σ_i p_i ln p_i )

  ENB ranges from 1 (all risk in one PC — a hidden single bet) to N (risk
  spread equally across all independent components — true diversification).

The punchline is the contrast between:
  - naive diversification:  ENB_weights = 1 / Σ w_i²   (inverse Herfindahl)
  - true risk diversification: ENB_risk (above)
A large gap means hidden concentration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries
from src.analytics.pca.model import fit_pca, PCARiskModel


@dataclass
class ConcentrationReport:
    tickers: list[str]
    n_holdings: int
    effective_bets_weights: float          # 1 / sum(w_i^2)
    effective_bets_risk: float             # PCA entropy-based ENB
    concentration_gap: float               # weights ENB - risk ENB
    pc_variance_contribution: np.ndarray   # p_i, sums to 1
    top_pc_share: float                    # p_1
    portfolio_volatility_annualized: float

    def pc_contribution_dataframe(self) -> pl.DataFrame:
        n = len(self.pc_variance_contribution)
        return pl.DataFrame({
            "pc": [f"PC{i+1}" for i in range(n)],
            "variance_share": self.pc_variance_contribution.tolist(),
        })

    def verdict(self) -> str:
        ratio = (self.effective_bets_risk / self.effective_bets_weights
                 if self.effective_bets_weights > 0 else 1.0)
        if ratio < 0.5:
            return "HIGH hidden concentration — risk far more concentrated than weights suggest"
        if ratio < 0.75:
            return "MODERATE hidden concentration"
        return "LOW hidden concentration — risk roughly as diversified as weights"

    def summary(self) -> str:
        return "\n".join([
            "-- Hidden Concentration Detector --",
            f"  Holdings                 {self.n_holdings:>8d}",
            f"  Effective bets (weights) {self.effective_bets_weights:>8.2f}",
            f"  Effective bets (risk)    {self.effective_bets_risk:>8.2f}",
            f"  Top PC variance share    {self.top_pc_share:>8.1%}",
            f"  Portfolio vol (ann.)     {self.portfolio_volatility_annualized:>8.2%}",
            f"  -> {self.verdict()}",
        ])


def detect_hidden_concentration(
    weights: dict[str, float],
    rs: ReturnSeries,
    method: str = "covariance",
    ppy: int = 252,
    model: PCARiskModel | None = None,
) -> ConcentrationReport:
    """
    Run the Hidden Concentration Detector for a portfolio.

    weights : {ticker: weight}
    rs      : ReturnSeries containing (at least) the portfolio tickers
    model   : optionally pass a pre-fitted PCARiskModel to avoid recomputing
    """
    tickers = list(weights.keys())
    aligned = rs.select(tickers)
    w = np.array([weights[t] for t in tickers])

    if model is None:
        model = fit_pca(aligned, method=method, ppy=ppy)

    # PC exposures of the portfolio: e = V' w
    e = model.loadings.T @ w               # (N,)
    lam = model.eigenvalues                # daily eigenvalues

    pc_var = lam * e ** 2                  # variance from each PC
    total_var = pc_var.sum()
    if total_var <= 0:
        raise ValueError("Portfolio variance is non-positive")

    p = pc_var / total_var                 # shares, sum to 1

    # Effective number of bets = exp(entropy)
    p_nonzero = p[p > 1e-15]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    enb_risk = float(np.exp(entropy))

    # Naive weight-based effective N (inverse Herfindahl)
    enb_weights = float(1.0 / np.sum(w ** 2))

    return ConcentrationReport(
        tickers=tickers,
        n_holdings=len(tickers),
        effective_bets_weights=enb_weights,
        effective_bets_risk=enb_risk,
        concentration_gap=enb_weights - enb_risk,
        pc_variance_contribution=p,
        top_pc_share=float(p[0]),
        portfolio_volatility_annualized=float(np.sqrt(total_var * ppy)),
    )
