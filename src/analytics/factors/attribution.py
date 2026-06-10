"""
Factor-based risk decomposition and return attribution.

This is the institutional payoff of the factor model. Given the estimated
betas B and the factor covariance Σ_f, total variance splits cleanly:

    Var(r) = B' Σ_f B   +   Var(eps)
             └ systematic ┘   └ specific ┘

and the systematic part decomposes per factor:

    contribution of factor i  =  beta_i * (Σ_f B)_i      (these sum to B' Σ_f B)

This answers: "How much of my risk is the market? Value? Idiosyncratic?"
— the question Barra/Axioma models are built around.

Return attribution is the analogue for performance:

    E[r] ≈ alpha + sum_i beta_i * E[f_i]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analytics.factors.engine import FactorModel
from src.analytics.factors.prepare import AlignedFactorData


@dataclass
class FactorRiskDecomposition:
    factor_names: list[str]
    total_vol: float                       # annualised
    systematic_vol: float
    specific_vol: float
    pct_systematic: float                  # systematic_var / total_var
    pct_specific: float
    factor_variance_contribution: dict[str, float]   # per-factor share of TOTAL variance
    factor_vol_contribution: dict[str, float]        # per-factor, in vol units

    def summary(self) -> str:
        lines = [
            f"-- Factor Risk Decomposition (σ = {self.total_vol:.2%}) --",
            f"  Systematic  {self.pct_systematic:>6.1%}  (vol {self.systematic_vol:.2%})",
            f"  Specific    {self.pct_specific:>6.1%}  (vol {self.specific_vol:.2%})",
            f"  {'factor':<10} {'% of total var':>14}",
        ]
        for f in self.factor_names:
            lines.append(f"  {f:<10} {self.factor_variance_contribution[f]:>14.1%}")
        return "\n".join(lines)


def factor_covariance(
    aligned: AlignedFactorData,
    periods_per_year: int = 252,
) -> np.ndarray:
    """Annualised sample covariance of the factor returns."""
    return np.cov(aligned.factors, rowvar=False, ddof=1) * periods_per_year


def decompose_factor_risk(
    model: FactorModel,
    aligned: AlignedFactorData,
    periods_per_year: int = 252,
) -> FactorRiskDecomposition:
    """
    Split total variance into systematic (factor) and specific (idiosyncratic),
    and attribute the systematic part to individual factors.
    """
    b = model.beta_vector()
    cov_f = factor_covariance(aligned, periods_per_year)

    systematic_var = float(b @ cov_f @ b)
    specific_var = model.idiosyncratic_vol ** 2
    total_var = systematic_var + specific_var

    if total_var <= 0:
        raise ValueError("Total variance is non-positive")

    # Per-factor contribution to systematic variance: beta_i * (Σ_f b)_i
    marginal = b * (cov_f @ b)            # sums to systematic_var
    var_contrib = {
        f: float(marginal[i] / total_var)        # as share of TOTAL variance
        for i, f in enumerate(model.factor_names)
    }
    vol_contrib = {
        f: float(marginal[i] / np.sqrt(total_var)) if total_var > 0 else 0.0
        for i, f in enumerate(model.factor_names)
    }

    return FactorRiskDecomposition(
        factor_names=model.factor_names,
        total_vol=float(np.sqrt(total_var)),
        systematic_vol=float(np.sqrt(systematic_var)),
        specific_vol=float(np.sqrt(specific_var)),
        pct_systematic=systematic_var / total_var,
        pct_specific=specific_var / total_var,
        factor_variance_contribution=var_contrib,
        factor_vol_contribution=vol_contrib,
    )


@dataclass
class ReturnAttribution:
    factor_names: list[str]
    total_excess_annualized: float           # arithmetic mean excess × ppy
    alpha_annualized: float
    factor_contribution: dict[str, float]    # annualised return from each factor
    reconciliation_error: float              # total - alpha - sum(factors) ≈ 0

    def summary(self) -> str:
        lines = [
            f"-- Return Attribution (total excess ann. = {self.total_excess_annualized:.2%}) --",
            f"  alpha       {self.alpha_annualized:>8.2%}",
            f"  {'factor':<10} {'contribution':>14}",
        ]
        for f in self.factor_names:
            lines.append(f"  {f:<10} {self.factor_contribution[f]:>14.2%}")
        lines.append(f"  {'(check)':<10} {self.reconciliation_error:>14.2e}")
        return "\n".join(lines)


def attribute_returns(
    model: FactorModel,
    aligned: AlignedFactorData,
    periods_per_year: int = 252,
) -> ReturnAttribution:
    """
    Decompose realised excess return into alpha + factor contributions.

    Uses the ARITHMETIC decomposition, which reconciles EXACTLY because for an
    OLS fit with an intercept the residual mean is zero:

        mean(r_excess) = alpha + sum_k beta_k * mean(f_k)

    Annualising by × ppy preserves the identity, so alpha + sum(factor
    contributions) equals the total annualised excess return to machine
    precision (reconciliation_error ≈ 0). This is the property an interviewer
    will probe ("do your contributions add up?") — here they do, by construction.
    """
    factor_means = aligned.factors.mean(axis=0)   # daily mean of each factor
    b = model.beta_vector()

    factor_contrib = {
        f: float(b[i] * factor_means[i] * periods_per_year)
        for i, f in enumerate(model.factor_names)
    }

    total_excess_ann = float(aligned.excess_returns.mean() * periods_per_year)
    alpha_ann = float(model.alpha * periods_per_year)

    recon_error = total_excess_ann - alpha_ann - sum(factor_contrib.values())

    return ReturnAttribution(
        factor_names=model.factor_names,
        total_excess_annualized=total_excess_ann,
        alpha_annualized=alpha_ann,
        factor_contribution=factor_contrib,
        reconciliation_error=recon_error,
    )
