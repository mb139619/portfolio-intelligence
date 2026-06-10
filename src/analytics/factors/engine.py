"""
Factor Engine — time-series factor model estimation.

Model:   r_excess(t) = alpha + sum_k beta_k * f_k(t) + eps(t)

Estimation:
  - OLS with an intercept (alpha)
  - Optional Newey-West (HAC) standard errors — recommended for daily data,
    which exhibits autocorrelation and heteroskedasticity that plain OLS
    standard errors understate.

Outputs everything a quant needs to judge the fit:
  betas + t-stats + p-values, alpha (annualised), R² / adj-R²,
  residuals, and annualised idiosyncratic (specific) volatility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.analytics.factors.prepare import AlignedFactorData


@dataclass
class FactorModel:
    factor_names: list[str]

    alpha: float                       # daily intercept
    alpha_annualized: float
    alpha_tstat: float
    alpha_pvalue: float

    betas: dict[str, float]
    beta_tstats: dict[str, float]
    beta_pvalues: dict[str, float]

    r_squared: float
    adj_r_squared: float

    residuals: np.ndarray = field(repr=False)
    fitted: np.ndarray = field(repr=False)
    idiosyncratic_vol: float = 0.0     # annualised std of residuals

    n_obs: int = 0
    hac_lags: int | None = None

    def beta_vector(self) -> np.ndarray:
        return np.array([self.betas[f] for f in self.factor_names])

    def summary(self) -> str:
        lines = [
            f"-- Factor Model ({self.n_obs} obs"
            + (f", HAC({self.hac_lags})" if self.hac_lags else ", OLS SE") + ") --",
            f"  R² = {self.r_squared:.3f}   adj-R² = {self.adj_r_squared:.3f}"
            f"   idio vol = {self.idiosyncratic_vol:.2%}",
            f"  alpha (ann.) = {self.alpha_annualized:>8.2%}  "
            f"(t={self.alpha_tstat:+.2f}, p={self.alpha_pvalue:.3f})",
            f"  {'factor':<10} {'beta':>8} {'t-stat':>8} {'p-value':>8}",
        ]
        for f in self.factor_names:
            star = " *" if self.beta_pvalues[f] < 0.05 else ""
            lines.append(
                f"  {f:<10} {self.betas[f]:>8.3f} {self.beta_tstats[f]:>8.2f} "
                f"{self.beta_pvalues[f]:>8.3f}{star}"
            )
        return "\n".join(lines)


def estimate_factor_model(
    aligned: AlignedFactorData,
    hac_lags: int | None = 5,
    periods_per_year: int = 252,
) -> FactorModel:
    """
    Estimate r_excess = alpha + B f + eps by OLS.

    hac_lags: if not None, use Newey-West HAC standard errors with this many
              lags. Default 5 (≈ one trading week) is a sensible daily choice.
              Set to None for classical OLS standard errors.
    """
    import statsmodels.api as sm

    y = aligned.excess_returns
    X = aligned.factors
    X_const = sm.add_constant(X)  # adds intercept column at position 0

    model = sm.OLS(y, X_const)
    if hac_lags is not None:
        results = model.fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    else:
        results = model.fit()

    params = results.params       # [alpha, beta_1, ..., beta_k]
    tvals = results.tvalues
    pvals = results.pvalues

    alpha = float(params[0])
    betas = {f: float(params[i + 1]) for i, f in enumerate(aligned.factor_names)}
    beta_t = {f: float(tvals[i + 1]) for i, f in enumerate(aligned.factor_names)}
    beta_p = {f: float(pvals[i + 1]) for i, f in enumerate(aligned.factor_names)}

    residuals = results.resid
    idio_vol = float(np.std(residuals, ddof=len(params)) * np.sqrt(periods_per_year))

    # Annualise alpha geometrically
    alpha_ann = float((1 + alpha) ** periods_per_year - 1)

    return FactorModel(
        factor_names=aligned.factor_names,
        alpha=alpha,
        alpha_annualized=alpha_ann,
        alpha_tstat=float(tvals[0]),
        alpha_pvalue=float(pvals[0]),
        betas=betas,
        beta_tstats=beta_t,
        beta_pvalues=beta_p,
        r_squared=float(results.rsquared),
        adj_r_squared=float(results.rsquared_adj),
        residuals=np.asarray(residuals),
        fitted=np.asarray(results.fittedvalues),
        idiosyncratic_vol=idio_vol,
        n_obs=aligned.n_obs,
        hac_lags=hac_lags,
    )


def rolling_betas(
    aligned: AlignedFactorData,
    window: int = 252,
) -> dict:
    """
    Rolling factor betas — the basis for Dynamic Factor Evolution.

    Returns a dict with:
      'dates'  : list of window-end dates
      'betas'  : dict[factor_name -> np.ndarray of betas over time]
      'alpha'  : np.ndarray of rolling alpha (daily)
      'r2'     : np.ndarray of rolling R²
    Uses plain OLS per window (fast; standard errors not needed for the path).
    """
    y = aligned.excess_returns
    X = aligned.factors
    n, k = X.shape
    dates = aligned.dates

    out_dates = []
    betas = {f: [] for f in aligned.factor_names}
    alphas, r2s = [], []

    X_const = np.column_stack([np.ones(n), X])

    for i in range(window, n + 1):
        xs = X_const[i - window:i]
        ys = y[i - window:i]
        # Least squares: (X'X)^-1 X'y
        coef, _, _, _ = np.linalg.lstsq(xs, ys, rcond=None)
        resid = ys - xs @ coef
        ss_res = float(resid @ resid)
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        out_dates.append(dates[i - 1])
        alphas.append(float(coef[0]))
        for j, f in enumerate(aligned.factor_names):
            betas[f].append(float(coef[j + 1]))
        r2s.append(r2)

    return {
        "dates": out_dates,
        "betas": {f: np.array(v) for f, v in betas.items()},
        "alpha": np.array(alphas),
        "r2": np.array(r2s),
    }
