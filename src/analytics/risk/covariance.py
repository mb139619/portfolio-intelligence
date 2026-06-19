"""
Covariance estimation — the single canonical source for Σ.

Every downstream consumer (risk decomposition, the optimiser, PCA-style work)
should obtain its covariance from here rather than calling ``np.cov`` directly,
so that the choice of estimator, the annualisation convention and the
conditioning diagnostics live in one place.

Why this matters for optimisation: the sample covariance is noisy and, as the
universe grows relative to the (effective) sample, ill-conditioned. A mean-
variance or min-variance optimiser inverts Σ and will happily amplify that
estimation error into extreme, unstable weights ("error maximisation"). Shrinkage
(Ledoit-Wolf) pulls the estimate toward a structured target, trading a little
bias for a large variance reduction and a well-conditioned matrix.

Backends
--------
  sample          : np.cov, ddof=1. Baseline; no conditioning guard.
  ledoit_wolf     : Ledoit-Wolf shrinkage toward a scaled identity (sklearn).
                    This is the previous default behaviour, kept for continuity.
  ledoit_wolf_cc  : Ledoit-Wolf shrinkage toward the CONSTANT-CORRELATION target
                    (Ledoit & Wolf 2003). Usually the better target in finance,
                    where assets have very different volatilities. RECOMMENDED.
  ewma            : RiskMetrics zero-mean EWMA (current-state covariance). Short
                    effective memory (~1/(1-λ) obs); fine for monitoring, but a
                    poor optimiser input for non-trivial universes — see note.

Factor-implied covariance (Σ = B Λ Bᵀ + D) lives in ``factor_covariance``: it
takes asset and factor returns rather than a ReturnSeries, so it has its own
constructor but returns the same ``CovarianceResult``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.analytics.correlation.matrices import (
    correlation_from_covariance,
    ewma_covariance,
)
from src.domain.returns import ReturnSeries


@dataclass
class CovarianceResult:
    """A covariance estimate plus the metadata needed to use it safely."""

    tickers: list[str]
    matrix: np.ndarray          # annualised iff `annualized`, else daily
    method: str
    annualized: bool
    ppy: int
    n_obs: int
    shrinkage: float | None = None   # shrinkage intensity δ for LW backends

    # --- derived quantities ---

    @property
    def volatilities(self) -> np.ndarray:
        """Per-asset volatilities = sqrt of the diagonal (annualised iff matrix is)."""
        return np.sqrt(np.clip(np.diag(self.matrix), 0.0, None))

    @property
    def correlation(self) -> np.ndarray:
        return correlation_from_covariance(self.matrix)

    @property
    def condition_number(self) -> float:
        """κ(Σ). Large (say > 1e4) means near-singular → unstable to invert."""
        return float(np.linalg.cond(self.matrix))

    def min_eigenvalue(self) -> float:
        return float(np.linalg.eigvalsh(self.matrix).min())

    def is_psd(self, tol: float = 1e-10) -> bool:
        return self.min_eigenvalue() >= -tol

    def to_dataframe(self) -> pl.DataFrame:
        df = pl.DataFrame({"ticker": self.tickers})
        for j, t in enumerate(self.tickers):
            df = df.with_columns(pl.Series(t, self.matrix[:, j].tolist()))
        return df

    def summary(self) -> str:
        return "\n".join([
            f"-- Covariance ({self.method}, {len(self.tickers)} assets, "
            f"{self.n_obs} obs{', annualised' if self.annualized else ''}) --",
            f"  condition number  {self.condition_number:>12.1f}",
            f"  min eigenvalue    {self.min_eigenvalue():>12.2e}  "
            f"(PSD: {self.is_psd()})",
            *([] if self.shrinkage is None
              else [f"  shrinkage δ       {self.shrinkage:>12.3f}"]),
        ])


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

def estimate_covariance(
    rs: ReturnSeries,
    method: str = "ledoit_wolf_cc",
    ppy: int = 252,
    annualize: bool = True,
    ewma_lambda: float = 0.94,
    ridge: float = 0.0,
) -> CovarianceResult:
    """
    Estimate the covariance matrix of a ReturnSeries.

    method     : one of {"sample", "ledoit_wolf", "ledoit_wolf_cc", "ewma"}.
    annualize  : multiply by `ppy` (covariance scales linearly with horizon).
    ridge      : optional λ added to the diagonal (in DAILY units, before
                 annualisation) as a last-resort conditioning guard. Default 0.
    """
    R = rs.to_numpy()
    T = R.shape[0]
    shrinkage: float | None = None

    if method == "sample":
        cov = np.cov(R, rowvar=False, ddof=1)
    elif method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf().fit(R)
        cov = lw.covariance_
        shrinkage = float(lw.shrinkage_)
    elif method == "ledoit_wolf_cc":
        cov, shrinkage = _ledoit_wolf_constant_correlation(R)
    elif method == "ewma":
        cov = ewma_covariance(R, lam=ewma_lambda)
    else:
        raise ValueError(f"Unknown covariance method: {method!r}")

    cov = _symmetrize(cov)
    if ridge > 0.0:
        cov = cov + ridge * np.eye(cov.shape[0])
    if annualize:
        cov = cov * ppy

    return CovarianceResult(
        tickers=list(rs.tickers), matrix=cov, method=method,
        annualized=annualize, ppy=ppy, n_obs=T, shrinkage=shrinkage,
    )


def factor_covariance(
    asset_returns: np.ndarray,
    factor_returns: np.ndarray,
    tickers: list[str],
    ppy: int = 252,
    annualize: bool = True,
) -> CovarianceResult:
    """
    Factor-implied covariance:  Σ = B Λ Bᵀ + D

    where B (N×K) are the asset factor loadings (OLS, intercept dropped),
    Λ (K×K) is the factor covariance, and D is the diagonal idiosyncratic
    (residual) covariance. PSD by construction (Λ and D are PSD).

    asset_returns  : (T, N) asset returns aligned to...
    factor_returns : (T, K) factor returns (same dates, same T).
    """
    Y = np.asarray(asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)
    if Y.shape[0] != F.shape[0]:
        raise ValueError(
            f"asset T={Y.shape[0]} != factor T={F.shape[0]}; align on date first"
        )
    T, N = Y.shape
    K = F.shape[1]

    X = np.column_stack([np.ones(T), F])           # intercept + factors
    coef, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)   # (K+1, N)
    B = coef[1:].T                                  # (N, K), drop intercept
    resid = Y - X @ coef                            # (T, N)

    Lambda = np.cov(F, rowvar=False, ddof=1)        # (K, K)
    Lambda = np.atleast_2d(Lambda)
    D = np.diag(resid.var(axis=0, ddof=K + 1))      # (N, N), DoF = T-(K+1)

    cov = _symmetrize(B @ Lambda @ B.T + D)
    if annualize:
        cov = cov * ppy

    return CovarianceResult(
        tickers=list(tickers), matrix=cov, method="factor",
        annualized=annualize, ppy=ppy, n_obs=T, shrinkage=None,
    )


# ──────────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────────

def _symmetrize(m: np.ndarray) -> np.ndarray:
    return 0.5 * (m + m.T)


def _ledoit_wolf_constant_correlation(R: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Ledoit & Wolf (2003) shrinkage toward the constant-correlation target.

    Returns (shrunk DAILY covariance, shrinkage intensity δ ∈ [0, 1]).

    Target F: keeps each asset's sample variance, but replaces every pairwise
    correlation with the average sample correlation r̄. The optimal δ minimises
    expected Frobenius distance to the true covariance:  δ* = (π − ρ) / γ, /T.
    """
    x = R - R.mean(axis=0)
    t, n = x.shape
    if n < 2:
        # No correlation structure to shrink; fall back to the sample variance.
        s = np.cov(R, rowvar=False, ddof=1).reshape(n, n)
        return s, 0.0

    sample = (x.T @ x) / t                       # MLE sample covariance (1/T)
    var = np.diag(sample)
    sqrtvar = np.sqrt(var)
    outer_sqrt = np.outer(sqrtvar, sqrtvar)

    # average off-diagonal sample correlation r̄
    corr = sample / outer_sqrt
    r_bar = (corr.sum() - n) / (n * (n - 1))

    # constant-correlation target F
    prior = r_bar * outer_sqrt
    np.fill_diagonal(prior, var)

    # π̂ : sum of asymptotic variances of the sample covariance entries
    y = x ** 2
    phi_mat = (y.T @ y) / t - sample ** 2
    pi_hat = phi_mat.sum()

    # ρ̂ : asymptotic covariance between target and sample entries
    #   diagonal (i=j): f_ii = s_ii  → contributes Σ_i π_ii
    #   off-diagonal:   f_ij = r̄ √(s_ii s_jj)
    cube_cross = ((x ** 3).T @ x) / t            # [i,j] = E[x_i^3 x_j]
    theta_ii = cube_cross - var[:, None] * sample        # ϑ_ii,ij
    theta_jj = cube_cross.T - var[None, :] * sample      # ϑ_jj,ij
    ratio = np.outer(sqrtvar, 1.0 / sqrtvar)             # [i,j] = √(s_ii/s_jj)
    off = (r_bar / 2.0) * (ratio * theta_jj + (1.0 / ratio) * theta_ii)
    np.fill_diagonal(off, 0.0)
    rho_hat = np.diag(phi_mat).sum() + off.sum()

    # γ̂ : misspecification of the target
    gamma_hat = ((prior - sample) ** 2).sum()

    kappa = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 0.0
    delta = float(np.clip(kappa / t, 0.0, 1.0))

    # Shrink. Rescale the sample part to the unbiased (ddof=1) estimator so the
    # un-shrunk limit (δ=0) matches the "sample" backend.
    sample_unbiased = sample * t / (t - 1)
    sigma = delta * prior + (1.0 - delta) * sample_unbiased
    return _symmetrize(sigma), delta
