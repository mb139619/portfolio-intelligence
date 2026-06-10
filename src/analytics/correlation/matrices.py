"""
Correlation estimators and the correlation→distance transform.

Estimators:
  - pearson   : standard linear correlation
  - spearman  : rank correlation (robust to outliers / non-linearity)
  - ewma      : exponentially weighted (RiskMetrics) — the "dynamic" current-
                state correlation, weighting recent observations more.

Distance:
  Mantegna's metric  d_ij = sqrt(2 (1 - rho_ij))  turns a correlation matrix
  into a proper distance matrix (rho=1 → d=0, rho=0 → d=√2, rho=-1 → d=2).
  This distance is the input to both hierarchical clustering and the MST.
"""

from __future__ import annotations

import numpy as np

from src.domain.returns import ReturnSeries


def correlation_matrix(rs: ReturnSeries, method: str = "pearson") -> np.ndarray:
    """Full-sample correlation matrix (N×N) for the given estimator."""
    R = rs.to_numpy()
    if method == "pearson":
        return np.corrcoef(R, rowvar=False)
    if method == "spearman":
        from scipy.stats import spearmanr
        rho, _ = spearmanr(R)
        # spearmanr returns a scalar for 2 columns; normalise to matrix
        rho = np.atleast_2d(rho)
        if rho.shape[0] != R.shape[1]:
            n = R.shape[1]
            full = np.eye(n)
            full[0, 1] = full[1, 0] = float(np.atleast_1d(rho).ravel()[0])
            return full
        return rho
    if method == "ewma":
        return correlation_from_covariance(ewma_covariance(R))
    raise ValueError(f"Unknown correlation method: {method}")


def ewma_covariance(R: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted covariance matrix (daily, not annualised)."""
    T = R.shape[0]
    excess = R - R.mean(axis=0)
    w = np.array([(1 - lam) * lam ** i for i in range(T - 1, -1, -1)])
    w /= w.sum()
    return np.einsum("t,ti,tj->ij", w, excess, excess)


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Normalise a covariance matrix to a correlation matrix."""
    d = np.sqrt(np.diag(cov))
    d = np.where(d == 0, 1.0, d)
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """
    Mantegna distance: d_ij = sqrt(2 (1 - rho_ij)).
    Returns a symmetric distance matrix with zero diagonal.
    """
    d = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(d, 0.0)
    return d
