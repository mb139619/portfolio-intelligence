"""
Minimum-variance portfolio.

Solve   min  wᵀ Σ w   subject to the declared `Constraints`.

Two regimes, deliberately distinct:

  unbounded long/short : the global minimum-variance portfolio has a closed
                         form, w = Σ⁻¹1 / (1ᵀ Σ⁻¹ 1). The feasible set is
                         larger, so its variance is ≤ the long-only one — but
                         with no w ≥ 0 the solution is highly sensitive to
                         estimation error in Σ, producing large offsetting
                         legs. USE WITH SHRUNK Σ.

  anything bounded      : solved as a QP via SLSQP. Non-negativity acts as an
                         implicit regulariser (Jagannathan & Ma 2003), which is
                         why long-only is the sane default.

The optimiser needs only Σ — no expected returns — which is exactly why
minimum variance is robust relative to full mean-variance: μ is the noisiest
input and the one that drives error maximisation.
"""

from __future__ import annotations

import numpy as np

from src.analytics.optimization.constraints import Constraints
from src.analytics.optimization.result import OptimizationResult


def _unpack(cov, tickers):
    """Accept a CovarianceResult or a raw (Σ, tickers) pair."""
    if hasattr(cov, "matrix"):
        Sigma = np.asarray(cov.matrix, dtype=float)
        tickers = list(cov.tickers) if tickers is None else tickers
    else:
        Sigma = np.asarray(cov, dtype=float)
        if tickers is None:
            tickers = [f"A{i}" for i in range(Sigma.shape[0])]
    if Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("covariance must be square")
    if len(tickers) != Sigma.shape[0]:
        raise ValueError(f"{len(tickers)} tickers != Σ dimension {Sigma.shape[0]}")
    return Sigma, list(tickers)


def min_variance(
    cov,
    constraints: Constraints | None = None,
    *,
    tickers: list[str] | None = None,
) -> OptimizationResult:
    """
    Minimum-variance weights under `constraints` (long-only and fully
    invested by default).
    """
    constraints = constraints or Constraints()
    Sigma, tickers = _unpack(cov, tickers)
    n = Sigma.shape[0]
    constraints.validate(n)
    ones = np.ones(n)

    # Closed form only where the feasible set is genuinely unbounded; any box
    # constraint makes the analytic solution wrong rather than merely loose.
    unbounded = (not constraints.long_only
                 and constraints.max_weight is None
                 and constraints.min_weight == 0.0)
    if unbounded:
        z = np.linalg.solve(Sigma, ones)          # solve, don't invert
        w = constraints.budget * z / (ones @ z)
        return _build(Sigma, tickers, w,
                      "min_variance (long-short, closed-form)",
                      constraints, success=True)

    from scipy.optimize import minimize

    res = minimize(
        fun=lambda w: float(w @ Sigma @ w),
        x0=constraints.start(n),
        jac=lambda w: 2.0 * Sigma @ w,
        method="SLSQP",
        bounds=constraints.bounds(n),
        constraints=(constraints.budget_constraint(),),
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = res.x
    if constraints.long_only:
        w = np.clip(w, constraints.min_weight, None)
    total = w.sum()
    if total:
        w = w * (constraints.budget / total)       # clean up SLSQP residuals
    return _build(Sigma, tickers, w, "min_variance (QP)", constraints,
                  success=bool(res.success), message=str(res.message),
                  n_iter=int(res.nit))


def _build(Sigma, tickers, w, method, constraints, *,
           success, message="", n_iter=None):
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    return OptimizationResult(
        method=method, tickers=tickers, weights=w,
        expected_volatility=vol, success=success,
        message=message, n_iter=n_iter, constraints=constraints,
    )
