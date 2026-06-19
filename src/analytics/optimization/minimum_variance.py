"""
Minimum-variance portfolio.

Solve   min  wᵀ Σ w   subject to   Σ wᵢ = 1   and, optionally,   w ≥ 0.

Two regimes, deliberately distinct (see the short-selling discussion):

  long_only=False : the global minimum-variance portfolio has a closed form,
                    w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1). The feasible set is larger, so its
                    variance is ≤ the long-only one — but with no w ≥ 0 the
                    solution is highly sensitive to estimation error in Σ
                    (large offsetting long/short legs). USE WITH SHRUNK Σ.

  long_only=True  : add w ≥ 0 (and an optional per-name cap). No closed form;
                    solved as a QP via SLSQP. The non-negativity constraint acts
                    as an implicit regulariser (Jagannathan & Ma 2003), which is
                    why this is the sane default.

The optimiser only needs Σ — no expected returns — which is exactly why min-
variance is robust relative to full mean-variance.
"""

from __future__ import annotations

import numpy as np

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
        raise ValueError(
            f"{len(tickers)} tickers != Σ dimension {Sigma.shape[0]}"
        )
    return Sigma, list(tickers)


def min_variance(
    cov,
    *,
    long_only: bool = True,
    max_weight: float | None = None,
    tickers: list[str] | None = None,
) -> OptimizationResult:
    """
    Minimum-variance weights.

    cov         : a CovarianceResult (preferred) or a square numpy Σ.
    long_only   : if True, enforce w ≥ 0 and solve the QP; otherwise use the
                  closed-form global minimum-variance portfolio.
    max_weight  : optional per-asset cap in [0, 1] (long-only only).
    """
    Sigma, tickers = _unpack(cov, tickers)
    n = Sigma.shape[0]
    ones = np.ones(n)

    if not long_only:
        # Closed form: w = Σ⁻¹1 / (1ᵀΣ⁻¹1). Solve, don't invert.
        z = np.linalg.solve(Sigma, ones)
        w = z / (ones @ z)
        return _build(Sigma, tickers, w, "min_variance (long-short, closed-form)",
                      success=True)

    if max_weight is not None and max_weight * n < 1.0 - 1e-9:
        raise ValueError(
            f"infeasible: max_weight={max_weight} × {n} assets < 1; "
            f"raise the cap to at least {1.0 / n:.3f}"
        )

    from scipy.optimize import minimize

    upper = 1.0 if max_weight is None else float(max_weight)
    bounds = [(0.0, upper)] * n
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0,
                    "jac": lambda w: ones},)

    res = minimize(
        fun=lambda w: float(w @ Sigma @ w),
        x0=ones / n,
        jac=lambda w: 2.0 * Sigma @ w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )
    w = np.clip(res.x, 0.0, None)
    w = w / w.sum()                          # clean up tiny SLSQP residuals
    return _build(Sigma, tickers, w, "min_variance (long-only, QP)",
                  success=bool(res.success), message=str(res.message),
                  n_iter=int(res.nit))


def _build(Sigma, tickers, w, method, *, success, message="", n_iter=None):
    vol = float(np.sqrt(max(w @ Sigma @ w, 0.0)))
    return OptimizationResult(
        method=method, tickers=tickers, weights=w,
        expected_volatility=vol, success=success,
        message=message, n_iter=n_iter,
    )
