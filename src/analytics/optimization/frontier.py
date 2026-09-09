"""
Efficient frontier.

Trace the efficient half of the mean-variance frontier — the set of minimum-
variance portfolios for a grid of target returns:

    min  wᵀ Σ w   s.t.   Σ wᵢ = 1,   μᵀ w = target,   (w ≥ 0 if long_only)

Unlike `min_variance`, the frontier needs expected returns μ — the noisy input
min-variance deliberately avoids. So μ is passed in explicitly; the caller owns
where it comes from (sample mean for illustration, Black-Litterman, or explicit
views) and the caveat that historical means are a poor forecast.

The lowest-return anchor of the efficient half is the global minimum-variance
portfolio (GMV); the frontier is traced from there up to the highest expected
return reachable under the constraints. Returns the annualised risk/return
coordinates plus the weights at each point, ready for `plot_efficient_frontier`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analytics.optimization.constraints import Constraints
from src.analytics.optimization.minimum_variance import _unpack, min_variance


@dataclass
class EfficientFrontier:
    tickers: list[str]
    volatilities: np.ndarray       # annualised σ per frontier point
    returns: np.ndarray            # annualised target returns per point
    weights: np.ndarray            # (n_points, N), each row sums to 1
    long_only: bool

    def __len__(self) -> int:
        return len(self.returns)

    @property
    def min_variance_point(self) -> tuple[float, float]:
        """(vol, return) of the leftmost (global minimum-variance) point."""
        i = int(np.argmin(self.volatilities))
        return float(self.volatilities[i]), float(self.returns[i])


def efficient_frontier(
    cov,
    mu: np.ndarray,
    constraints: Constraints | None = None,
    *,
    n_points: int = 40,
    tickers: list[str] | None = None,
) -> EfficientFrontier:
    """
    cov         : CovarianceResult (preferred) or a square Σ. Must be in the
                  SAME units as `mu` (both annualised, or both per-period).
    mu          : expected returns aligned to the covariance's assets.
    constraints : the same mandate every other optimiser takes.
    n_points    : portfolios traced from the GMV return up to the max reachable.
    """
    from scipy.optimize import minimize

    constraints = constraints or Constraints()
    Sigma, tickers = _unpack(cov, tickers)
    n = Sigma.shape[0]
    constraints.validate(n)
    mu = np.asarray(mu, dtype=float)
    if len(mu) != n:
        raise ValueError(f"mu length {len(mu)} != Σ dimension {n}")

    # Anchor the efficient half at the global minimum-variance portfolio.
    gmv = min_variance(Sigma, constraints, tickers=tickers)
    lo = float(gmv.weights @ mu)
    # Highest reachable expected return: the best single asset under long-only;
    # cap long-short at the same level so the chart stays meaningful.
    hi = float(mu.max())
    if hi <= lo + 1e-12:
        # Degenerate (e.g. GMV already at the max-return asset): single point.
        return EfficientFrontier(tickers, np.array([gmv.expected_volatility]),
                                 np.array([lo]), gmv.weights[None, :],
                                 constraints.long_only)

    targets = np.linspace(lo, hi, n_points)
    bounds = constraints.bounds(n)

    vols, rets, ws = [], [], []
    x0 = gmv.weights.copy()
    for m in targets:
        cons = (
            constraints.budget_constraint(),
            {"type": "eq", "fun": lambda w, m=m: w @ mu - m, "jac": lambda w: mu},
        )
        res = minimize(
            fun=lambda w: float(w @ Sigma @ w),
            x0=x0, jac=lambda w: 2.0 * Sigma @ w,
            method="SLSQP", bounds=bounds, constraints=cons,
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not res.success:
            continue
        w = res.x
        if constraints.long_only:
            w = np.clip(w, constraints.min_weight, None)
            w = w * (constraints.budget / w.sum())
        x0 = w                                   # warm-start the next point
        vols.append(float(np.sqrt(max(w @ Sigma @ w, 0.0))))
        rets.append(float(w @ mu))
        ws.append(w)

    return EfficientFrontier(
        tickers=tickers,
        volatilities=np.array(vols),
        returns=np.array(rets),
        weights=np.array(ws),
        long_only=constraints.long_only,
    )
