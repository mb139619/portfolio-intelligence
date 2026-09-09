"""
Risk parity — equal risk contribution.

Minimum variance asks which portfolio has the least risk and usually answers
"almost all of it in the lowest-volatility asset". Risk parity asks a different
question: which portfolio has every asset contributing the *same amount* of
risk? The answer is far more diversified, and it still needs no expected
returns.

With Σ and weights w, the risk contribution of asset i is

    RC_i = w_i (Σw)_i / σ_p ,   Σ_i RC_i = σ_p

and the target is RC_i = σ_p / n for every i. The objective below minimises the
squared dispersion of the RC around that target, which is zero exactly at the
solution and is well behaved under box constraints — unlike the log-barrier
formulation, which is elegant unconstrained and awkward once caps appear.

What it assumes, and where that bites
-------------------------------------
Risk parity equalises *contributions to variance*. It is therefore a statement
about second moments only, and it inherits every weakness of Σ. Three
consequences worth stating rather than discovering:

  * **It has no view on return.** Equalising risk is not optimal under any
    standard objective unless all assets have identical Sharpe ratios and
    identical correlations. That assumption is rarely argued for; it is
    usually just implied.
  * **It levers up whatever is quiet.** The lowest-volatility asset gets the
    largest weight, so risk parity systematically overweights bonds in a
    falling-rate era and is exposed if that volatility was artificially
    suppressed. Real implementations lever the whole book to a volatility
    target, which this one deliberately does not do.
  * **It is only as good as Σ.** Less brittle than mean-variance, because no μ
    is involved, but a badly conditioned covariance still misallocates. Use a
    shrunk estimator.
"""

from __future__ import annotations

import numpy as np

from src.analytics.optimization.constraints import Constraints
from src.analytics.optimization.minimum_variance import _unpack
from src.analytics.optimization.result import OptimizationResult


def risk_contributions(weights: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Per-asset contribution to portfolio volatility. Sums to σ_p by construction,
    which is the identity that makes "equal contribution" a meaningful target.
    """
    w = np.asarray(weights, dtype=float)
    port_var = float(w @ Sigma @ w)
    if port_var <= 0:
        return np.zeros_like(w)
    return w * (Sigma @ w) / np.sqrt(port_var)


def risk_parity(
    cov,
    constraints: Constraints | None = None,
    *,
    budgets: np.ndarray | None = None,
    tickers: list[str] | None = None,
) -> OptimizationResult:
    """
    Equal-risk-contribution weights under `constraints`.

    budgets : optional risk budget per asset (any positive vector, normalised
              internally). Defaults to equal. This is the generalisation that
              makes the function useful beyond the textbook case — "40% of the
              risk in equities" is a risk budget, not a weight.
    """
    constraints = constraints or Constraints()
    Sigma, tickers = _unpack(cov, tickers)
    n = Sigma.shape[0]
    constraints.validate(n)

    if not constraints.long_only:
        # Risk contributions are not well defined when weights can flip sign:
        # a negative w_i (Σw)_i is a negative "contribution", the targets stop
        # being reachable, and the optimiser converges to something that is not
        # risk parity in any useful sense.
        raise ValueError(
            "risk_parity requires long-only weights; risk contributions are "
            "not meaningful when weights may be negative."
        )

    if budgets is None:
        target = np.full(n, 1.0 / n)
    else:
        b = np.asarray(budgets, dtype=float)
        if len(b) != n:
            raise ValueError(f"budgets length {len(b)} != {n} assets")
        if (b <= 0).any():
            raise ValueError("risk budgets must be strictly positive")
        target = b / b.sum()

    from scipy.optimize import minimize

    def objective(w):
        port_var = float(w @ Sigma @ w)
        if port_var <= 0:
            return 1e6
        rc = w * (Sigma @ w) / port_var          # shares, sum to 1
        return float(((rc - target) ** 2).sum())

    # A floor keeps the search away from w_i = 0, where the risk contribution
    # is identically zero and its gradient vanishes — the optimiser would
    # otherwise be free to park an asset at the boundary and call it done.
    floor = max(constraints.min_weight, 1e-6)
    bounds = [(floor, b[1]) for b in constraints.bounds(n)]

    # Start from inverse volatility, not equal weight. This is the exact ERC
    # solution when every pairwise correlation is the same, so it is the
    # principled initialisation rather than a nudge — and it matters: on a
    # book holding BTC at 76% volatility alongside TLT at 9%, equal weight is
    # far enough away that SLSQP stalls at an objective of 3e-2 and reports
    # success anyway. From inverse volatility the same problem converges to
    # 3e-15 in twelve iterations.
    #
    # That "reports success anyway" is why the dispersion check below exists.
    # Trusting res.success alone would have returned a portfolio whose risk
    # contributions are visibly unequal, under the name risk parity.
    vols = np.sqrt(np.clip(np.diag(Sigma), 1e-18, None))
    x0 = (1.0 / vols) / np.sum(1.0 / vols) * constraints.budget
    x0 = np.clip(x0, floor, constraints.max_weight or np.inf)
    x0 = x0 * (constraints.budget / x0.sum())

    res = minimize(
        objective, x0=x0, method="SLSQP",
        bounds=bounds, constraints=(constraints.budget_constraint(),),
        options={"ftol": 1e-14, "maxiter": 1000},
    )
    w = np.clip(res.x, floor, None)
    w = w * (constraints.budget / w.sum())

    rc = risk_contributions(w, Sigma)
    dispersion = float(np.std(rc / rc.sum())) if rc.sum() else float("nan")

    return OptimizationResult(
        method="risk_parity (equal risk contribution)",
        tickers=tickers, weights=w,
        expected_volatility=float(np.sqrt(max(w @ Sigma @ w, 0.0))),
        success=bool(res.success) and dispersion < 1e-3,
        message=(str(res.message) if not res.success else
                 f"RC dispersion {dispersion:.2e}"),
        n_iter=int(res.nit),
        constraints=constraints,
    )
