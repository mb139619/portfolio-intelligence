"""
Hierarchical Risk Parity (López de Prado, 2016).

Mean-variance inverts Σ, and inversion is where estimation error becomes
leverage: small perturbations in a nearly-singular covariance produce large,
offsetting positions. Minimum variance inherits the problem in milder form.
HRP sidesteps it entirely — **it never inverts anything**. That is the whole
idea, and the reason it tends to hold up out of sample where mean-variance
does not.

Three steps:

1. **Tree clustering** — group assets by correlation distance. Reuses
   `cluster_from_correlation`, deliberately: the risk engine already builds
   this dendrogram for the correlation page, and a second implementation would
   eventually disagree with the first about the same tree.
2. **Quasi-diagonalisation** — reorder so correlated assets sit adjacent,
   which is exactly `ClusteringResult.quasi_diagonal_order`.
3. **Recursive bisection** — split the ordered list in half repeatedly,
   allocating between each pair of halves inversely to their cluster variance.
   Only diagonal information is used at each step, so no matrix is inverted.

Where it stops being valid
--------------------------
* **The tree is an estimate too.** HRP is robust to the *inversion* of a noisy
  Σ, not to Σ being wrong. A correlation matrix estimated on too short a window
  produces an unstable dendrogram, and the allocation moves with it.
* **The bisection is arbitrary at the margin.** Splitting the ordered list down
  the middle is a convention, not an optimum; a different linkage method gives
  a different order and therefore different weights. `single` linkage is the
  classic choice and matches the MST topology the risk engine already computes.
* **It optimises nothing.** There is no objective function, so HRP cannot be
  said to be optimal under any criterion. It is a heuristic that happens to
  degrade gracefully — which is a different and more honest claim than the one
  mean-variance makes.
* **Constraints are applied afterwards.** Caps are enforced by projection (see
  `_apply_caps`), not inside the recursion, so a heavily capped HRP portfolio
  is no longer strictly HRP.
"""

from __future__ import annotations

import numpy as np

from src.analytics.correlation.matrices import correlation_from_covariance
from src.analytics.optimization.constraints import Constraints
from src.analytics.optimization.minimum_variance import _unpack
from src.analytics.optimization.result import OptimizationResult


def _inverse_variance_weights(sub: np.ndarray) -> np.ndarray:
    """Naive risk parity inside a cluster: weight inversely to variance."""
    ivp = 1.0 / np.diag(sub)
    total = ivp.sum()
    return ivp / total if total > 0 else np.full(len(ivp), 1.0 / len(ivp))


def _cluster_variance(Sigma: np.ndarray, idx: list[int]) -> float:
    """Variance of a cluster held at its own inverse-variance weights."""
    sub = Sigma[np.ix_(idx, idx)]
    w = _inverse_variance_weights(sub)
    return float(w @ sub @ w)


def recursive_bisection(Sigma: np.ndarray, order: list[int]) -> np.ndarray:
    """
    Split the quasi-diagonal ordering in half repeatedly, splitting capital
    between each pair inversely to cluster variance.

    Nothing here inverts a matrix; every step reads a variance off a small
    sub-block. That is what makes HRP insensitive to the conditioning of Σ.
    """
    w = np.ones(Sigma.shape[0])
    clusters = [list(order)]

    while clusters:
        split = []
        for c in clusters:
            if len(c) > 1:
                mid = len(c) // 2
                split.append(c[:mid])
                split.append(c[mid:])
        clusters = split

        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left = _cluster_variance(Sigma, left)
            v_right = _cluster_variance(Sigma, right)
            total = v_left + v_right
            # The riskier half gets the smaller share.
            alpha = 1.0 - v_left / total if total > 0 else 0.5
            for j in left:
                w[j] *= alpha
            for j in right:
                w[j] *= 1.0 - alpha
    return w


def _apply_caps(w: np.ndarray, constraints: Constraints) -> np.ndarray:
    """
    Project onto the box by capping and redistributing, iteratively.

    HRP has no natural way to express a cap — the recursion allocates by
    variance, not by mandate — so this is applied afterwards and is honestly a
    projection rather than part of the method. Redistribution goes to the
    uncapped names in proportion to their existing weights, which preserves the
    hierarchy's relative ordering among them.
    """
    hi = constraints.max_weight
    lo = constraints.min_weight
    w = np.clip(w, lo, None)
    w = w * (constraints.budget / w.sum())
    if hi is None:
        return w

    for _ in range(100):
        over = w > hi + 1e-12
        if not over.any():
            break
        excess = (w[over] - hi).sum()
        w[over] = hi
        free = ~over
        room = w[free].sum()
        if room <= 0:
            w[free] = excess / max(free.sum(), 1)
            break
        w[free] += excess * w[free] / room
    return w


def hrp(
    cov,
    constraints: Constraints | None = None,
    *,
    linkage_method: str = "single",
    tickers: list[str] | None = None,
) -> OptimizationResult:
    """
    Hierarchical Risk Parity weights under `constraints`.

    linkage_method : "single" is the classic choice and matches the MST
                     topology the correlation module already builds; "ward"
                     gives more balanced clusters and a different allocation.
                     That the answer depends on this is a property of the
                     method, not a bug.
    """
    from src.analytics.correlation.clustering import cluster_from_correlation

    constraints = constraints or Constraints()
    Sigma, tickers = _unpack(cov, tickers)
    n = Sigma.shape[0]
    constraints.validate(n)

    if not constraints.long_only:
        raise ValueError(
            "hrp produces long-only weights by construction; the recursion "
            "splits capital between clusters and cannot express a short."
        )

    if n == 1:
        w = np.array([constraints.budget])
        return _build(Sigma, tickers, w, constraints, "n=1")

    corr = correlation_from_covariance(Sigma)
    clustering = cluster_from_correlation(
        corr, tickers, n_clusters=min(3, n), linkage_method=linkage_method
    )
    w = recursive_bisection(Sigma, clustering.quasi_diagonal_order)
    w = _apply_caps(w, constraints)

    return _build(Sigma, tickers, w, constraints,
                  f"linkage={linkage_method}, order="
                  f"{'-'.join(clustering.ordered_tickers())}")


def _build(Sigma, tickers, w, constraints, message) -> OptimizationResult:
    return OptimizationResult(
        method="hrp (hierarchical risk parity)",
        tickers=tickers, weights=w,
        expected_volatility=float(np.sqrt(max(w @ Sigma @ w, 0.0))),
        success=True,          # a heuristic: it always produces an allocation
        message=message,
        constraints=constraints,
    )
