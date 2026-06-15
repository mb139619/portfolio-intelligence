"""
Tail risk — beyond historical VaR/CVaR.

Historical VaR/CVaR (in performance.py) read the empirical quantile: they only
"see" the tail that already happened and say nothing about losses larger than
the worst observation. Two standard extensions fill that gap:

  1. Cornish-Fisher (modified VaR) — a parametric correction to the Gaussian
     quantile using the sample skewness and excess kurtosis. Cheap, closed-form,
     and a big improvement over Gaussian VaR for the mildly non-normal returns
     typical of most portfolios.

  2. Extreme Value Theory, peaks-over-threshold (POT) — fit a Generalised Pareto
     Distribution (GPD) to the exceedances beyond a high threshold. This is the
     principled way to extrapolate INTO the tail, including loss levels never
     observed, and to read the tail's shape (the GPD shape parameter ξ).

All functions work on the LOSS convention where a positive number is a loss, to
match the VaR/CVaR sign convention used elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# ──────────────────────────────────────────────────────────────
# Cornish-Fisher (modified VaR / CVaR)
# ──────────────────────────────────────────────────────────────

def cornish_fisher_quantile(z: float, skew: float, excess_kurt: float) -> float:
    """
    Cornish-Fisher expansion of a standard-normal quantile z, correcting for
    skewness S and excess kurtosis K:

        z_cf = z + (z²−1)/6·S + (z³−3z)/24·K − (2z³−5z)/36·S²
    """
    return (
        z
        + (z**2 - 1) / 6 * skew
        + (z**3 - 3 * z) / 24 * excess_kurt
        - (2 * z**3 - 5 * z) / 36 * skew**2
    )


def cornish_fisher_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Modified (Cornish-Fisher) VaR — positive number = loss. Reduces to Gaussian
    VaR when skew and excess kurtosis are zero.
    """
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    s = float(stats.skew(returns))
    k = float(stats.kurtosis(returns))   # excess kurtosis (Fisher)

    z = stats.norm.ppf(1 - confidence)   # negative (left tail)
    z_cf = cornish_fisher_quantile(z, s, k)
    return float(-(mu + sigma * z_cf))


def gaussian_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """Plain Gaussian (parametric) VaR — for comparison with the CF correction."""
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    z = stats.norm.ppf(1 - confidence)
    return float(-(mu + sigma * z))


# ──────────────────────────────────────────────────────────────
# Extreme Value Theory — peaks over threshold (GPD)
# ──────────────────────────────────────────────────────────────

@dataclass
class EVTResult:
    threshold: float            # loss threshold u (positive)
    shape: float                # GPD shape parameter ξ (xi): tail heaviness
    scale: float                # GPD scale parameter β (beta)
    n_exceedances: int
    n_total: int
    confidence: float
    var: float                  # EVT VaR at `confidence` (positive = loss)
    cvar: float                 # EVT expected shortfall (positive = loss)

    @property
    def exceedance_rate(self) -> float:
        return self.n_exceedances / self.n_total

    def summary(self) -> str:
        tail = ("heavy-tailed (ξ>0)" if self.shape > 0.02
                else "thin-tailed (ξ<0)" if self.shape < -0.02
                else "exponential tail (ξ≈0)")
        return "\n".join([
            f"-- EVT / Peaks-Over-Threshold (GPD) --",
            f"  threshold (u)     {self.threshold:>8.2%}",
            f"  exceedances       {self.n_exceedances} / {self.n_total} "
            f"({self.exceedance_rate:.1%})",
            f"  shape (xi)        {self.shape:>8.3f}   {tail}",
            f"  scale (beta)      {self.scale:>8.4f}",
            f"  VaR  ({self.confidence:.0%})     {self.var:>8.2%}",
            f"  CVaR ({self.confidence:.0%})     {self.cvar:>8.2%}",
        ])


def fit_evt_pot(
    returns: np.ndarray,
    confidence: float = 0.99,
    threshold_quantile: float = 0.95,
) -> EVTResult:
    """
    Peaks-over-threshold EVT. Fit a GPD to losses exceeding a high threshold,
    then derive VaR and CVaR at `confidence`.

    threshold_quantile : the quantile of losses used as the threshold u. The
        exceedances above u are modelled by a GPD. 0.95 is a common choice;
        EVT is sensitive to it (the bias/variance trade-off of threshold choice).

    POT VaR (Pickands–Balkema–de Haan):
        VaR_p = u + (β/ξ) · [ ( (n/N_u)·(1−p) )^(−ξ) − 1 ]
    and for ξ < 1 the expected shortfall is:
        CVaR_p = (VaR_p + β − ξ·u) / (1 − ξ)
    """
    losses = -np.asarray(returns)                 # losses positive
    n = len(losses)
    u = float(np.quantile(losses, threshold_quantile))
    exceed = losses[losses > u] - u
    n_u = len(exceed)
    if n_u < 10:
        raise ValueError(
            f"Too few exceedances ({n_u}) above the threshold to fit a GPD; "
            f"lower threshold_quantile or use more data."
        )

    # Fit GPD to exceedances (location fixed at 0)
    xi, _, beta = stats.genpareto.fit(exceed, floc=0.0)

    p = confidence
    ratio = (n / n_u) * (1 - p)
    if abs(xi) < 1e-8:
        var = u + beta * (-np.log(ratio))         # ξ → 0 limit (exponential)
    else:
        var = u + (beta / xi) * (ratio ** (-xi) - 1)

    # Expected shortfall (requires ξ < 1 for a finite mean)
    if xi < 1:
        cvar = (var + beta - xi * u) / (1 - xi)
    else:
        cvar = np.inf

    return EVTResult(
        threshold=u, shape=float(xi), scale=float(beta),
        n_exceedances=n_u, n_total=n, confidence=confidence,
        var=float(var), cvar=float(cvar),
    )


# ──────────────────────────────────────────────────────────────
# Convenience: compare all tail estimates side by side
# ──────────────────────────────────────────────────────────────

def tail_risk_comparison(returns: np.ndarray, confidence: float = 0.99) -> dict:
    """
    Compare Gaussian, Cornish-Fisher, historical and EVT VaR at one confidence
    level — the four side by side make the impact of tail-modelling explicit.
    """
    from src.analytics.performance import historical_var, historical_cvar

    out = {
        "gaussian_var": gaussian_var(returns, confidence),
        "cornish_fisher_var": cornish_fisher_var(returns, confidence),
        "historical_var": historical_var(returns, confidence),
        "historical_cvar": historical_cvar(returns, confidence),
    }
    try:
        evt = fit_evt_pot(returns, confidence=confidence)
        out["evt_var"] = evt.var
        out["evt_cvar"] = evt.cvar
        out["evt_shape"] = evt.shape
    except ValueError:
        out["evt_var"] = None
        out["evt_cvar"] = None
        out["evt_shape"] = None
    return out
