"""
Parametric stress — hypothetical shocks.

factor_shock : apply an instantaneous move to factor returns and propagate
               through the portfolio's factor betas. PnL = sum_i beta_i * shock_i.
               Exact and fully additive.

macro_shock  : for variables not in the FF factor set (rates, oil, USD, VIX),
               first estimate the portfolio's sensitivity to daily moves in
               those variables (a linear/delta regression), then apply a shock
               in native units. First-order approximation, clearly labelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.analytics.stress.historical import StressResult


# ─────────────────────────────────────────────────────────────────────────
# Factor shocks (FF space)
# ─────────────────────────────────────────────────────────────────────────

def run_factor_shock(model, shock: dict, name: str = "factor_shock") -> StressResult:
    """
    Apply a hypothetical move to factors and propagate via betas.
    shock: {factor_name: move}. Factors absent from the shock move 0.
    PnL = sum_i beta_i * shock_i (exact, additive).
    """
    unknown = [f for f in shock if f not in model.factor_names]
    if unknown:
        raise ValueError(f"Shock references unknown factors: {unknown}")

    contributions = {
        f: float(model.betas[f] * shock.get(f, 0.0)) for f in model.factor_names
    }
    total = float(sum(contributions.values()))

    return StressResult(
        scenario=name, method="factor_shock",
        total_pnl=total, contributions=contributions, window="instantaneous",
        note="PnL = beta · shock (first-order, additive).",
    )


# ─────────────────────────────────────────────────────────────────────────
# Macro sensitivities + macro shocks
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MacroSensitivities:
    """Linear sensitivities of portfolio returns to daily macro moves."""
    variables: list[str]
    betas: dict[str, float]      # portfolio return per 1 unit of macro move
    r_squared: float
    n_obs: int

    def summary(self) -> str:
        lines = [f"-- Macro Sensitivities (R²={self.r_squared:.3f}, n={self.n_obs}) --"]
        for v in self.variables:
            lines.append(f"  d(portfolio)/d({v}) = {self.betas[v]:+.4f}")
        return "\n".join(lines)


def estimate_macro_sensitivities(
    port_returns: np.ndarray,
    macro_changes: pl.DataFrame,
    variables: list[str],
) -> MacroSensitivities:
    """
    Regress portfolio daily returns on daily macro moves to get sensitivities.

    macro_changes : wide DataFrame, one column per variable, ALREADY expressed
                    as daily changes/returns aligned to port_returns:
                      - rate variables: daily yield change (decimal)
                      - oil/USD: daily return (decimal)
                      - VIX: daily level change
    The caller is responsible for alignment (same length, same dates).
    """
    import statsmodels.api as sm

    X = macro_changes.select(variables).to_numpy()
    if X.shape[0] != len(port_returns):
        raise ValueError("port_returns and macro_changes must have equal length")

    X_const = sm.add_constant(X)
    res = sm.OLS(port_returns, X_const).fit()

    betas = {v: float(res.params[i + 1]) for i, v in enumerate(variables)}
    return MacroSensitivities(
        variables=variables, betas=betas,
        r_squared=float(res.rsquared), n_obs=len(port_returns),
    )


def run_macro_shock(
    sens: MacroSensitivities,
    shock: dict,
    name: str = "macro_shock",
) -> StressResult:
    """
    Apply a macro shock in native units via estimated sensitivities.
    shock: {variable: move}. PnL = sum_v beta_v * move_v (first-order).
    """
    unknown = [v for v in shock if v not in sens.betas]
    if unknown:
        raise ValueError(f"Shock references variables without sensitivities: {unknown}")

    contributions = {
        v: float(sens.betas[v] * shock.get(v, 0.0)) for v in sens.variables
    }
    total = float(sum(contributions.values()))

    return StressResult(
        scenario=name, method="macro_shock",
        total_pnl=total, contributions=contributions, window="instantaneous",
        note="first-order: PnL = sensitivity · shock; ignores convexity and "
             "second-order effects.",
    )
