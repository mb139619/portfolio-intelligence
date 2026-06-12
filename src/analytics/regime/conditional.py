"""
Regime-conditional analytics.

The payoff of regime detection: once each date carries a regime label (estimated
on the broad market), recompute the PORTFOLIO's behaviour within each regime.
This makes non-stationarity concrete and measurable.

The robust, near-universal effect is VOLATILITY: it rises sharply (often roughly
doubling) in the stress regime, and the return/vol ratio typically flips sign.

Two further effects are commonly cited but are CONDITIONAL, not guaranteed:
  - the portfolio's market BETA *often* rises in stress, but beta is
    cov(p,m)/var(m), and in stress both terms inflate and can cancel — so beta
    may stay flat or even fall while absolute risk rises. Read it, don't assume.
  - average cross-asset CORRELATION *often* rises toward 1 in stress
    (diversification breakdown), but this is most visible on broad cross-asset
    universes; a book already dominated by one factor starts highly correlated
    and has little room to climb.

These are the empirical counterpart to the stress-testing caveat that constant
full-sample betas understate crisis losses — when the effect is present.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries


def regime_conditional_stats(
    portfolio_returns: np.ndarray,
    states: np.ndarray,
    labels: list[str],
    ppy: int = 252,
) -> pl.DataFrame:
    """Per-regime annualised return, volatility, Sharpe-like ratio and frequency."""
    rows = []
    for i, lab in enumerate(labels):
        mask = states == i
        r = portfolio_returns[mask]
        if len(r) == 0:
            continue
        ann_ret = float(np.mean(r) * ppy)
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(ppy)) if len(r) > 1 else float("nan")
        rows.append({
            "regime": lab,
            "frequency": float(mask.mean()),
            "ann_return": ann_ret,
            "ann_volatility": ann_vol,
            "return_vol_ratio": ann_ret / ann_vol if ann_vol and ann_vol > 0 else float("nan"),
            "n_days": int(mask.sum()),
        })
    return pl.DataFrame(rows)


def regime_conditional_beta(
    portfolio_excess: np.ndarray,
    market_excess: np.ndarray,
    states: np.ndarray,
    labels: list[str],
) -> pl.DataFrame:
    """
    Market beta of the portfolio within each regime, via cov/var.
    Demonstrates the "beta rises in stress" effect directly.
    Inputs should be excess returns aligned to `states`.
    """
    rows = []
    for i, lab in enumerate(labels):
        mask = states == i
        p, m = portfolio_excess[mask], market_excess[mask]
        if len(p) < 2:
            continue
        var_m = np.var(m, ddof=1)
        beta = float(np.cov(p, m, ddof=1)[0, 1] / var_m) if var_m > 0 else float("nan")
        rows.append({"regime": lab, "beta": beta, "n_days": int(mask.sum())})
    return pl.DataFrame(rows)


def regime_conditional_avg_correlation(
    rs: ReturnSeries,
    states: np.ndarray,
    labels: list[str],
) -> pl.DataFrame:
    """
    Average pairwise correlation of the universe within each regime.
    The stress regime typically shows markedly higher average correlation.
    `states` must be aligned to the rows of `rs`.
    """
    R = rs.to_numpy()
    n_assets = R.shape[1]
    iu = np.triu_indices(n_assets, k=1)
    rows = []
    for i, lab in enumerate(labels):
        mask = states == i
        if mask.sum() < 3:
            continue
        corr = np.corrcoef(R[mask], rowvar=False)
        rows.append({
            "regime": lab,
            "avg_correlation": float(np.mean(corr[iu])),
            "n_days": int(mask.sum()),
        })
    return pl.DataFrame(rows)


def align_states_to_returns(
    state_dates: list,
    states: np.ndarray,
    target_dates: pl.Series,
) -> np.ndarray:
    """
    Map a regime-state series (estimated on the market, on its own dates) onto a
    target return calendar by date. Returns an int array aligned to target_dates;
    dates without a state are marked -1 (caller should mask them out).
    """
    state_map = {d: int(s) for d, s in zip(state_dates, states)}
    return np.array([state_map.get(d, -1) for d in target_dates.to_list()])
