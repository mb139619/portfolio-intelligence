"""
Regime detection — Gaussian hidden Markov / Markov-switching model.

A market environment alternates between hidden states (calm vs stress, bull vs
bear) with different mean and variance. We fit a Markov-switching model with
switching mean AND variance — i.e. a Gaussian HMM on the observed return series:

    r_t = mu_{S_t} + e_t,   e_t ~ N(0, sigma^2_{S_t}),   S_t a hidden Markov chain

Estimated by EM (Hamilton filter + Kim smoother). We fit on a BROAD MARKET
series (e.g. the Fama-French Mkt-RF factor), not on the portfolio: a regime is a
property of the market environment, which the portfolio then inherits. Once
states are known, portfolio analytics are conditioned on them (see conditional.py).

Implementation note: states are canonicalised by ascending volatility, so
state 0 is always the lowest-vol ("calm") regime and the last state the
highest-vol ("stress") regime — stable, interpretable labels across refits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeModel:
    n_states: int
    states: np.ndarray                 # most-likely state per date (0..k-1), vol-sorted
    smoothed_probs: np.ndarray = field(repr=False)   # (T, k)
    means: np.ndarray = field(repr=False)            # per-state daily mean (sorted)
    volatilities: np.ndarray = field(repr=False)     # per-state daily vol (sorted)
    transition_matrix: np.ndarray = field(repr=False)
    expected_durations: np.ndarray = field(repr=False)
    dates: list = field(repr=False)
    labels: list[str] = field(default_factory=list)
    log_likelihood: float = 0.0

    @property
    def current_state(self) -> int:
        return int(self.states[-1])

    @property
    def current_label(self) -> str:
        return self.labels[self.current_state]

    def current_probabilities(self) -> dict[str, float]:
        return {self.labels[i]: float(self.smoothed_probs[-1, i])
                for i in range(self.n_states)}

    def summary(self) -> str:
        lines = [f"-- Regime Model ({self.n_states} states, logL={self.log_likelihood:.0f}) --",
                 f"  {'regime':<12} {'ann.ret':>9} {'ann.vol':>9} {'avg.dur':>9} {'freq':>7}"]
        freqs = np.bincount(self.states, minlength=self.n_states) / len(self.states)
        for i in range(self.n_states):
            # Annualised return shown geometrically and clipped: raw fitted
            # intercepts for rare regimes can be noisy, and mean×252 can blow up
            # visually — geometric annualisation is the honest, bounded figure.
            ann_ret = (1.0 + self.means[i]) ** 252 - 1.0
            lines.append(
                f"  {self.labels[i]:<12} {ann_ret:>9.1%} "
                f"{self.volatilities[i]*np.sqrt(252):>9.1%} "
                f"{self.expected_durations[i]:>8.0f}d {freqs[i]:>7.1%}"
            )
        probs = self.current_probabilities()
        lines.append(f"  Current: {self.current_label} "
                     f"(p={probs[self.current_label]:.0%})")
        return "\n".join(lines)


def _label_by_vol(n_states: int) -> list[str]:
    if n_states == 2:
        return ["calm", "stress"]
    if n_states == 3:
        return ["calm", "normal", "stress"]
    return [f"vol_{i}" for i in range(n_states)]


def fit_regimes(
    returns: np.ndarray,
    dates: list,
    n_states: int = 2,
    search_reps: int = 20,
    seed: int = 42,
) -> RegimeModel:
    """
    Fit a Gaussian Markov-switching model (switching mean and variance).

    returns : 1-D array of the series to detect regimes on (e.g. Mkt-RF daily).
    search_reps : random restarts for EM — guards against local optima.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    y = np.asarray(returns, dtype=float)
    np.random.seed(seed)

    mod = MarkovRegression(y, k_regimes=n_states, trend="c", switching_variance=True)
    res = mod.fit(search_reps=search_reps, disp=False)

    # statsmodels orders res.params as: [transition params ...,
    #   const[0..k-1] (means), sigma2[0..k-1] (variances)].
    # The means are the k params immediately before the k variances; slicing
    # from the END is robust to the number of transition params (= k*(k-1)).
    raw_means = np.asarray(res.params[-2 * n_states:-n_states])
    raw_vars = np.asarray(res.params[-n_states:])
    raw_vols = np.sqrt(np.abs(raw_vars))

    smoothed = np.asarray(res.smoothed_marginal_probabilities)  # (T, k)
    trans = np.asarray(res.regime_transition)
    if trans.ndim == 3:
        trans = trans[:, :, 0]
    # statsmodels returns a COLUMN-stochastic matrix (columns sum to 1).
    # Transpose to the standard ROW-stochastic convention where
    # transition_matrix[i, j] = P(state j at t+1 | state i at t).
    trans = trans.T
    durations = np.asarray(res.expected_durations)

    # --- canonicalise by ascending volatility ---
    order = np.argsort(raw_vols)
    means = raw_means[order]
    vols = raw_vols[order]
    smoothed = smoothed[:, order]
    durations = durations[order]
    # reorder transition matrix rows and columns
    trans = trans[np.ix_(order, order)]

    states = smoothed.argmax(axis=1)

    return RegimeModel(
        n_states=n_states,
        states=states,
        smoothed_probs=smoothed,
        means=means,
        volatilities=vols,
        transition_matrix=trans,
        expected_durations=durations,
        dates=list(dates),
        labels=_label_by_vol(n_states),
        log_likelihood=float(res.llf),
    )
