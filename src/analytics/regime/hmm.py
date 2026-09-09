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

SMOOTHED vs FILTERED — read this before using a regime as a signal
------------------------------------------------------------------
The model exposes both, and choosing wrongly is a look-ahead bug that no
amount of careful date-truncation will catch.

  smoothed_probs[s] = P(state at s | ALL observations through T)
  filtered_probs[s] = P(state at s | observations through s only)

The Kim smoother runs backwards, so a smoothed probability at time s
incorporates everything that happened *after* s. For describing history
that is exactly right — asked when the crisis was, you should use all the
evidence — and it is what the dashboard and the regime-conditional
analytics use.

As a trading signal it is catastrophic: the model has already seen the
crash it is supposed to be warning about. Anything inside `backtest/` must
use `filtered_probs` / `filtered_states`. The two agree most of the time
and diverge precisely at turning points, which is where a regime strategy
would act — so the disagreement is concentrated exactly where it does the
most damage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeModel:
    n_states: int
    states: np.ndarray                 # most-likely state per date (0..k-1), vol-sorted
    smoothed_probs: np.ndarray = field(repr=False)   # (T, k), uses future data
    filtered_probs: np.ndarray = field(repr=False)   # (T, k), causal
    means: np.ndarray = field(repr=False)            # per-state daily mean (sorted)
    volatilities: np.ndarray = field(repr=False)     # per-state daily vol (sorted)
    transition_matrix: np.ndarray = field(repr=False)
    expected_durations: np.ndarray = field(repr=False)
    dates: list = field(repr=False)
    labels: list[str] = field(default_factory=list)
    log_likelihood: float = 0.0
    # Annualisation factor for the series this was fitted on. Must come from the
    # data's calendar: a 24/7 crypto series has 365 observations a year, and
    # reporting its regime volatility at 252 understates it by a factor of
    # sqrt(365/252) ≈ 1.20.
    ppy: int = 252

    @property
    def filtered_states(self) -> np.ndarray:
        """
        Most-likely state per date using only information available then.

        This is the one a strategy may use. `states` is its smoothed twin
        and looks into the future by construction.
        """
        return self.filtered_probs.argmax(axis=1)

    @property
    def current_state(self) -> int:
        return int(self.states[-1])

    @property
    def current_label(self) -> str:
        return self.labels[self.current_state]

    def current_probabilities(self) -> dict[str, float]:
        return {self.labels[i]: float(self.smoothed_probs[-1, i])
                for i in range(self.n_states)}

    def probabilities_at(self, i: int, filtered: bool = True) -> dict[str, float]:
        """
        State probabilities at row `i`.

        Defaults to the causal estimate: the safe default is the one that
        cannot leak, so a caller has to ask explicitly for the version that
        sees the future.
        """
        probs = self.filtered_probs if filtered else self.smoothed_probs
        return {self.labels[k]: float(probs[i, k]) for k in range(self.n_states)}

    def disagreement(self) -> float:
        """
        Largest gap between the filtered and smoothed probability of any
        state on any date. A diagnostic: near zero means the smoother added
        little, and a large value means the two tell materially different
        stories at some point in the sample.
        """
        return float(np.abs(self.smoothed_probs - self.filtered_probs).max())

    def summary(self) -> str:
        lines = [f"-- Regime Model ({self.n_states} states, "
                 f"logL={self.log_likelihood:.0f}, {self.ppy}/yr) --",
                 f"  {'regime':<12} {'ann.ret':>9} {'ann.vol':>9} "
                 f"{'avg.dur':>9} {'freq':>7}"]
        freqs = np.bincount(self.states, minlength=self.n_states) / len(self.states)
        for i in range(self.n_states):
            # Annualised return shown geometrically and clipped: raw fitted
            # intercepts for rare regimes can be noisy, and mean×ppy can blow up
            # visually — geometric annualisation is the honest, bounded figure.
            ann_ret = (1.0 + self.means[i]) ** self.ppy - 1.0
            lines.append(
                f"  {self.labels[i]:<12} {ann_ret:>9.1%} "
                f"{self.volatilities[i]*np.sqrt(self.ppy):>9.1%} "
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
    ppy: int = 252,
) -> RegimeModel:
    """
    Fit a Gaussian Markov-switching model (switching mean and variance).

    returns : 1-D array of the series to detect regimes on (e.g. Mkt-RF daily).
    search_reps : random restarts for EM — guards against local optima.
    ppy : observations per year, used only to annualise the reported figures.
          Pass `rs.periods_per_year` rather than the default when the series is
          not exchange-traded — a 24/7 series has 365.

    The estimation itself is calendar-agnostic: it sees a sequence of returns
    and knows nothing about weekends. Only the reporting needs `ppy`.
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
    # The causal counterpart, needed by anything that trades on the state.
    filtered = np.asarray(res.filtered_marginal_probabilities)  # (T, k)
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
    filtered = filtered[:, order]
    durations = durations[order]
    # reorder transition matrix rows and columns
    trans = trans[np.ix_(order, order)]

    states = smoothed.argmax(axis=1)

    return RegimeModel(
        n_states=n_states,
        states=states,
        smoothed_probs=smoothed,
        filtered_probs=filtered,
        means=means,
        volatilities=vols,
        transition_matrix=trans,
        expected_durations=durations,
        dates=list(dates),
        labels=_label_by_vol(n_states),
        log_likelihood=float(res.llf),
        ppy=ppy,
    )
