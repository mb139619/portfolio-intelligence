"""
Volatility-state baseline.

A deliberately simple regime detector to benchmark the HMM against: classify
each day by its trailing realised volatility into quantile buckets. No latent
states, no EM — just "are we in a high-vol or low-vol environment right now?"

It is fast, transparent and surprisingly hard to beat, which is exactly why it
makes a good baseline: if the HMM doesn't add interpretive value over this, the
extra machinery isn't earning its keep.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VolatilityStates:
    n_states: int
    states: np.ndarray              # state per date (0 = lowest vol)
    realized_vol: np.ndarray = field(repr=False)   # annualised rolling vol
    thresholds: np.ndarray = field(repr=False)     # quantile cut points
    dates: list = field(repr=False)
    labels: list[str] = field(default_factory=list)
    window: int = 21

    @property
    def current_state(self) -> int:
        return int(self.states[-1])

    @property
    def current_label(self) -> str:
        return self.labels[self.current_state]

    def summary(self) -> str:
        lines = [f"-- Volatility States (baseline, {self.n_states} buckets, "
                 f"{self.window}d window) --"]
        freqs = np.bincount(self.states, minlength=self.n_states) / len(self.states)
        for i in range(self.n_states):
            mask = self.states == i
            avg_vol = np.nanmean(self.realized_vol[mask]) if mask.any() else np.nan
            lines.append(f"  {self.labels[i]:<10} avg ann.vol {avg_vol:>7.1%}  "
                         f"freq {freqs[i]:>6.1%}")
        lines.append(f"  Current: {self.current_label}")
        return "\n".join(lines)


def _labels(n: int) -> list[str]:
    if n == 2:
        return ["low_vol", "high_vol"]
    if n == 3:
        return ["low_vol", "mid_vol", "high_vol"]
    return [f"vol_q{i}" for i in range(n)]


def volatility_states(
    returns: np.ndarray,
    dates: list,
    window: int = 21,
    n_states: int = 2,
    ppy: int = 252,
) -> VolatilityStates:
    """
    Classify each day by trailing realised volatility into `n_states` quantile
    buckets. The first `window-1` days (no full window yet) inherit the first
    available classification.
    """
    r = np.asarray(returns, dtype=float)
    T = len(r)

    # Trailing realised vol (annualised), NaN until the window fills
    vol = np.full(T, np.nan)
    for i in range(window, T + 1):
        vol[i - 1] = np.std(r[i - window:i], ddof=1) * np.sqrt(ppy)
    # back-fill the warm-up period with the first valid value
    first_valid = np.argmax(~np.isnan(vol))
    vol[:first_valid] = vol[first_valid]

    # Quantile thresholds on the valid vol distribution
    qs = np.linspace(0, 1, n_states + 1)[1:-1]
    thresholds = np.quantile(vol[~np.isnan(vol)], qs)
    states = np.digitize(vol, thresholds)

    return VolatilityStates(
        n_states=n_states,
        states=states,
        realized_vol=vol,
        thresholds=thresholds,
        dates=list(dates),
        labels=_labels(n_states),
        window=window,
    )
