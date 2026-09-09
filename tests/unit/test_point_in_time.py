"""
Point-in-time producers.

`Context` has been able to *consume* a publication lag and a causal regime
estimate since the contracts landed. Until these producers existed both were
structurally always absent — a guarantee that looked enforced and was not.
These tests exist to keep them wired.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.regime.hmm import fit_regimes
from src.ingestion import french, prices, rates

# ──────────────────────────────────────────────────────────────
# Publication lag
# ──────────────────────────────────────────────────────────────

class TestPublicationLag:
    def test_every_price_source_declares_one(self):
        for ticker, src in prices.PRICE_REGISTRY.items():
            assert src.publication_lag_days >= 0, ticker

    def test_unregistered_tickers_get_the_yahoo_default(self):
        """The open-universe fallback must still answer, not raise."""
        assert prices.publication_lag_days("SOME_ETF") == 0

    def test_daily_closes_are_knowable_same_day(self):
        """
        Zero is correct here and is not an oversight: a close is published at
        the close. Whether you could have TRADED at it is an execution
        question, not a data-availability one.
        """
        assert prices.publication_lag_days("SPY") == 0
        assert prices.publication_lag_days("BTC-USD") == 0

    def test_rates_lag_by_a_business_day(self):
        assert rates.publication_lag_days("USD_FEDFUNDS") == 1
        assert rates.publication_lag_days("EUR_DFR") == 1

    def test_unknown_rate_series_raises(self):
        with pytest.raises(ValueError, match="Unknown rate series"):
            rates.publication_lag_days("NOT_A_SERIES")

    def test_french_factors_lag_by_weeks(self):
        """
        The single most important lag in the project. Measured against this
        repo's own store the factor data has run 35 days behind the prices, so
        anything near zero here would silently readmit look-ahead into every
        factor-based backtest.
        """
        assert french.publication_lag_days("FF5") >= 30
        assert french.publication_lag_days("MOM") >= 30

    def test_every_factor_dataset_declares_one(self):
        for name, ds in french.FACTOR_REGISTRY.items():
            assert ds.publication_lag_days >= 30, name

    def test_unknown_factor_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown French dataset"):
            french.publication_lag_days("FF99")

    def test_factor_lag_dominates_price_lag(self):
        """
        A backtest must wait for its slowest input. If this ever inverts, the
        engine's "max lag across sources" rule would quietly stop binding.
        """
        assert french.publication_lag_days("FF5") > prices.publication_lag_days("SPY")

    def test_dataset_files_view_matches_the_registry(self):
        assert french.DATASET_FILES == {
            k: v.filename for k, v in french.FACTOR_REGISTRY.items()
        }


# ──────────────────────────────────────────────────────────────
# Filtered vs smoothed regime probabilities
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def switching_model():
    """A series with two genuine volatility regimes and a clear switch."""
    rng = np.random.default_rng(7)
    y = np.concatenate([
        rng.normal(0.0005, 0.005, 500),
        rng.normal(-0.001, 0.022, 250),
        rng.normal(0.0005, 0.005, 500),
    ])
    dates = list(range(len(y)))
    return fit_regimes(y, dates, n_states=2, search_reps=3)


class TestFilteredProbabilities:
    def test_both_estimates_are_exposed(self, switching_model):
        m = switching_model
        assert m.filtered_probs.shape == m.smoothed_probs.shape

    def test_filtered_probabilities_are_a_distribution(self, switching_model):
        np.testing.assert_allclose(
            switching_model.filtered_probs.sum(axis=1), 1.0, atol=1e-6
        )

    def test_they_agree_exactly_on_the_last_observation(self, switching_model):
        """
        The defining property, and the one that proves the right arrays are
        being read. At the final date there is no future left to smooth over,
        so P(state_T | data through T) is the same estimate under either name.
        """
        m = switching_model
        np.testing.assert_allclose(
            m.filtered_probs[-1], m.smoothed_probs[-1], atol=1e-8
        )

    def test_they_disagree_earlier_in_the_sample(self, switching_model):
        """
        If these never diverged, the smoother would be adding nothing and the
        distinction would not matter. It does diverge — around the switches.
        """
        assert switching_model.disagreement() > 0.05

    def test_filtered_states_come_from_filtered_probabilities(self, switching_model):
        m = switching_model
        np.testing.assert_array_equal(
            m.filtered_states, m.filtered_probs.argmax(axis=1)
        )

    def test_smoothed_states_are_the_default_for_description(self, switching_model):
        """`states` stays smoothed: the dashboard and conditional analytics want it."""
        m = switching_model
        np.testing.assert_array_equal(m.states, m.smoothed_probs.argmax(axis=1))

    def test_the_hard_classification_actually_differs(self, switching_model):
        """
        Not a rounding difference. On the real market series this reclassifies
        about 8.6% of days while leaving the aggregate stress frequency within
        half a point -- which is exactly why the bug survives casual review.
        """
        m = switching_model
        assert (m.states != m.filtered_states).sum() > 0

    def test_probabilities_at_defaults_to_the_causal_estimate(self, switching_model):
        """The safe default is the one that cannot leak."""
        m = switching_model
        i = 400
        default = m.probabilities_at(i)
        causal = m.probabilities_at(i, filtered=True)
        assert default == causal

    def test_the_leaky_estimate_must_be_asked_for(self, switching_model):
        m = switching_model
        i = int(np.argmax(np.abs(m.smoothed_probs - m.filtered_probs).max(axis=1)))
        assert m.probabilities_at(i, filtered=False) != m.probabilities_at(i)

    def test_labels_line_up_with_both(self, switching_model):
        m = switching_model
        assert set(m.probabilities_at(0)) == set(m.labels)
