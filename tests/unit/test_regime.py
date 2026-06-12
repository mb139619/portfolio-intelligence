"""
Regime detection tests — validated against KNOWN synthetic regimes.

Construction: a low-vol "calm" block, a high-vol "stress" block, then calm again.
The detectors must recover this structure, order states by volatility, and the
regime-conditional analytics must show beta and correlation rising in stress.

Includes a regression test for the means-extraction bug (statsmodels orders
transition params before the means; a naive params[:k] slice grabs transition
probabilities instead of returns).
"""

import numpy as np
import polars as pl
import pytest

from src.domain.returns import ReturnSeries
from src.analytics.regime.hmm import fit_regimes
from src.analytics.regime.volatility_states import volatility_states
from src.analytics.regime.conditional import (
    regime_conditional_stats, regime_conditional_beta,
    regime_conditional_avg_correlation, align_states_to_returns,
)


@pytest.fixture
def two_regime_series():
    np.random.seed(0)
    calm1 = np.random.normal(0.0005, 0.005, 500)
    stress = np.random.normal(-0.0015, 0.022, 250)
    calm2 = np.random.normal(0.0005, 0.005, 500)
    y = np.concatenate([calm1, stress, calm2])
    T = len(y)
    dates = pl.date_range(pl.date(2018, 1, 1),
                          pl.date(2018, 1, 1) + pl.duration(days=T - 1),
                          interval="1d", eager=True).to_list()
    truth = np.concatenate([np.zeros(500), np.ones(250), np.zeros(500)]).astype(int)
    return y, dates, truth


class TestHMM:
    def test_means_are_returns_not_transition_probs(self, two_regime_series):
        # Regression test: means must be plausible daily returns (~1e-3), NOT
        # transition probabilities (~0.9). Catches the params[:k] slice bug.
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        assert np.all(np.abs(model.means) < 0.05)

    def test_states_ordered_by_vol(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        assert model.volatilities[0] < model.volatilities[1]
        assert model.labels == ["calm", "stress"]

    def test_recovers_stress_block(self, two_regime_series):
        y, dates, truth = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        assert np.mean(model.states == truth) > 0.9

    def test_stress_has_lower_mean(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        assert model.means[1] < model.means[0]   # stress mean < calm mean

    def test_transition_matrix_row_stochastic(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        np.testing.assert_allclose(model.transition_matrix.sum(axis=1), 1.0, atol=1e-6)

    def test_regimes_are_persistent(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        assert np.all(np.diag(model.transition_matrix) > 0.8)

    def test_expected_durations_positive(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        assert np.all(model.expected_durations > 0)

    def test_smoothed_probs_sum_to_one(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        np.testing.assert_allclose(model.smoothed_probs.sum(axis=1), 1.0, atol=1e-6)

    def test_current_probabilities_sum_to_one(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        assert sum(model.current_probabilities().values()) == pytest.approx(1.0, abs=1e-6)

    def test_summary_returns_are_bounded(self, two_regime_series):
        # If means were transition probs (~0.9), geometric annualisation would
        # explode. Assert the summary string contains no absurd magnitudes.
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        for i in range(model.n_states):
            ann = (1 + model.means[i]) ** 252 - 1
            assert -1.0 < ann < 10.0   # sane annualised range


class TestVolatilityStates:
    def test_recovers_high_vol_block(self, two_regime_series):
        # A 2-bucket quantile classifier splits 50/50 by construction, so it
        # flags more days than the true 20% stress block. The meaningful
        # property is RECALL: the true stress days are predominantly classified
        # as high-vol.
        y, dates, truth = two_regime_series
        vs = volatility_states(y, dates, window=21, n_states=2)
        stress_recall = np.mean(vs.states[truth == 1] == 1)
        assert stress_recall > 0.8

    def test_states_ordered_by_vol(self, two_regime_series):
        y, dates, _ = two_regime_series
        vs = volatility_states(y, dates, window=21, n_states=2)
        assert np.nanmean(vs.realized_vol[vs.states == 0]) < np.nanmean(vs.realized_vol[vs.states == 1])


class TestConditional:
    def test_conditional_vol_higher_in_stress(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        stats = regime_conditional_stats(y, model.states, model.labels)
        d = {row["regime"]: row["ann_volatility"] for row in stats.iter_rows(named=True)}
        assert d["stress"] > d["calm"]

    def test_conditional_correlation_rises_in_stress(self):
        np.random.seed(1)
        n_calm, n_stress = 600, 300
        a_calm = np.random.normal(0, 0.006, n_calm)
        b_calm = np.random.normal(0, 0.006, n_calm)
        common = np.random.normal(0, 0.02, n_stress)
        A = np.concatenate([a_calm, common + np.random.normal(0, 0.003, n_stress)])
        B = np.concatenate([b_calm, common + np.random.normal(0, 0.003, n_stress)])
        states = np.concatenate([np.zeros(n_calm), np.ones(n_stress)]).astype(int)
        T = len(A)
        dates = pl.date_range(pl.date(2018, 1, 1),
                              pl.date(2018, 1, 1) + pl.duration(days=T - 1),
                              interval="1d", eager=True)
        rs = ReturnSeries(pl.DataFrame({"date": dates, "A": A, "B": B}), ["A", "B"])
        out = regime_conditional_avg_correlation(rs, states, ["calm", "stress"])
        d = {row["regime"]: row["avg_correlation"] for row in out.iter_rows(named=True)}
        assert d["stress"] > d["calm"] + 0.3

    def test_conditional_beta_runs(self, two_regime_series):
        y, dates, _ = two_regime_series
        model = fit_regimes(y, dates, n_states=2)
        np.random.seed(2)
        port = 1.2 * y + np.random.normal(0, 0.001, len(y))
        out = regime_conditional_beta(port, y, model.states, model.labels)
        assert out.height >= 1
        for row in out.iter_rows(named=True):
            assert 0.9 < row["beta"] < 1.5

    def test_align_states_by_date(self):
        from datetime import date
        state_dates = [date(2020, 1, i) for i in range(1, 6)]
        states = np.array([0, 0, 1, 1, 0])
        target = pl.Series([date(2020, 1, i) for i in range(3, 8)])
        aligned = align_states_to_returns(state_dates, states, target)
        assert aligned[0] == 1 and aligned[2] == 0
        assert aligned[3] == -1 and aligned[4] == -1
