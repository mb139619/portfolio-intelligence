"""
Factor engine tests.

The gold-standard test for an estimator: generate data from a KNOWN factor
model, then check the engine recovers the true parameters.
"""

import numpy as np
import polars as pl
import pytest

from src.analytics.factors.prepare import align_factors, AlignedFactorData
from src.analytics.factors.engine import estimate_factor_model, rolling_betas
from src.analytics.factors.attribution import (
    decompose_factor_risk, attribute_returns, factor_covariance,
)


@pytest.fixture
def known_model_data():
    """
    Generate returns from a known model:
        r_excess = alpha + 1.2*Mkt + (-0.4)*SMB + 0.6*HML + eps
    with small noise so the estimator should recover betas precisely.
    """
    np.random.seed(123)
    T = 2000
    true_alpha = 0.0002
    true_betas = np.array([1.2, -0.4, 0.6])

    # Factor returns (daily, realistic-ish scales)
    factors = np.random.normal(0, 0.01, size=(T, 3))
    noise = np.random.normal(0, 0.003, size=T)        # idiosyncratic
    excess = true_alpha + factors @ true_betas + noise

    return AlignedFactorData(
        dates=list(range(T)),
        excess_returns=excess,
        factors=factors,
        factor_names=["Mkt-RF", "SMB", "HML"],
        rf=np.zeros(T),
    ), true_alpha, true_betas


class TestParameterRecovery:
    def test_betas_recovered(self, known_model_data):
        aligned, true_alpha, true_betas = known_model_data
        model = estimate_factor_model(aligned, hac_lags=5)
        est = model.beta_vector()
        np.testing.assert_allclose(est, true_betas, atol=0.02)

    def test_alpha_recovered(self, known_model_data):
        aligned, true_alpha, _ = known_model_data
        model = estimate_factor_model(aligned, hac_lags=5)
        assert model.alpha == pytest.approx(true_alpha, abs=2e-4)

    def test_high_r2_low_noise(self, known_model_data):
        aligned, _, _ = known_model_data
        model = estimate_factor_model(aligned, hac_lags=5)
        assert model.r_squared > 0.9

    def test_significant_betas(self, known_model_data):
        aligned, _, _ = known_model_data
        model = estimate_factor_model(aligned, hac_lags=5)
        # All three true factors should be highly significant
        for f in aligned.factor_names:
            assert model.beta_pvalues[f] < 0.01

    def test_ols_vs_hac_same_point_estimates(self, known_model_data):
        # HAC changes standard errors, NOT the coefficients
        aligned, _, _ = known_model_data
        m_ols = estimate_factor_model(aligned, hac_lags=None)
        m_hac = estimate_factor_model(aligned, hac_lags=5)
        np.testing.assert_allclose(m_ols.beta_vector(), m_hac.beta_vector(), atol=1e-10)


class TestFactorRiskDecomposition:
    def test_systematic_plus_specific(self, known_model_data):
        aligned, _, _ = known_model_data
        model = estimate_factor_model(aligned)
        decomp = decompose_factor_risk(model, aligned)
        # Shares sum to 1
        assert decomp.pct_systematic + decomp.pct_specific == pytest.approx(1.0, rel=1e-6)

    def test_factor_contributions_sum_to_systematic(self, known_model_data):
        aligned, _, _ = known_model_data
        model = estimate_factor_model(aligned)
        decomp = decompose_factor_risk(model, aligned)
        total_factor_share = sum(decomp.factor_variance_contribution.values())
        assert total_factor_share == pytest.approx(decomp.pct_systematic, rel=1e-6)

    def test_mostly_systematic_low_noise(self, known_model_data):
        aligned, _, _ = known_model_data
        model = estimate_factor_model(aligned)
        decomp = decompose_factor_risk(model, aligned)
        # With small idiosyncratic noise, risk is mostly systematic
        assert decomp.pct_systematic > 0.8

    def test_factor_cov_shape(self, known_model_data):
        aligned, _, _ = known_model_data
        cov = factor_covariance(aligned)
        assert cov.shape == (3, 3)
        np.testing.assert_array_almost_equal(cov, cov.T)


class TestReturnAttribution:
    def test_attribution_runs(self, known_model_data):
        aligned, _, _ = known_model_data
        model = estimate_factor_model(aligned)
        attr = attribute_returns(model, aligned)
        assert set(attr.factor_contribution.keys()) == set(aligned.factor_names)


class TestRollingBetas:
    def test_rolling_shape(self, known_model_data):
        aligned, _, true_betas = known_model_data
        roll = rolling_betas(aligned, window=252)
        n_windows = aligned.n_obs - 252 + 1
        assert len(roll["dates"]) == n_windows
        assert roll["betas"]["Mkt-RF"].shape == (n_windows,)

    def test_rolling_betas_near_true(self, known_model_data):
        aligned, _, true_betas = known_model_data
        roll = rolling_betas(aligned, window=500)
        # Average rolling beta should be close to the true Mkt beta
        assert roll["betas"]["Mkt-RF"].mean() == pytest.approx(true_betas[0], abs=0.05)


class TestAlignFactors:
    def test_align_inner_join(self):
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10),
                              interval="1d", eager=True)
        returns = pl.Series("ret", np.random.normal(0, 0.01, len(dates)))
        # Factor data covers only part of the range
        fdates = dates[2:8]
        factor_wide = pl.DataFrame({
            "date": fdates,
            "Mkt-RF": np.random.normal(0, 0.01, len(fdates)),
            "RF": np.full(len(fdates), 0.0001),
        })
        aligned = align_factors(returns, dates, factor_wide, ["Mkt-RF"])
        assert aligned.n_obs == len(fdates)   # inner join keeps overlap only

    def test_excess_return_subtracts_rf(self):
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 5),
                              interval="1d", eager=True)
        returns = pl.Series("ret", [0.01] * len(dates))
        factor_wide = pl.DataFrame({
            "date": dates,
            "Mkt-RF": [0.005] * len(dates),
            "RF": [0.002] * len(dates),
        })
        aligned = align_factors(returns, dates, factor_wide, ["Mkt-RF"])
        # excess = 0.01 - 0.002 = 0.008
        np.testing.assert_allclose(aligned.excess_returns, 0.008, atol=1e-9)
