"""
Tests for the methodological fixes:
  - Sharpe/Sortino computed on excess returns; risk-free as scalar OR series
  - return attribution reconciles EXACTLY (alpha + factors = total excess)
"""

import numpy as np
import polars as pl
import pytest

from src.analytics.performance import (
    sharpe_ratio, excess_returns, to_daily_rf, align_risk_free, compute_metrics,
)
from src.analytics.factors.prepare import AlignedFactorData
from src.analytics.factors.engine import estimate_factor_model
from src.analytics.factors.attribution import attribute_returns


class TestRiskFreeHandling:
    def test_scalar_rf_to_daily(self):
        rf = to_daily_rf(0.0252, n=10, ppy=252)
        assert rf.shape == (10,)
        assert rf[0] == pytest.approx(0.0001)

    def test_series_rf_passthrough(self):
        series = np.full(5, 0.0002)
        rf = to_daily_rf(series, n=5)
        np.testing.assert_allclose(rf, series)

    def test_series_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            to_daily_rf(np.zeros(4), n=5)

    def test_excess_returns_subtract_rf(self):
        r = np.full(10, 0.001)
        ex = excess_returns(r, 0.0252, ppy=252)  # daily rf ≈ 0.0001
        np.testing.assert_allclose(ex, 0.0009, atol=1e-9)

    def test_sharpe_with_zero_rf_positive_drift(self):
        np.random.seed(1)
        r = np.random.normal(0.0005, 0.01, 1000)
        s = sharpe_ratio(r, rf=0.0)
        assert s > 0

    def test_sharpe_scalar_vs_series_consistency(self):
        # A constant rf series must give the same Sharpe as the scalar rate
        np.random.seed(2)
        r = np.random.normal(0.0004, 0.01, 800)
        rate = 0.0252
        s_scalar = sharpe_ratio(r, rf=rate)
        rf_series = np.full(len(r), rate / 252)
        s_series = sharpe_ratio(r, rf=rf_series)
        assert s_scalar == pytest.approx(s_series, rel=1e-9)

    def test_higher_rf_lowers_sharpe(self):
        np.random.seed(3)
        r = np.random.normal(0.0005, 0.01, 1000)
        assert sharpe_ratio(r, rf=0.00) > sharpe_ratio(r, rf=0.05)

    def test_compute_metrics_flags_rf_kind(self):
        np.random.seed(4)
        r = np.random.normal(0.0004, 0.01, 500)
        m_scalar = compute_metrics(r, rf=0.04)
        m_series = compute_metrics(r, rf=np.full(500, 0.04 / 252))
        assert m_scalar.rf_kind == "scalar"
        assert m_series.rf_kind == "series"

    def test_align_risk_free(self):
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10),
                              interval="1d", eager=True)
        returns = pl.Series("ret", np.random.normal(0, 0.01, len(dates)))
        rf_wide = pl.DataFrame({
            "date": dates[2:8],
            "RF": np.full(6, 0.0001),
        })
        r_al, rf_al, d_al = align_risk_free(returns, dates, rf_wide)
        assert len(r_al) == 6 == len(rf_al)


class TestAttributionReconciliation:
    @pytest.fixture
    def model_and_data(self):
        np.random.seed(42)
        T = 2000
        F = np.random.normal(0, 0.01, size=(T, 5))
        betas = np.array([1.1, 0.2, -0.4, 0.1, -0.3])
        alpha = 0.0002
        y = alpha + F @ betas + np.random.normal(0, 0.003, T)
        aligned = AlignedFactorData(
            list(range(T)), y, F, ["Mkt-RF", "SMB", "HML", "RMW", "CMA"], np.zeros(T)
        )
        return estimate_factor_model(aligned), aligned

    def test_reconciles_to_machine_precision(self, model_and_data):
        model, aligned = model_and_data
        attr = attribute_returns(model, aligned)
        # alpha + sum(factor contributions) == total excess, exactly
        reconstructed = attr.alpha_annualized + sum(attr.factor_contribution.values())
        assert reconstructed == pytest.approx(attr.total_excess_annualized, abs=1e-10)

    def test_reconciliation_error_is_zero(self, model_and_data):
        model, aligned = model_and_data
        attr = attribute_returns(model, aligned)
        assert abs(attr.reconciliation_error) < 1e-10

    def test_alpha_annualized_is_arithmetic(self, model_and_data):
        model, aligned = model_and_data
        # alpha_annualized == daily alpha × 252 (not compounded)
        assert model.alpha_annualized == pytest.approx(model.alpha * 252, rel=1e-12)


class TestEWMAZeroMean:
    def test_ewma_is_zero_mean_convention(self):
        # Zero-mean EWMA = weighted sum of outer products of RAW returns,
        # NOT demeaned. Verify against an explicit raw-return computation.
        from src.analytics.correlation.matrices import ewma_covariance
        np.random.seed(5)
        R = np.random.normal(0.001, 0.01, size=(500, 3))
        cov = ewma_covariance(R, lam=0.94)

        T = R.shape[0]
        w = np.array([(1 - 0.94) * 0.94 ** i for i in range(T - 1, -1, -1)])
        w /= w.sum()
        expected = np.einsum("t,ti,tj->ij", w, R, R)  # raw, no demeaning
        np.testing.assert_allclose(cov, expected, rtol=1e-12)

    def test_ewma_symmetric_psd(self):
        from src.analytics.correlation.matrices import ewma_covariance
        np.random.seed(6)
        R = np.random.normal(0, 0.01, size=(400, 4))
        cov = ewma_covariance(R)
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)

    def test_decomposition_ewma_matches_matrices(self):
        # The risk-decomposition EWMA branch must equal the matrices.py
        # implementation (× annualisation), i.e. one source of truth.
        from src.analytics.correlation.matrices import ewma_covariance as ewma_m
        from src.analytics.risk.decomposition import covariance_matrix
        from src.domain.returns import ReturnSeries

        np.random.seed(7)
        T = 600
        R = np.random.normal(0, 0.01, size=(T, 3))
        dates = pl.date_range(pl.date(2020, 1, 1),
                              pl.date(2020, 1, 1) + pl.duration(days=T - 1),
                              interval="1d", eager=True)
        rs = ReturnSeries(pl.DataFrame({"date": dates, "A": R[:, 0], "B": R[:, 1], "C": R[:, 2]}),
                          ["A", "B", "C"])
        from_decomp = covariance_matrix(rs, method="ewma", ppy=252)
        from_matrices = ewma_m(R, lam=0.94) * 252
        np.testing.assert_allclose(from_decomp, from_matrices, rtol=1e-12)
