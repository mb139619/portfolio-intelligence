"""
Covariance estimator tests.

Validated against known properties rather than smoke:
  - legacy methods are bit-for-bit the previous implementation (no regression),
  - constant-correlation Ledoit-Wolf: δ ∈ [0,1], PSD, and beats the sample
    covariance in Frobenius loss on a small-sample equicorrelated DGP,
  - factor-implied covariance reconstructs the structural Σ = B Λ Bᵀ + D.
"""

import numpy as np
import polars as pl
import pytest

from src.analytics.risk.covariance import (
    estimate_covariance,
    factor_covariance,
)
from src.domain.returns import ReturnSeries


def _make_rs(R: np.ndarray, tickers: list[str]) -> ReturnSeries:
    import datetime as dt
    n = R.shape[0]
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    data = pl.DataFrame({"date": dates,
                         **{t: R[:, j] for j, t in enumerate(tickers)}})
    return ReturnSeries(data, tickers)


@pytest.fixture
def rs5():
    rng = np.random.default_rng(0)
    tickers = ["A", "B", "C", "D", "E"]
    R = rng.standard_normal((600, 5)) * 0.01
    return _make_rs(R, tickers)


class TestBackwardCompat:
    def test_sample_matches_npcov(self, rs5):
        R = rs5.to_numpy()
        got = estimate_covariance(rs5, method="sample", ppy=252).matrix
        np.testing.assert_allclose(got, np.cov(R, rowvar=False, ddof=1) * 252)

    def test_ledoit_wolf_matches_sklearn(self, rs5):
        from sklearn.covariance import LedoitWolf
        R = rs5.to_numpy()
        got = estimate_covariance(rs5, method="ledoit_wolf", ppy=252).matrix
        np.testing.assert_allclose(got, LedoitWolf().fit(R).covariance_ * 252)

    def test_ewma_matches_primitive(self, rs5):
        from src.analytics.correlation.matrices import ewma_covariance
        R = rs5.to_numpy()
        got = estimate_covariance(rs5, method="ewma", ppy=252).matrix
        np.testing.assert_allclose(got, ewma_covariance(R, 0.94) * 252)

    def test_decompose_risk_delegate_unchanged(self, rs5):
        # covariance_matrix now delegates; result must equal the estimator's.
        from src.analytics.risk.decomposition import covariance_matrix
        a = covariance_matrix(rs5, method="sample", ppy=252)
        b = estimate_covariance(rs5, method="sample", ppy=252).matrix
        np.testing.assert_allclose(a, b)


class TestResultHelpers:
    def test_psd_and_symmetric(self, rs5):
        res = estimate_covariance(rs5, method="ledoit_wolf_cc")
        np.testing.assert_allclose(res.matrix, res.matrix.T)
        assert res.is_psd()

    def test_correlation_unit_diagonal(self, rs5):
        res = estimate_covariance(rs5, method="sample")
        np.testing.assert_allclose(np.diag(res.correlation), np.ones(5), atol=1e-12)

    def test_volatilities_match_diagonal(self, rs5):
        res = estimate_covariance(rs5, method="sample")
        np.testing.assert_allclose(res.volatilities, np.sqrt(np.diag(res.matrix)))


class TestLedoitWolfCC:
    def test_shrinkage_in_unit_interval(self, rs5):
        res = estimate_covariance(rs5, method="ledoit_wolf_cc")
        assert 0.0 <= res.shrinkage <= 1.0

    def test_shrinkage_decreases_with_sample_size(self):
        # More data → less need to shrink.
        rng = np.random.default_rng(1)
        small = _make_rs(rng.standard_normal((40, 8)) * 0.01, list("ABCDEFGH"))
        large = _make_rs(rng.standard_normal((4000, 8)) * 0.01, list("ABCDEFGH"))
        d_small = estimate_covariance(small, method="ledoit_wolf_cc").shrinkage
        d_large = estimate_covariance(large, method="ledoit_wolf_cc").shrinkage
        assert d_small > d_large

    def test_beats_sample_in_frobenius_on_equicorrelated_small_sample(self):
        # True Σ: unit vars, constant correlation 0.5. With T barely > N the
        # sample covariance is noisy; CC shrinkage should be closer to truth.
        rng = np.random.default_rng(7)
        n, T, rho = 12, 30, 0.5
        true = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
        L = np.linalg.cholesky(true)
        losses_sample, losses_cc = [], []
        for _ in range(50):
            X = rng.standard_normal((T, n)) @ L.T
            rs = _make_rs(X, [f"A{i}" for i in range(n)])
            s = estimate_covariance(rs, method="sample", annualize=False).matrix
            cc = estimate_covariance(
                rs, method="ledoit_wolf_cc", annualize=False
            ).matrix
            losses_sample.append(np.sum((s - true) ** 2))
            losses_cc.append(np.sum((cc - true) ** 2))
        assert np.mean(losses_cc) < np.mean(losses_sample)

    def test_converges_to_sample_when_target_is_true(self):
        # When the constant-correlation target IS the truth and T is large, the
        # CC estimate and the sample estimate both approach it and agree closely
        # (off-diagonals included — unlike the uncorrelated case, where CC
        # correctly shrinks noise off-diagonals to ~0 and SHOULD differ).
        rng = np.random.default_rng(3)
        n, T, rho = 6, 8000, 0.4
        true = (1 - rho) * np.eye(n) + rho * np.ones((n, n))
        X = (rng.standard_normal((T, n)) @ np.linalg.cholesky(true).T) * 0.01
        rs = _make_rs(X, list("ABCDEF"))
        s = estimate_covariance(rs, method="sample", annualize=False).matrix
        cc = estimate_covariance(rs, method="ledoit_wolf_cc", annualize=False).matrix
        np.testing.assert_allclose(cc, s, rtol=0.10, atol=1e-7)


class TestFactorCovariance:
    def test_psd_and_shape(self):
        rng = np.random.default_rng(5)
        T, N, K = 1000, 6, 3
        F = rng.standard_normal((T, K)) * 0.01
        B = rng.standard_normal((N, K))
        Y = F @ B.T + rng.standard_normal((T, N)) * 0.005
        res = factor_covariance(Y, F, list("ABCDEF"))
        assert res.matrix.shape == (N, N)
        assert res.is_psd()
        assert res.method == "factor"

    def test_recovers_structure(self):
        # Build assets from known factors; factor-implied Σ should be close to
        # the realised sample Σ (both estimate the same population object).
        rng = np.random.default_rng(9)
        T, N, K = 5000, 5, 2
        F = rng.standard_normal((T, K)) * 0.01
        B = np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0],
                      [0.5, 0.5], [0.3, -0.4]])
        Y = F @ B.T + rng.standard_normal((T, N)) * 0.003
        fac = factor_covariance(Y, F, list("ABCDE"), annualize=False).matrix
        sample = np.cov(Y, rowvar=False, ddof=1)
        np.testing.assert_allclose(fac, sample, atol=5e-6)


def test_unknown_method_raises(rs5):
    with pytest.raises(ValueError):
        estimate_covariance(rs5, method="nope")
