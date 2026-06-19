"""
Minimum-variance optimiser tests — validated against optimality invariants.

  - weights are a valid long-only portfolio (≥0, sum to 1, respect caps),
  - the min-variance portfolio has lower variance than equal weight (it is the
    minimum) and than any feasible alternative,
  - on a diagonal Σ the long-only solution is inverse-variance,
  - the long-short closed form matches Σ⁻¹1/(1ᵀΣ⁻¹1) and has variance ≤ the
    long-only solution (it optimises over a larger set).
"""

import numpy as np
import pytest

from src.analytics.optimization import OptimizationResult, min_variance


def _vol(Sigma, w):
    return float(np.sqrt(w @ Sigma @ w))


@pytest.fixture
def Sigma3():
    # 3 assets, distinct vols and mild correlations
    vol = np.array([0.10, 0.20, 0.30])
    C = np.array([[1.0, 0.3, 0.1],
                  [0.3, 1.0, 0.2],
                  [0.1, 0.2, 1.0]])
    return np.outer(vol, vol) * C


class TestLongOnly:
    def test_valid_portfolio(self, Sigma3):
        r = min_variance(Sigma3)
        assert isinstance(r, OptimizationResult) and r.success
        assert r.weights.min() >= -1e-9
        assert r.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_lower_variance_than_equal_weight(self, Sigma3):
        r = min_variance(Sigma3)
        n = Sigma3.shape[0]
        ew = np.ones(n) / n
        assert _vol(Sigma3, r.weights) <= _vol(Sigma3, ew) + 1e-12

    def test_no_alternative_beats_it(self, Sigma3):
        # Random feasible long-only portfolios must not have lower variance.
        r = min_variance(Sigma3)
        opt_v = _vol(Sigma3, r.weights)
        rng = np.random.default_rng(0)
        for _ in range(2000):
            w = rng.random(3)
            w /= w.sum()
            assert _vol(Sigma3, w) >= opt_v - 1e-8

    def test_diagonal_is_inverse_variance(self):
        Sigma = np.diag([0.04, 0.01, 0.09])      # variances
        r = min_variance(Sigma)
        inv = 1.0 / np.diag(Sigma)
        expected = inv / inv.sum()
        np.testing.assert_allclose(r.weights, expected, atol=1e-5)  # SLSQP tol

    def test_two_asset_analytic(self):
        s1, s2, rho = 0.15, 0.25, 0.2
        Sigma = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])
        r = min_variance(Sigma)
        w1 = (s2**2 - rho*s1*s2) / (s1**2 + s2**2 - 2*rho*s1*s2)
        np.testing.assert_allclose(r.weights, [w1, 1 - w1], atol=1e-6)

    def test_max_weight_cap_respected(self, Sigma3):
        r = min_variance(Sigma3, max_weight=0.5)
        assert r.weights.max() <= 0.5 + 1e-6
        assert r.weights.sum() == pytest.approx(1.0, abs=1e-8)

    def test_infeasible_cap_raises(self, Sigma3):
        with pytest.raises(ValueError):
            min_variance(Sigma3, max_weight=0.2)   # 0.2 × 3 < 1


class TestLongShort:
    def test_matches_closed_form(self, Sigma3):
        r = min_variance(Sigma3, long_only=False)
        z = np.linalg.solve(Sigma3, np.ones(3))
        expected = z / z.sum()
        np.testing.assert_allclose(r.weights, expected, atol=1e-12)
        assert r.weights.sum() == pytest.approx(1.0, abs=1e-10)

    def test_variance_not_above_long_only(self, Sigma3):
        # Larger feasible set ⇒ variance no higher than the long-only optimum.
        ls = min_variance(Sigma3, long_only=False)
        lo = min_variance(Sigma3, long_only=True)
        assert _vol(Sigma3, ls.weights) <= _vol(Sigma3, lo.weights) + 1e-9


def test_accepts_covariance_result():
    import datetime as dt

    import polars as pl

    from src.analytics.risk.covariance import estimate_covariance
    from src.domain.returns import ReturnSeries

    rng = np.random.default_rng(2)
    R = rng.standard_normal((500, 4)) * 0.01
    dates = [dt.date(2021, 1, 1) + dt.timedelta(days=i) for i in range(500)]
    rs = ReturnSeries(pl.DataFrame({"date": dates,
                     **{t: R[:, j] for j, t in enumerate("ABCD")}}), list("ABCD"))
    cov = estimate_covariance(rs, method="ledoit_wolf_cc")
    r = min_variance(cov)
    assert r.tickers == ["A", "B", "C", "D"]
    assert r.success and r.weights.sum() == pytest.approx(1.0, abs=1e-8)



class TestEfficientFrontier:
    def test_leftmost_point_is_gmv(self, Sigma3):
        from src.analytics.optimization import efficient_frontier, min_variance
        mu = np.array([0.05, 0.08, 0.12])
        ef = efficient_frontier(Sigma3, mu, n_points=25)
        gmv = min_variance(Sigma3, long_only=True)
        v_left, _ = ef.min_variance_point
        assert abs(v_left - gmv.expected_volatility) < 1e-5

    def test_returns_and_vols_monotone(self, Sigma3):
        from src.analytics.optimization import efficient_frontier
        mu = np.array([0.05, 0.08, 0.12])
        ef = efficient_frontier(Sigma3, mu, n_points=30)
        # along the efficient half both target return and vol are non-decreasing
        assert np.all(np.diff(ef.returns) >= -1e-9)
        assert np.all(np.diff(ef.volatilities) >= -1e-6)

    def test_weights_valid_long_only(self, Sigma3):
        from src.analytics.optimization import efficient_frontier
        mu = np.array([0.05, 0.08, 0.12])
        ef = efficient_frontier(Sigma3, mu, n_points=15, long_only=True)
        assert np.all(ef.weights >= -1e-6)
        np.testing.assert_allclose(ef.weights.sum(axis=1), 1.0, atol=1e-6)

    def test_mu_length_mismatch_raises(self, Sigma3):
        from src.analytics.optimization import efficient_frontier
        with pytest.raises(ValueError):
            efficient_frontier(Sigma3, np.array([0.1, 0.1]))
