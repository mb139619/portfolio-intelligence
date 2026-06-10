"""
PCA risk model + Hidden Concentration Detector tests.

Validated against KNOWN structures:
  - one common factor  -> PC1 explains ~everything, ENB_risk ~ 1
  - independent assets  -> explained variance spread out, ENB_risk ~ N
"""

import numpy as np
import polars as pl
import pytest

from src.domain.returns import ReturnSeries
from src.analytics.pca.model import fit_pca
from src.analytics.pca.concentration import detect_hidden_concentration


def _make_rs(returns: np.ndarray, tickers: list[str]) -> ReturnSeries:
    T = returns.shape[0]
    dates = pl.date_range(pl.date(2020, 1, 1),
                          pl.date(2020, 1, 1) + pl.duration(days=T - 1),
                          interval="1d", eager=True)
    data = {"date": dates}
    for i, t in enumerate(tickers):
        data[t] = returns[:, i]
    return ReturnSeries(data=pl.DataFrame(data), tickers=tickers)


@pytest.fixture
def one_factor_returns():
    """5 assets all driven by ONE common factor + tiny idiosyncratic noise."""
    np.random.seed(1)
    T = 1500
    common = np.random.normal(0, 0.01, T)
    betas = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    noise = np.random.normal(0, 0.001, size=(T, 5))
    R = np.outer(common, betas) + noise
    return _make_rs(R, ["A", "B", "C", "D", "E"])


@pytest.fixture
def independent_returns():
    """5 mutually independent assets with similar volatility."""
    np.random.seed(2)
    T = 1500
    R = np.random.normal(0, 0.01, size=(T, 5))
    return _make_rs(R, ["A", "B", "C", "D", "E"])


class TestPCAModel:
    def test_explained_variance_sums_to_one(self, independent_returns):
        m = fit_pca(independent_returns)
        assert m.explained_variance_ratio.sum() == pytest.approx(1.0, rel=1e-6)

    def test_eigenvalues_descending_nonneg(self, independent_returns):
        m = fit_pca(independent_returns)
        assert np.all(np.diff(m.eigenvalues) <= 1e-12)
        assert np.all(m.eigenvalues >= -1e-12)

    def test_scores_variance_equals_eigenvalues(self, one_factor_returns):
        # For covariance PCA, var(scores_i) == eigenvalue_i
        m = fit_pca(one_factor_returns)
        score_var = m.scores.var(axis=0, ddof=1)
        np.testing.assert_allclose(score_var, m.eigenvalues, rtol=1e-6, atol=1e-12)

    def test_one_factor_pc1_dominates(self, one_factor_returns):
        m = fit_pca(one_factor_returns)
        # First PC should explain the vast majority of variance
        assert m.explained_variance_ratio[0] > 0.95
        assert m.n_factors_for(0.90) == 1

    def test_independent_spread_variance(self, independent_returns):
        m = fit_pca(independent_returns)
        # No single PC should dominate
        assert m.explained_variance_ratio[0] < 0.45
        assert m.n_factors_for(0.90) >= 4

    def test_eigen_portfolio_sums_to_one(self, independent_returns):
        m = fit_pca(independent_returns)
        ep = m.eigen_portfolio(0, normalize="sum")
        assert sum(ep.values()) == pytest.approx(1.0, abs=1e-9)

    def test_correlation_method(self, independent_returns):
        m = fit_pca(independent_returns, method="correlation")
        assert m.explained_variance_ratio.sum() == pytest.approx(1.0, rel=1e-6)


class TestHiddenConcentration:
    def test_pc_shares_sum_to_one(self, independent_returns):
        w = {t: 0.2 for t in independent_returns.tickers}
        rep = detect_hidden_concentration(w, independent_returns)
        assert rep.pc_variance_contribution.sum() == pytest.approx(1.0, rel=1e-6)

    def test_one_factor_low_enb(self, one_factor_returns):
        # Equal weights across 5 names → ENB_weights = 5,
        # but all risk is one factor → ENB_risk ≈ 1: the hidden concentration.
        w = {t: 0.2 for t in one_factor_returns.tickers}
        rep = detect_hidden_concentration(w, one_factor_returns)
        assert rep.effective_bets_weights == pytest.approx(5.0, rel=1e-6)
        assert rep.effective_bets_risk < 1.5
        assert rep.top_pc_share > 0.9
        assert "HIGH" in rep.verdict()

    def test_independent_high_enb(self, independent_returns):
        # Equal weights across 5 independent assets → ENB_risk ≈ 5
        w = {t: 0.2 for t in independent_returns.tickers}
        rep = detect_hidden_concentration(w, independent_returns)
        assert rep.effective_bets_risk > 4.0
        assert "LOW" in rep.verdict()

    def test_enb_risk_le_n(self, independent_returns):
        w = {t: 0.2 for t in independent_returns.tickers}
        rep = detect_hidden_concentration(w, independent_returns)
        assert 1.0 <= rep.effective_bets_risk <= 5.0 + 1e-9

    def test_reuses_passed_model(self, one_factor_returns):
        w = {t: 0.2 for t in one_factor_returns.tickers}
        model = fit_pca(one_factor_returns)
        rep = detect_hidden_concentration(w, one_factor_returns, model=model)
        assert rep.effective_bets_risk < 1.5
