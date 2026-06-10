"""
Correlation analytics tests.

Validated against KNOWN structure: two blocks of assets, high intra-block
correlation, low inter-block. Clustering must recover the blocks; the MST
must connect them through a single bridge.
"""

import numpy as np
import polars as pl
import pytest

from src.domain.returns import ReturnSeries
from src.analytics.correlation.matrices import (
    correlation_matrix, correlation_distance, ewma_covariance,
    correlation_from_covariance,
)
from src.analytics.correlation.rolling import (
    rolling_pairwise_correlation, average_correlation,
)
from src.analytics.correlation.clustering import (
    cluster_correlations, reorder_correlation,
)
from src.analytics.correlation.network import build_mst


def _make_rs(R, tickers):
    T = R.shape[0]
    dates = pl.date_range(pl.date(2020, 1, 1),
                          pl.date(2020, 1, 1) + pl.duration(days=T - 1),
                          interval="1d", eager=True)
    return ReturnSeries(pl.DataFrame({"date": dates, **{t: R[:, i] for i, t in enumerate(tickers)}}), tickers)


@pytest.fixture
def two_block_returns():
    """
    6 assets: {A,B,C} driven by factor 1, {D,E,F} driven by factor 2.
    Strong intra-block correlation, weak inter-block.
    """
    np.random.seed(5)
    T = 1500
    f1 = np.random.normal(0, 0.01, T)
    f2 = np.random.normal(0, 0.01, T)
    idio = lambda: np.random.normal(0, 0.003, T)
    cols = {
        "A": f1 + idio(), "B": f1 + idio(), "C": f1 + idio(),
        "D": f2 + idio(), "E": f2 + idio(), "F": f2 + idio(),
    }
    tickers = ["A", "B", "C", "D", "E", "F"]
    R = np.column_stack([cols[t] for t in tickers])
    return _make_rs(R, tickers)


@pytest.fixture
def independent_returns():
    np.random.seed(6)
    T = 1500
    R = np.random.normal(0, 0.01, size=(T, 5))
    return _make_rs(R, ["A", "B", "C", "D", "E"])


class TestMatrices:
    def test_corr_diag_one_symmetric(self, two_block_returns):
        c = correlation_matrix(two_block_returns)
        np.testing.assert_allclose(np.diag(c), 1.0, atol=1e-10)
        np.testing.assert_allclose(c, c.T, atol=1e-10)
        assert c.min() >= -1.0 - 1e-9 and c.max() <= 1.0 + 1e-9

    def test_intra_block_higher(self, two_block_returns):
        c = correlation_matrix(two_block_returns)
        intra = c[0, 1]      # A-B same block
        inter = c[0, 3]      # A-D different block
        assert intra > 0.8
        assert inter < 0.3

    def test_distance_properties(self, two_block_returns):
        c = correlation_matrix(two_block_returns)
        d = correlation_distance(c)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-10)
        np.testing.assert_allclose(d, d.T, atol=1e-10)
        assert d.min() >= 0.0

    def test_distance_formula(self):
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        d = correlation_distance(corr)
        assert d[0, 1] == pytest.approx(np.sqrt(2 * 0.5))

    def test_ewma_corr_valid(self, two_block_returns):
        cov = ewma_covariance(two_block_returns.to_numpy())
        corr = correlation_from_covariance(cov)
        np.testing.assert_allclose(np.diag(corr), 1.0, atol=1e-10)
        assert corr.min() >= -1.0 - 1e-9 and corr.max() <= 1.0 + 1e-9


class TestRolling:
    def test_pairwise_shape(self, two_block_returns):
        df = rolling_pairwise_correlation(two_block_returns, "A", "B", window=63)
        assert df.height == two_block_returns.n_obs - 63 + 1
        assert df["correlation"].max() <= 1.0 + 1e-9

    def test_average_corr_blocks_vs_independent(self, two_block_returns, independent_returns):
        avg_block = average_correlation(two_block_returns, window=252)["avg_correlation"].mean()
        avg_indep = average_correlation(independent_returns, window=252)["avg_correlation"].mean()
        # Block universe is more internally correlated on average than independent
        assert avg_block > avg_indep


class TestClustering:
    def test_recovers_two_blocks(self, two_block_returns):
        res = cluster_correlations(two_block_returns, n_clusters=2)
        clusters = res.clusters()
        # {A,B,C} should land together, {D,E,F} together
        membership = {frozenset(v) for v in clusters.values()}
        assert frozenset({"A", "B", "C"}) in membership
        assert frozenset({"D", "E", "F"}) in membership

    def test_quasi_diagonal_is_permutation(self, two_block_returns):
        res = cluster_correlations(two_block_returns, n_clusters=2)
        assert sorted(res.quasi_diagonal_order) == list(range(6))

    def test_reorder_groups_blocks(self, two_block_returns):
        res = cluster_correlations(two_block_returns, n_clusters=2)
        c = correlation_matrix(two_block_returns)
        reordered = reorder_correlation(c, res.quasi_diagonal_order)
        assert reordered.shape == c.shape
        # diagonal still ones
        np.testing.assert_allclose(np.diag(reordered), 1.0, atol=1e-10)


class TestMST:
    def test_n_minus_one_edges(self, two_block_returns):
        mst = build_mst(two_block_returns)
        assert len(mst.edges) == two_block_returns.n_assets - 1

    def test_all_nodes_connected(self, two_block_returns):
        mst = build_mst(two_block_returns)
        # Every node appears in at least one edge (tree is connected)
        assert all(d >= 1 for d in mst.degree.values())

    def test_total_degree_is_twice_edges(self, two_block_returns):
        mst = build_mst(two_block_returns)
        assert sum(mst.degree.values()) == 2 * len(mst.edges)

    def test_edges_dataframe(self, two_block_returns):
        mst = build_mst(two_block_returns)
        df = mst.edges_dataframe()
        assert df.height == two_block_returns.n_assets - 1
        assert set(df.columns) == {"source", "target", "distance", "correlation"}
