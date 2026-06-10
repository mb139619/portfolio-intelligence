"""
Unit tests — analytics on synthetic data + ParquetStore round-trip.
No live network needed.
"""

import numpy as np
import polars as pl
import pytest

from src.analytics.performance import (
    total_return, annualized_return, annualized_volatility,
    sharpe_ratio, max_drawdown, historical_var, historical_cvar,
    compute_metrics,
)
from src.analytics.risk.decomposition import covariance_matrix, decompose_risk
from src.domain.portfolio import Portfolio, Position, Asset, AssetClass
from src.domain.returns import ReturnSeries
from src.store.parquet_store import ParquetStore


@pytest.fixture
def random_returns():
    np.random.seed(42)
    n = 250
    r = np.random.normal(0.0005, 0.01, size=(n, 3))
    dates = pl.date_range(pl.date(2023, 1, 1),
                          pl.date(2023, 1, 1) + pl.duration(days=n - 1),
                          interval="1d", eager=True)
    df = pl.DataFrame({"date": dates, "SPY": r[:, 0], "TLT": r[:, 1], "GLD": r[:, 2]})
    return ReturnSeries(data=df, tickers=["SPY", "TLT", "GLD"])


@pytest.fixture
def synthetic_prices():
    np.random.seed(7)
    n = 300
    dates = pl.date_range(pl.date(2022, 1, 1),
                          pl.date(2022, 1, 1) + pl.duration(days=n - 1),
                          interval="1d", eager=True)
    price = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.012, n))
    return pl.DataFrame({
        "date": dates, "ticker": ["SPY"] * n,
        "open": price, "high": price * 1.01, "low": price * 0.99,
        "close": price, "adj_close": price, "volume": [1_000_000] * n,
    })


class TestPerformance:
    def test_total_return_flat(self):
        assert total_return(np.zeros(100)) == pytest.approx(0.0)

    def test_annualized_sign(self):
        assert annualized_return(np.full(252, 0.001)) > 0
        assert annualized_return(np.full(252, -0.001)) < 0

    def test_vol_nonneg(self):
        assert annualized_volatility(np.random.normal(0, 0.01, 252)) >= 0

    def test_mdd_negative(self):
        assert max_drawdown(np.array([0.01, -0.05, 0.02, -0.03])) < 0

    def test_mdd_zero_when_monotone(self):
        assert max_drawdown(np.full(50, 0.01)) == pytest.approx(0.0, abs=1e-10)

    def test_var_positive(self):
        assert historical_var(np.random.normal(0, 0.01, 1000)) > 0

    def test_cvar_geq_var(self):
        r = np.random.normal(0, 0.01, 1000)
        assert historical_cvar(r, 0.95) >= historical_var(r, 0.95)

    def test_metrics_run(self, random_returns):
        m = compute_metrics(random_returns.to_numpy_series("SPY"))
        assert m.n_observations == 250
        assert 0 <= m.hit_rate <= 1


class TestRisk:
    def test_rc_sums_to_vol(self, random_returns):
        d = decompose_risk({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}, random_returns)
        assert np.sum(d.risk_contribution) == pytest.approx(d.portfolio_volatility, rel=1e-6)

    def test_pct_rc_sums_to_one(self, random_returns):
        d = decompose_risk({"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}, random_returns)
        assert np.sum(d.percent_contribution) == pytest.approx(1.0, rel=1e-6)

    def test_cov_symmetric_psd(self, random_returns):
        cov = covariance_matrix(random_returns)
        np.testing.assert_array_almost_equal(cov, cov.T)
        assert np.all(np.linalg.eigvalsh(cov) >= -1e-10)

    def test_cov_methods(self, random_returns):
        for m in ("sample", "ledoit_wolf", "ewma"):
            cov = covariance_matrix(random_returns, method=m)
            assert cov.shape == (3, 3)


class TestPortfolio:
    def test_weights_sum(self):
        p = Portfolio.equal_weight("EW", [
            Asset("A", "A", AssetClass.EQUITY),
            Asset("B", "B", AssetClass.EQUITY),
        ])
        assert sum(p.weights.values()) == pytest.approx(1.0)

    def test_invalid_weights(self):
        a = Asset("SPY", "S&P", AssetClass.EQUITY)
        with pytest.raises(ValueError):
            Portfolio("Bad", [Position(a, 0.5), Position(a, 0.3)])


class TestParquetStore:
    def test_write_read_roundtrip(self, tmp_path, synthetic_prices):
        store = ParquetStore(tmp_path)
        n = store.write_prices("SPY", synthetic_prices)
        assert n == len(synthetic_prices)
        assert store.available_tickers() == ["SPY"]

        wide = store.read_prices(["SPY"])
        assert "SPY" in wide.columns
        assert len(wide) == len(synthetic_prices)

    def test_returns_from_store(self, tmp_path, synthetic_prices):
        store = ParquetStore(tmp_path)
        store.write_prices("SPY", synthetic_prices)
        rs = store.read_returns(["SPY"])
        assert rs.n_obs == len(synthetic_prices) - 1   # one lost to differencing

    def test_upsert_dedupes(self, tmp_path, synthetic_prices):
        store = ParquetStore(tmp_path)
        store.write_prices("SPY", synthetic_prices)
        # Write again — should dedupe on date, not double up
        store.write_prices("SPY", synthetic_prices, upsert=True)
        assert store.read_prices(["SPY"]).height == len(synthetic_prices)

    def test_last_date(self, tmp_path, synthetic_prices):
        store = ParquetStore(tmp_path)
        store.write_prices("SPY", synthetic_prices)
        assert store.last_date("SPY") == synthetic_prices["date"].max()

    def test_sql_engine(self, tmp_path, synthetic_prices):
        store = ParquetStore(tmp_path)
        store.write_prices("SPY", synthetic_prices)
        out = store.sql("SELECT ticker, count(*) AS n FROM prices() GROUP BY ticker")
        assert out["n"][0] == len(synthetic_prices)
