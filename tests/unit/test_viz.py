"""
Tests for the viz module (figures are produced and well-formed) and for the
factor-alignment date-gap warning.
"""

import numpy as np
import polars as pl
import plotly.graph_objects as go
import pytest

from src.domain.returns import ReturnSeries
from src.analytics.factors.prepare import align_factors
from src.analytics.factors.engine import estimate_factor_model
from src.analytics.correlation.network import build_mst
from src.viz.plots import (
    plot_performance, plot_factor_exposures, plot_mst_network,
)


def _make_rs(R, tickers):
    T = R.shape[0]
    dates = pl.date_range(pl.date(2020, 1, 1),
                          pl.date(2020, 1, 1) + pl.duration(days=T - 1),
                          interval="1d", eager=True)
    return ReturnSeries(pl.DataFrame({"date": dates, **{t: R[:, i] for i, t in enumerate(tickers)}}), tickers)


@pytest.fixture
def rs5():
    np.random.seed(3)
    f = np.random.normal(0, 0.01, 800)
    R = np.column_stack([0.9 * f + np.random.normal(0, 0.004, 800) for _ in range(5)])
    return _make_rs(R, ["A", "B", "C", "D", "E"])


class TestVizFigures:
    def test_performance_returns_figure(self, rs5):
        r = rs5.to_numpy_series("A")
        fig = plot_performance(r, rs5.dates.to_list())
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2   # equity + drawdown

    def test_mst_network_returns_figure(self, rs5):
        mst = build_mst(rs5)
        fig = plot_mst_network(mst)
        assert isinstance(fig, go.Figure)
        # one trace per edge + one node trace
        assert len(fig.data) == len(mst.edges) + 1

    def test_factor_exposures_figure(self):
        np.random.seed(8)
        T = 600
        F = np.random.normal(0, 0.01, size=(T, 3))
        y = F @ np.array([1.0, -0.3, 0.5]) + np.random.normal(0, 0.003, T)
        from src.analytics.factors.prepare import AlignedFactorData
        aligned = AlignedFactorData(list(range(T)), y, F, ["Mkt-RF", "SMB", "HML"], np.zeros(T))
        model = estimate_factor_model(aligned)
        fig = plot_factor_exposures(model)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1   # single bar trace
        assert len(fig.data[0].y) == 3


class TestAlignmentWarning:
    def test_warns_on_dropped_dates(self, caplog):
        # Returns span 10 days, factors only cover 6 → 4 dropped
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 10),
                              interval="1d", eager=True)
        returns = pl.Series("ret", np.random.normal(0, 0.01, len(dates)))
        fdates = dates[:6]
        factor_wide = pl.DataFrame({
            "date": fdates,
            "Mkt-RF": np.random.normal(0, 0.01, len(fdates)),
            "RF": np.full(len(fdates), 0.0001),
        })
        from loguru import logger
        import io
        sink = io.StringIO()
        handler_id = logger.add(sink, level="WARNING")
        align_factors(returns, dates, factor_wide, ["Mkt-RF"])
        logger.remove(handler_id)
        out = sink.getvalue()
        assert "dropped" in out
        assert "NOT in the factor model" in out

    def test_coverage_property(self):
        dates = pl.date_range(pl.date(2023, 1, 1), pl.date(2023, 1, 5),
                              interval="1d", eager=True)
        returns = pl.Series("ret", [0.01] * len(dates))
        factor_wide = pl.DataFrame({
            "date": dates,
            "Mkt-RF": [0.005] * len(dates),
            "RF": [0.001] * len(dates),
        })
        aligned = align_factors(returns, dates, factor_wide, ["Mkt-RF"])
        assert "obs" in aligned.coverage
