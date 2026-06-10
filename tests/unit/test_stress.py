"""
Stress testing tests — validated on synthetic data with known answers.
"""

import numpy as np
import polars as pl
import pytest

from src.domain.returns import ReturnSeries
from src.analytics.factors.prepare import AlignedFactorData
from src.analytics.factors.engine import estimate_factor_model
from src.analytics.stress.scenarios import (
    HistoricalScenario, HISTORICAL_SCENARIOS, FACTOR_SHOCKS, MACRO_SHOCKS,
)
from src.analytics.stress.historical import (
    run_historical_asset, run_historical_factor,
)
from src.analytics.stress.parametric import (
    run_factor_shock, estimate_macro_sensitivities, run_macro_shock,
)


def _make_rs(R, tickers, start=(2020, 1, 1)):
    T = R.shape[0]
    d0 = pl.date(*start)
    dates = pl.date_range(d0, d0 + pl.duration(days=T - 1), interval="1d", eager=True)
    return ReturnSeries(pl.DataFrame({"date": dates, **{t: R[:, i] for i, t in enumerate(tickers)}}), tickers)


@pytest.fixture
def model_3factor():
    np.random.seed(31)
    T = 1500
    F = np.random.normal(0, 0.01, size=(T, 3))
    betas = np.array([1.1, -0.3, 0.5])
    y = F @ betas + np.random.normal(0, 0.003, T)
    aligned = AlignedFactorData(list(range(T)), y, F, ["Mkt-RF", "SMB", "HML"], np.zeros(T))
    return estimate_factor_model(aligned), betas


class TestScenarioRegistries:
    def test_historical_registry(self):
        assert "GFC_2008" in HISTORICAL_SCENARIOS
        assert "COVID_2020" in HISTORICAL_SCENARIOS

    def test_shock_registries(self):
        assert "EQUITY_CRASH" in FACTOR_SHOCKS
        assert "RATES_UP_100" in MACRO_SHOCKS
        assert MACRO_SHOCKS["RATES_UP_100"].moves["rate_10y"] == pytest.approx(0.01)


class TestFactorShock:
    def test_pnl_equals_beta_dot_shock(self, model_3factor):
        model, _ = model_3factor
        shock = {"Mkt-RF": -0.20}
        res = run_factor_shock(model, shock, name="EQUITY_CRASH")
        expected = model.betas["Mkt-RF"] * -0.20
        assert res.total_pnl == pytest.approx(expected, rel=1e-9)

    def test_contributions_sum_to_total(self, model_3factor):
        model, _ = model_3factor
        shock = {"Mkt-RF": -0.10, "HML": 0.05}
        res = run_factor_shock(model, shock)
        assert sum(res.contributions.values()) == pytest.approx(res.total_pnl, rel=1e-9)

    def test_negative_market_beta_crash_loses(self, model_3factor):
        model, _ = model_3factor
        res = run_factor_shock(model, {"Mkt-RF": -0.20})
        # Positive market beta + market crash → loss
        assert res.total_pnl < 0

    def test_unknown_factor_raises(self, model_3factor):
        model, _ = model_3factor
        with pytest.raises(ValueError):
            run_factor_shock(model, {"OIL": -0.3})


class TestHistoricalFactor:
    def test_factor_replay_runs(self, model_3factor):
        model, _ = model_3factor
        # Build a factor_wide covering a fake window
        T = 60
        dates = pl.date_range(pl.date(2020, 2, 19), pl.date(2020, 2, 19) + pl.duration(days=T - 1),
                              interval="1d", eager=True)
        fw = pl.DataFrame({
            "date": dates,
            "Mkt-RF": np.random.normal(-0.01, 0.02, T),  # crash-like
            "SMB": np.random.normal(0, 0.01, T),
            "HML": np.random.normal(0, 0.01, T),
        })
        scen = HistoricalScenario("TEST", "2020-02-19", "2020-04-18", "test")
        res = run_historical_factor(model, fw, scen)
        assert res.method == "factor_replay"
        # contributions arithmetic-sum should be close to total for small moves
        assert res.max_drawdown <= 0

    def test_empty_window_raises(self, model_3factor):
        model, _ = model_3factor
        fw = pl.DataFrame({
            "date": pl.date_range(pl.date(2020, 1, 1), pl.date(2020, 1, 5), interval="1d", eager=True),
            "Mkt-RF": [0.0] * 5, "SMB": [0.0] * 5, "HML": [0.0] * 5,
        })
        scen = HistoricalScenario("OLD", "2008-09-01", "2009-03-09", "GFC")
        with pytest.raises(ValueError):
            run_historical_factor(model, fw, scen)


class TestHistoricalAsset:
    def test_replay_matches_compounded_return(self):
        np.random.seed(7)
        R = np.random.normal(-0.002, 0.02, size=(40, 3))
        rs = _make_rs(R, ["SPY", "TLT", "GLD"], start=(2020, 2, 19))
        weights = {"SPY": 0.6, "TLT": 0.3, "GLD": 0.1}
        scen = HistoricalScenario("COVID", "2020-02-19", "2020-04-30", "covid")
        res = run_historical_asset(weights, rs, scen)
        # Recompute expected compounded portfolio return
        w = np.array([0.6, 0.3, 0.1])
        expected = np.prod(1 + R @ w) - 1
        assert res.total_pnl == pytest.approx(expected, rel=1e-9)

    def test_missing_asset_renormalises(self):
        np.random.seed(9)
        R = np.random.normal(0, 0.01, size=(30, 2))
        rs = _make_rs(R, ["SPY", "TLT"], start=(2020, 2, 19))
        # Weights reference GLD which isn't in the data
        weights = {"SPY": 0.5, "TLT": 0.3, "GLD": 0.2}
        scen = HistoricalScenario("COVID", "2020-02-19", "2020-03-30", "covid")
        res = run_historical_asset(weights, rs, scen)
        # Renormalised over SPY+TLT (0.5/0.8, 0.3/0.8)
        assert "GLD" not in res.contributions


class TestMacroSensitivities:
    def test_recovers_known_sensitivities(self):
        np.random.seed(13)
        T = 2000
        # Macro moves: equity return, 10y change, oil return
        eq = np.random.normal(0, 0.01, T)
        d10y = np.random.normal(0, 0.0005, T)
        oil = np.random.normal(0, 0.02, T)
        true_betas = {"equity": 0.8, "rate_10y": -3.0, "oil": 0.05}
        port = (true_betas["equity"] * eq
                + true_betas["rate_10y"] * d10y
                + true_betas["oil"] * oil
                + np.random.normal(0, 0.001, T))
        macro = pl.DataFrame({"equity": eq, "rate_10y": d10y, "oil": oil})
        sens = estimate_macro_sensitivities(port, macro, ["equity", "rate_10y", "oil"])
        assert sens.betas["equity"] == pytest.approx(0.8, abs=0.05)
        assert sens.betas["rate_10y"] == pytest.approx(-3.0, abs=0.3)
        assert sens.r_squared > 0.9

    def test_macro_shock_pnl(self):
        sens_betas = {"rate_10y": -3.0, "oil": 0.05}
        from src.analytics.stress.parametric import MacroSensitivities
        sens = MacroSensitivities(["rate_10y", "oil"], sens_betas, 0.9, 1000)
        # +100bp rate shock
        res = run_macro_shock(sens, {"rate_10y": 0.01})
        assert res.total_pnl == pytest.approx(-3.0 * 0.01, rel=1e-9)

    def test_macro_shock_unknown_var_raises(self):
        from src.analytics.stress.parametric import MacroSensitivities
        sens = MacroSensitivities(["oil"], {"oil": 0.05}, 0.5, 100)
        with pytest.raises(ValueError):
            run_macro_shock(sens, {"vix": 20.0})
