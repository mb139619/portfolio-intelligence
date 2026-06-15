"""
Tail risk tests — validated against known distributional properties.

Gaussian data: Cornish-Fisher ≈ Gaussian VaR, EVT shape ξ ≈ 0.
Student-t (fat tails): EVT shape ξ > 0, and modified VaR exceeds Gaussian VaR.
"""

import numpy as np
import pytest

from src.analytics.risk.tail import (
    cornish_fisher_quantile, cornish_fisher_var, gaussian_var,
    fit_evt_pot, tail_risk_comparison,
)


class TestCornishFisher:
    def test_reduces_to_gaussian_when_normal(self):
        # Zero skew/kurtosis → CF quantile equals the plain normal quantile
        from scipy import stats
        z = stats.norm.ppf(0.05)
        assert cornish_fisher_quantile(z, 0.0, 0.0) == pytest.approx(z)

    def test_cf_close_to_gaussian_on_normal_data(self):
        np.random.seed(0)
        r = np.random.normal(0, 0.01, 20000)
        cf = cornish_fisher_var(r, 0.99)
        g = gaussian_var(r, 0.99)
        assert cf == pytest.approx(g, rel=0.10)   # small sample skew/kurt only

    def test_cf_exceeds_gaussian_for_fat_left_tail(self):
        # Negatively-skewed, fat-tailed losses → CF VaR should exceed Gaussian
        np.random.seed(1)
        r = np.random.standard_t(df=3, size=50000) * 0.01
        r = r - 0.02 * (r < 0) * np.abs(r)        # add mild negative skew
        assert cornish_fisher_var(r, 0.99) > gaussian_var(r, 0.99)

    def test_var_positive_for_typical_returns(self):
        np.random.seed(2)
        r = np.random.normal(0.0003, 0.01, 5000)
        assert cornish_fisher_var(r, 0.95) > 0


class TestEVT:
    def test_shape_near_zero_for_gaussian(self):
        np.random.seed(3)
        r = np.random.normal(0, 0.01, 20000)
        evt = fit_evt_pot(r, confidence=0.99, threshold_quantile=0.95)
        # Gaussian tail → GPD shape ξ close to 0 (light tail)
        assert abs(evt.shape) < 0.2

    def test_shape_positive_for_fat_tails(self):
        np.random.seed(4)
        r = np.random.standard_t(df=3, size=30000) * 0.01
        evt = fit_evt_pot(r, confidence=0.99, threshold_quantile=0.95)
        # Student-t(3) is heavy-tailed → ξ > 0
        assert evt.shape > 0.05

    def test_cvar_geq_var(self):
        np.random.seed(5)
        r = np.random.standard_t(df=5, size=20000) * 0.01
        evt = fit_evt_pot(r, confidence=0.99)
        assert evt.cvar >= evt.var

    def test_evt_var_exceeds_threshold(self):
        np.random.seed(6)
        r = np.random.normal(0, 0.01, 20000)
        evt = fit_evt_pot(r, confidence=0.99, threshold_quantile=0.95)
        # 99% VaR must sit beyond the 95%-loss threshold it was built on
        assert evt.var > evt.threshold

    def test_too_few_exceedances_raises(self):
        r = np.random.normal(0, 0.01, 50)
        with pytest.raises(ValueError, match="exceedances"):
            fit_evt_pot(r, threshold_quantile=0.99)

    def test_exceedance_rate(self):
        np.random.seed(7)
        r = np.random.normal(0, 0.01, 10000)
        evt = fit_evt_pot(r, threshold_quantile=0.95)
        assert evt.exceedance_rate == pytest.approx(0.05, abs=0.01)


class TestComparison:
    def test_all_methods_present(self):
        np.random.seed(8)
        r = np.random.standard_t(df=4, size=20000) * 0.01
        out = tail_risk_comparison(r, confidence=0.99)
        for key in ("gaussian_var", "cornish_fisher_var", "historical_var",
                    "historical_cvar", "evt_var", "evt_cvar"):
            assert key in out

    def test_fat_tail_ordering(self):
        # For heavy tails, modelling the tail should not understate it:
        # historical CVaR >= historical VaR, and EVT VaR is finite & positive.
        np.random.seed(9)
        r = np.random.standard_t(df=3, size=40000) * 0.01
        out = tail_risk_comparison(r, confidence=0.99)
        assert out["historical_cvar"] >= out["historical_var"]
        assert out["evt_var"] is not None and out["evt_var"] > 0
