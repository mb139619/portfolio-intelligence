"""
Constraints, risk parity and HRP.

Two tests here are doing real work rather than checking plumbing.
`test_contributions_really_are_equal` asserts the property that *defines* risk
parity — without it the function is just another allocator with a confident
name. And `TestHRPIsRobustToConditioning` shows the thing HRP is actually for:
it produces a sane allocation on a covariance matrix that makes minimum
variance take enormous offsetting positions, because it never inverts one.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analytics.optimization import (
    LONG_SHORT,
    Constraints,
    efficient_frontier,
    hrp,
    min_variance,
    risk_contributions,
    risk_parity,
)
from src.analytics.optimization.hrp import recursive_bisection


def cov_matrix(vols, corr):
    d = np.diag(vols)
    return d @ corr @ d


@pytest.fixture
def sigma():
    """Three assets: two correlated equities and a diversifying bond."""
    corr = np.array([[1.0, 0.85, -0.20],
                     [0.85, 1.0, -0.15],
                     [-0.20, -0.15, 1.0]])
    return cov_matrix(np.array([0.18, 0.22, 0.07]), corr)


@pytest.fixture
def sigma_wide():
    """A deliberately wide volatility spread, like an equity/crypto book."""
    corr = np.array([[1.0, 0.7, 0.1, 0.2],
                     [0.7, 1.0, 0.05, 0.25],
                     [0.1, 0.05, 1.0, -0.1],
                     [0.2, 0.25, -0.1, 1.0]])
    return cov_matrix(np.array([0.16, 0.24, 0.08, 0.80]), corr)


# ──────────────────────────────────────────────────────────────
# Constraints
# ──────────────────────────────────────────────────────────────

class TestConstraints:
    def test_the_default_is_long_only_and_fully_invested(self):
        c = Constraints()
        assert c.is_default and c.long_only and c.budget == 1.0

    def test_refuses_a_cap_that_cannot_reach_the_budget(self):
        with pytest.raises(ValueError, match="infeasible"):
            Constraints(max_weight=0.2).validate(3)      # 0.2 × 3 < 1

    def test_refuses_a_floor_that_overshoots_the_budget(self):
        with pytest.raises(ValueError, match="infeasible"):
            Constraints(min_weight=0.4).validate(3)      # 0.4 × 3 > 1

    def test_refuses_a_negative_floor_while_long_only(self):
        with pytest.raises(ValueError, match="long_only"):
            Constraints(min_weight=-0.1)

    def test_refuses_inverted_bounds(self):
        with pytest.raises(ValueError, match="exceeds max_weight"):
            Constraints(min_weight=0.5, max_weight=0.2)

    def test_bounds_open_up_when_shorting_is_allowed(self):
        assert Constraints(long_only=False).bounds(3) == [(None, None)] * 3
        assert Constraints().bounds(3) == [(0.0, 1.0)] * 3

    def test_the_starting_point_is_feasible(self):
        c = Constraints(max_weight=0.4)
        w = c.start(4)
        assert w.sum() == pytest.approx(1.0)
        assert (w <= 0.4 + 1e-12).all()

    def test_one_mandate_serves_every_optimiser(self, sigma):
        """The whole reason this type exists."""
        c = Constraints(max_weight=0.5)
        for result in (min_variance(sigma, c), risk_parity(sigma, c), hrp(sigma, c)):
            assert (result.weights <= 0.5 + 1e-6).all(), result.method
            assert result.weights.sum() == pytest.approx(1.0)
            assert result.constraints is c


# ──────────────────────────────────────────────────────────────
# Risk parity
# ──────────────────────────────────────────────────────────────

class TestRiskParity:
    def test_contributions_really_are_equal(self, sigma):
        """The defining property. Without it this is just another allocator."""
        r = risk_parity(sigma)
        rc = risk_contributions(r.weights, sigma)
        shares = rc / rc.sum()
        np.testing.assert_allclose(shares, 1.0 / len(shares), atol=1e-6)

    def test_it_converges_on_a_wide_volatility_spread(self, sigma_wide):
        """
        The case that broke the first implementation: starting from equal
        weight, SLSQP stalled and reported success. Inverse volatility is the
        exact ERC solution under equal correlations, so it is the right
        starting point, not a nudge.
        """
        r = risk_parity(sigma_wide)
        assert r.success, r.message
        rc = risk_contributions(r.weights, sigma_wide)
        assert np.std(rc / rc.sum()) < 1e-6

    def test_the_quietest_asset_gets_the_largest_weight(self, sigma):
        r = risk_parity(sigma)
        assert int(np.argmax(r.weights)) == int(np.argmin(np.sqrt(np.diag(sigma))))

    def test_it_holds_everything(self, sigma):
        """Unlike minimum variance, which will drop assets outright."""
        assert (risk_parity(sigma).weights > 1e-4).all()

    def test_risk_budgets_are_honoured(self, sigma):
        budgets = np.array([0.5, 0.25, 0.25])
        r = risk_parity(sigma, budgets=budgets)
        rc = risk_contributions(r.weights, sigma)
        np.testing.assert_allclose(rc / rc.sum(), budgets, atol=1e-5)

    def test_rejects_non_positive_budgets(self, sigma):
        with pytest.raises(ValueError, match="strictly positive"):
            risk_parity(sigma, budgets=np.array([0.5, 0.0, 0.5]))

    def test_refuses_to_pretend_shorts_make_sense(self, sigma):
        with pytest.raises(ValueError, match="long-only"):
            risk_parity(sigma, LONG_SHORT)

    def test_it_is_more_diversified_than_minimum_variance(self, sigma_wide):
        rp = risk_parity(sigma_wide)
        mv = min_variance(sigma_wide)
        assert rp.effective_n_positions > mv.effective_n_positions
        # ...and pays for it in volatility, which is the trade being made.
        assert rp.expected_volatility >= mv.expected_volatility


# ──────────────────────────────────────────────────────────────
# HRP
# ──────────────────────────────────────────────────────────────

class TestHRP:
    def test_weights_are_a_long_only_allocation(self, sigma):
        r = hrp(sigma)
        assert r.weights.sum() == pytest.approx(1.0)
        assert (r.weights >= 0).all()

    def test_it_reuses_the_shared_clustering(self, sigma):
        """
        The order is recorded in the message, which is how we know it came from
        `cluster_from_correlation` rather than a second linkage built here.
        """
        assert "order=" in hrp(sigma).message

    def test_the_ordering_drives_the_allocation(self, sigma_wide):
        """
        The tree is part of the method, so the order it produces changes the
        answer. Tested on the bisection directly rather than by comparing two
        linkage methods: those often agree, because a clean correlation
        structure yields the same leaf order either way, and a test that
        happens to pass on one fixture and not another is worse than none.
        """
        # The permutation has to change the GROUPING, not just the direction:
        # reversing an order is a symmetry of the bisection, because
        # {0,1}|{2,3} reversed is {3,2}|{1,0} — the same two clusters, swapped.
        a = recursive_bisection(sigma_wide, [0, 1, 2, 3])   # {0,1} vs {2,3}
        b = recursive_bisection(sigma_wide, [0, 2, 1, 3])   # {0,2} vs {1,3}
        assert not np.allclose(a, b, atol=1e-8)
        assert a.sum() == pytest.approx(1.0) and b.sum() == pytest.approx(1.0)

    def test_reversing_the_order_is_a_symmetry(self, sigma_wide):
        """Worth pinning: it is the reason the test above uses a real shuffle."""
        a = recursive_bisection(sigma_wide, [0, 1, 2, 3])
        b = recursive_bisection(sigma_wide, [3, 2, 1, 0])
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_both_linkage_methods_produce_a_valid_allocation(self, sigma_wide):
        for method in ("single", "ward", "average"):
            w = hrp(sigma_wide, linkage_method=method).weights
            assert w.sum() == pytest.approx(1.0) and (w >= 0).all()

    def test_caps_are_respected(self, sigma_wide):
        r = hrp(sigma_wide, Constraints(max_weight=0.35))
        assert (r.weights <= 0.35 + 1e-9).all()
        assert r.weights.sum() == pytest.approx(1.0)

    def test_refuses_shorts(self, sigma):
        with pytest.raises(ValueError, match="long-only"):
            hrp(sigma, LONG_SHORT)

    def test_a_single_asset_takes_everything(self):
        assert hrp(np.array([[0.04]])).weights == pytest.approx([1.0])

    def test_bisection_splits_capital_inversely_to_variance(self):
        """Two uncorrelated assets: the quieter one must get more."""
        Sigma = np.diag([0.04, 0.01])          # vols 20% and 10%
        w = recursive_bisection(Sigma, [0, 1])
        assert w[1] > w[0]
        assert w.sum() == pytest.approx(1.0)


class TestHRPIsRobustToConditioning:
    """
    The actual argument for HRP. Mean-variance and its relatives invert Σ, and
    inversion is where estimation error becomes leverage; HRP never inverts
    anything, so it degrades gracefully where they do not.
    """

    @staticmethod
    def near_singular(n=6, rho=0.999):
        """
        Near-perfect correlation AND a spread of volatilities. Both are
        required: with identical volatilities the problem is symmetric and the
        GMV solution is exactly equal weight however ill-conditioned Σ is, so a
        symmetric fixture proves nothing.
        """
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        return cov_matrix(np.linspace(0.10, 0.30, n), corr)

    def test_the_fixture_really_is_ill_conditioned(self):
        assert np.linalg.cond(self.near_singular()) > 1e4

    def test_minimum_variance_takes_offsetting_positions(self):
        ls = min_variance(self.near_singular(), LONG_SHORT)
        assert ls.gross_exposure > 2.0, (
            "the fixture is not ill-conditioned enough to make the point"
        )
        assert ls.weights.min() < 0        # genuinely short somewhere

    def test_hrp_stays_sane_on_the_same_matrix(self):
        Sigma = self.near_singular()
        r = hrp(Sigma)
        assert r.gross_exposure == pytest.approx(1.0)
        assert (r.weights >= 0).all()
        assert not np.isnan(r.weights).any()

    def test_risk_parity_also_stays_long_only_there(self):
        r = risk_parity(self.near_singular())
        assert r.gross_exposure == pytest.approx(1.0)


# ──────────────────────────────────────────────────────────────
# The refactored optimisers still behave
# ──────────────────────────────────────────────────────────────

class TestRefactoredOptimisers:
    def test_minimum_variance_takes_the_shared_constraints(self, sigma):
        capped = min_variance(sigma, Constraints(max_weight=0.5))
        assert (capped.weights <= 0.5 + 1e-6).all()

    def test_the_frontier_takes_them_too(self, sigma):
        mu = np.array([0.08, 0.10, 0.03])
        ef = efficient_frontier(sigma, mu, Constraints(max_weight=0.6), n_points=12)
        assert len(ef) > 1
        assert (ef.weights <= 0.6 + 1e-6).all()

    def test_the_result_records_the_mandate_it_was_given(self, sigma):
        c = Constraints(max_weight=0.5)
        assert min_variance(sigma, c).constraints is c
        assert "w ≤ 50.0%" in c.describe()
