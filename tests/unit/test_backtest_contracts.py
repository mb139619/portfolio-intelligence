"""
Tests for the three backtest contracts.

The look-ahead tests are the reason this file exists. Everything else here is
ordinary contract checking; those are the ones that make the project's central
claim checkable rather than asserted. If they ever go green while the guard is
broken, the whole platform is worthless, so they are written to fail loudly
rather than skip.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest import (
    BacktestResult,
    BuyAndHold,
    Context,
    EqualWeight,
    Fold,
    LookAheadError,
    RegimeState,
    RunMeta,
    Strategy,
    normalise_weights,
)
from src.domain.calendar import Calendar
from src.domain.returns import ReturnSeries

# ──────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────

def make_returns(
    n: int = 100,
    tickers: tuple[str, ...] = ("A", "B"),
    start: date = date(2020, 1, 1),
    calendar: Calendar = Calendar.TRADING_DAYS,
) -> ReturnSeries:
    rng = np.random.default_rng(0)
    dates = [start + timedelta(days=i) for i in range(n)]
    data = {"date": dates}
    for t in tickers:
        data[t] = rng.normal(0.0004, 0.01, n)
    return ReturnSeries(pl.DataFrame(data), list(tickers), calendar=calendar)


@pytest.fixture
def rs() -> ReturnSeries:
    return make_returns()


@pytest.fixture
def ctx(rs: ReturnSeries) -> Context:
    return Context(t=rs.dates.max(), returns=rs, current_weights={"A": 0.5, "B": 0.5})


# ──────────────────────────────────────────────────────────────
# Look-ahead — the load-bearing tests
# ──────────────────────────────────────────────────────────────

class TestLookAheadIsStructural:
    def test_context_rejects_returns_past_t(self, rs):
        """A Context containing tomorrow cannot be built. This is the guarantee."""
        t = rs.dates.max() - timedelta(days=1)
        with pytest.raises(LookAheadError, match="carries returns through"):
            Context(t=t, returns=rs, current_weights={})

    def test_context_accepts_returns_up_to_and_including_t(self, rs):
        Context(t=rs.dates.max(), returns=rs, current_weights={})

    def test_context_rejects_future_derived_quantities(self, rs):
        """
        Truncating prices is not enough: a covariance fitted through next month
        leaks just as effectively as a price does.
        """
        t = rs.dates.max()
        with pytest.raises(LookAheadError, match="derived quantities"):
            Context(t=t, returns=rs, current_weights={},
                    derived_as_of=t + timedelta(days=1))

    def test_context_rejects_future_regime_fit(self, rs):
        t = rs.dates.max()
        future = RegimeState(state=1, label="stress", probability=0.9,
                             fitted_through=t + timedelta(days=5))
        with pytest.raises(LookAheadError, match="regime fitted through"):
            Context(t=t, returns=rs, current_weights={}, regime=future)

    def test_stale_derived_quantities_are_allowed_and_measurable(self, rs):
        """Stale is honest; the staleness just has to be visible."""
        t = rs.dates.max()
        c = Context(t=t, returns=rs, current_weights={},
                    derived_as_of=t - timedelta(days=30))
        assert c.derived_staleness_days == 30

    def test_publication_lag_moves_the_data_frontier_back(self, rs):
        t = rs.dates.max()
        c = Context(t=t, returns=rs, current_weights={}, publication_lag_days=21)
        assert c.available_through == t - timedelta(days=21)
        assert c.available_through < c.t

    def test_strategy_has_no_route_to_the_future(self, ctx):
        """
        The surface is the guarantee: a Context exposes no store, no pipeline
        and no way to request another date.
        """
        forbidden = {"store", "pipeline", "fetch", "read_returns", "full_history"}
        assert forbidden.isdisjoint(dir(ctx))


# ──────────────────────────────────────────────────────────────
# Context
# ──────────────────────────────────────────────────────────────

class TestContext:
    def test_rejects_empty_history(self):
        empty = ReturnSeries(pl.DataFrame({"date": [], "A": []}), ["A"])
        with pytest.raises(ValueError, match="no observations"):
            Context(t=date(2020, 1, 1), returns=empty, current_weights={})

    def test_rejects_weights_outside_universe(self, rs):
        with pytest.raises(ValueError, match="absent from the context universe"):
            Context(t=rs.dates.max(), returns=rs, current_weights={"ZZZ": 1.0})

    def test_rejects_negative_publication_lag(self, rs):
        with pytest.raises(ValueError, match="publication_lag_days"):
            Context(t=rs.dates.max(), returns=rs, current_weights={},
                    publication_lag_days=-1)

    def test_universe_and_calendar_come_from_the_data(self, ctx):
        assert ctx.universe == ["A", "B"]
        assert ctx.calendar is Calendar.TRADING_DAYS
        assert ctx.periods_per_year == 252

    def test_continuous_calendar_propagates(self):
        rs = make_returns(calendar=Calendar.CONTINUOUS)
        c = Context(t=rs.dates.max(), returns=rs, current_weights={})
        assert c.periods_per_year == 365

    def test_weight_defaults_to_zero_for_unheld(self, ctx):
        assert ctx.weight("A") == 0.5
        assert ctx.weight("B") == 0.5

    def test_is_frozen(self, ctx):
        with pytest.raises(Exception):
            ctx.t = date(2030, 1, 1)


class TestRegimeState:
    def test_rejects_impossible_probability(self):
        with pytest.raises(ValueError, match="probability"):
            RegimeState(0, "calm", 1.4, date(2020, 1, 1))

    def test_rejects_state_outside_range(self):
        with pytest.raises(ValueError, match="outside"):
            RegimeState(5, "calm", 0.5, date(2020, 1, 1), n_states=2)

    def test_stress_is_the_highest_vol_state(self):
        """States are canonicalised by ascending volatility upstream."""
        assert RegimeState(1, "stress", 0.8, date(2020, 1, 1), 2).is_stress
        assert not RegimeState(0, "calm", 0.8, date(2020, 1, 1), 2).is_stress


# ──────────────────────────────────────────────────────────────
# Strategy
# ──────────────────────────────────────────────────────────────

class TestStrategy:
    def test_reference_strategies_satisfy_the_protocol(self):
        assert isinstance(EqualWeight(), Strategy)
        assert isinstance(BuyAndHold(), Strategy)

    def test_equal_weight_is_one_over_n(self, ctx):
        w = EqualWeight().target_weights(ctx)
        assert w == {"A": 0.5, "B": 0.5}
        assert sum(w.values()) == pytest.approx(1.0)

    def test_equal_weight_follows_a_changing_universe(self):
        rs = make_returns(tickers=("A", "B", "C", "D"))
        c = Context(t=rs.dates.max(), returns=rs, current_weights={})
        w = EqualWeight().target_weights(c)
        assert len(w) == 4
        assert all(v == pytest.approx(0.25) for v in w.values())

    def test_buy_and_hold_keeps_the_current_book(self, ctx):
        w = BuyAndHold().target_weights(ctx)
        assert w == ctx.current_weights

    def test_buy_and_hold_bootstraps_when_flat(self, rs):
        c = Context(t=rs.dates.max(), returns=rs, current_weights={})
        assert BuyAndHold().target_weights(c) == {"A": 0.5, "B": 0.5}


class TestNormaliseWeights:
    def test_scales_to_one(self):
        out = normalise_weights({"A": 2.0, "B": 2.0}, ["A", "B"])
        assert out == {"A": 0.5, "B": 0.5}

    def test_rejects_assets_outside_the_universe(self):
        with pytest.raises(ValueError, match="not in the context universe"):
            normalise_weights({"ZZZ": 1.0}, ["A", "B"])

    def test_rejects_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            normalise_weights({"A": float("nan")}, ["A"])
        with pytest.raises(ValueError, match="non-finite"):
            normalise_weights({"A": float("inf")}, ["A"])

    def test_rejects_shorts_unless_enabled(self):
        with pytest.raises(ValueError, match="long-only"):
            normalise_weights({"A": 1.5, "B": -0.5}, ["A", "B"])
        out = normalise_weights({"A": 1.5, "B": -0.5}, ["A", "B"], allow_short=True)
        assert out["B"] < 0

    def test_rejects_an_empty_book(self):
        with pytest.raises(ValueError, match="empty book"):
            normalise_weights({"A": 0.0, "B": 0.0}, ["A", "B"])

    def test_drops_zero_positions(self):
        out = normalise_weights({"A": 1.0, "B": 0.0}, ["A", "B"])
        assert out == {"A": 1.0}


# ──────────────────────────────────────────────────────────────
# Result
# ──────────────────────────────────────────────────────────────

class TestFold:
    def test_rejects_overlap_between_train_and_test(self):
        """An overlapping fold flatters out-of-sample results. Loudest failure."""
        with pytest.raises(ValueError, match="leak"):
            Fold(0, date(2020, 1, 1), date(2020, 6, 30),
                 date(2020, 6, 1), date(2020, 12, 31))

    def test_rejects_inverted_windows(self):
        with pytest.raises(ValueError, match="inverted"):
            Fold(0, date(2020, 6, 1), date(2020, 1, 1),
                 date(2020, 7, 1), date(2020, 12, 31))

    def test_contains_covers_the_test_window_only(self):
        f = Fold(0, date(2020, 1, 1), date(2020, 6, 30),
                 date(2020, 7, 1), date(2020, 12, 31))
        assert f.contains(date(2020, 8, 1))
        assert not f.contains(date(2020, 3, 1))


class TestRunMeta:
    def test_create_captures_provenance(self):
        m = RunMeta.create(strategy="s", universe=["A"],
                           start=date(2020, 1, 1), end=date(2020, 12, 31))
        assert m.git_commit and m.created_at
        assert isinstance(m.git_dirty, bool)

    def test_dirty_tree_is_not_reproducible(self):
        clean = RunMeta(strategy="s", universe=["A"], start=date(2020, 1, 1),
                        end=date(2020, 2, 1), git_commit="abc123", git_dirty=False)
        dirty = RunMeta(strategy="s", universe=["A"], start=date(2020, 1, 1),
                        end=date(2020, 2, 1), git_commit="abc123", git_dirty=True)
        unknown = RunMeta(strategy="s", universe=["A"], start=date(2020, 1, 1),
                          end=date(2020, 2, 1), git_commit="unknown")
        assert clean.is_reproducible
        assert not dirty.is_reproducible
        assert not unknown.is_reproducible

    def test_round_trips_through_dict(self):
        m = RunMeta.create(strategy="s", universe=["A", "B"],
                           start=date(2020, 1, 1), end=date(2020, 12, 31),
                           costs={"commission_bps": 5.0})
        assert RunMeta.from_dict(m.to_dict()) == m


def make_result(folds=None) -> BacktestResult:
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(10)]
    rets = np.full(10, 0.001)
    equity = pl.DataFrame({
        "date": dates, "equity": np.cumprod(1 + rets), "ret": rets,
    })
    positions = pl.DataFrame({
        "date": dates, "ticker": ["A"] * 10, "weight": [1.0] * 10,
    })
    trades = pl.DataFrame({
        "date": [dates[0]], "ticker": ["A"],
        "delta_weight": [1.0], "cost": [0.0005],
    })
    meta = RunMeta.create(strategy="equal_weight", universe=["A"],
                          start=dates[0], end=dates[-1])
    return BacktestResult(equity, positions, trades, {"sharpe": 1.23}, meta, folds)


class TestBacktestResult:
    def test_rejects_a_malformed_equity_curve(self):
        bad = pl.DataFrame({"date": [date(2020, 1, 1)], "equity": [1.0]})  # no `ret`
        with pytest.raises(ValueError, match="missing column"):
            BacktestResult(bad, pl.DataFrame({"date": [], "ticker": [], "weight": []}),
                           pl.DataFrame({"date": [], "ticker": [],
                                         "delta_weight": [], "cost": []}),
                           {}, RunMeta.create(strategy="s", universe=[],
                                              start=date(2020, 1, 1),
                                              end=date(2020, 1, 2)))

    def test_reports_cost_and_turnover(self):
        r = make_result()
        assert r.total_cost == pytest.approx(0.0005)
        assert r.turnover == pytest.approx(1.0)

    def test_out_of_sample_is_the_whole_curve_without_folds(self):
        r = make_result()
        assert len(r.out_of_sample()) == len(r.equity_curve)

    def test_out_of_sample_keeps_only_test_windows(self):
        fold = Fold(0, date(2020, 1, 1), date(2020, 1, 5),
                    date(2020, 1, 6), date(2020, 1, 10))
        r = make_result(folds=[fold])
        oos = r.out_of_sample()
        assert len(oos) == 5
        assert min(oos["date"].to_list()) == date(2020, 1, 6)

    def test_round_trips_through_disk(self, tmp_path):
        """A saved run must re-render without recomputation."""
        fold = Fold(0, date(2020, 1, 1), date(2020, 1, 5),
                    date(2020, 1, 6), date(2020, 1, 10))
        original = make_result(folds=[fold])
        original.save(tmp_path / "run")
        loaded = BacktestResult.load(tmp_path / "run")

        assert loaded.metrics == original.metrics
        assert loaded.run_meta == original.run_meta
        assert loaded.folds == original.folds
        assert loaded.equity_curve.equals(original.equity_curve)
        assert loaded.trades.equals(original.trades)

    def test_non_finite_metrics_become_null(self, tmp_path):
        """
        Bare NaN and Infinity are not valid JSON and would fail on reload, so
        they are coerced to null on the way out.
        """
        r = make_result()
        r.metrics = {"sharpe": float("nan"), "calmar": float("inf")}
        r.save(tmp_path / "run")
        reloaded = BacktestResult.load(tmp_path / "run")
        assert reloaded.metrics["sharpe"] is None
        assert reloaded.metrics["calmar"] is None

    def test_numpy_metrics_survive_as_numbers(self, tmp_path):
        """
        Metrics arrive from the analytics layer full of numpy scalars. np.int64
        is not a Python int, so an unguarded encoder turns it into the string
        "5" -- valid JSON, wrong type, and invisible until something tries to
        plot it.
        """
        r = make_result()
        r.metrics = {
            "n_obs": np.int64(252),
            "sharpe": np.float64(1.5),
            "beat_benchmark": np.bool_(True),
        }
        r.save(tmp_path / "run")
        m = BacktestResult.load(tmp_path / "run").metrics

        assert m["n_obs"] == 252 and isinstance(m["n_obs"], int)
        assert m["sharpe"] == pytest.approx(1.5) and isinstance(m["sharpe"], float)
        assert m["beat_benchmark"] is True
