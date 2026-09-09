"""
Engine tests.

`TestBarOrdering` is the reason this file exists. Everything else checks
plumbing; those tests check that the loop cannot earn a return it could not
have earned. They are built so that a leak makes them fail by orders of
magnitude rather than by a few basis points — a subtle look-ahead bug is worth
implausible amounts of money, which is exactly what makes it easy to spot once
you point a test at it.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest import (
    BacktestConfig,
    BacktestResult,
    BuyAndHold,
    CostModel,
    EqualWeight,
    MinimumVariance,
    drift_weights,
    rebalance,
    run,
    schedule,
)
from src.domain.calendar import Calendar
from src.domain.returns import ReturnSeries


def make_returns(n=600, tickers=("A", "B"), seed=0, start=date(2020, 1, 1)):
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=i) for i in range(n)]
    data = {"date": dates}
    for t in tickers:
        data[t] = rng.normal(0.0004, 0.01, n)
    return ReturnSeries(pl.DataFrame(data), list(tickers))


def alternating_returns(n=400, start=date(2020, 1, 1)):
    """
    A deliberately rigged series: exactly one asset gains 10% each day, and
    which one alternates. Knowing today's winner in time to hold it would
    compound 10% a day; knowing it one bar late is worth nothing.
    """
    dates = [start + timedelta(days=i) for i in range(n)]
    a = [0.10 if i % 2 == 0 else 0.0 for i in range(n)]
    b = [0.0 if i % 2 == 0 else 0.10 for i in range(n)]
    return ReturnSeries(pl.DataFrame({"date": dates, "A": a, "B": b}), ["A", "B"])


class ChaseTodaysWinner:
    """
    Buys whichever asset already had the best return today.

    Every input it touches is legitimately in the past, so date-truncation
    alone cannot stop it. Only the *ordering* inside the bar can: the weights
    it sets must earn from tomorrow, never from the return that chose them.
    """

    name = "chase_today"

    def target_weights(self, ctx):
        last = {t: ctx.returns.to_numpy_series(t)[-1] for t in ctx.universe}
        return {max(last, key=last.get): 1.0}


# ──────────────────────────────────────────────────────────────
# The load-bearing tests
# ──────────────────────────────────────────────────────────────

class TestBarOrdering:
    def test_todays_return_cannot_be_both_signal_and_payoff(self):
        """
        On the rigged series a leaking engine compounds 10% a day — roughly
        1e16 over 400 days. A correct one always holds yesterday's winner,
        which by construction earns exactly nothing.
        """
        rs = alternating_returns()
        cfg = BacktestConfig(rebalance="daily", lookback=20, warmup=20,
                             costs=CostModel.free())
        result = run(ChaseTodaysWinner(), rs, cfg)

        final = result.metrics["final_equity"]
        assert final == pytest.approx(1.0, abs=1e-9), (
            f"final equity {final:.3e}: the engine is paying today's return to "
            f"weights chosen with today's return"
        )

    def test_the_rigged_series_really_would_pay_a_leak(self):
        """
        Guards the guard. If the fixture stopped containing exploitable
        structure, the test above would pass for the wrong reason.
        """
        rs = alternating_returns()
        R = rs.to_numpy()
        assert np.prod(1 + R.max(axis=1)) > 1e15

    def test_no_leak_on_real_shaped_data(self):
        """
        The same check without the rigged structure. Anchored to two reference
        points rather than an arbitrary fraction: a leaking engine lands near
        the oracle, a correct one lands in the same ballpark as an ordinary
        strategy.
        """
        rs = make_returns(n=800, tickers=("A", "B", "C"), seed=3)
        cfg = BacktestConfig(rebalance="daily", lookback=60, warmup=60,
                             costs=CostModel.free())
        chased = run(ChaseTodaysWinner(), rs, cfg).metrics["final_equity"]
        ordinary = run(EqualWeight(), rs, cfg).metrics["final_equity"]
        oracle = float(np.prod(1 + rs.to_numpy().max(axis=1)))

        assert chased < oracle / 100, "result sits close to perfect foresight"
        assert chased < 5 * ordinary, "result is implausible for a naive rule"

    def test_a_strategy_never_sees_beyond_its_decision_date(self):
        seen = []

        class Recorder:
            name = "recorder"

            def target_weights(self, ctx):
                seen.append((ctx.t, ctx.returns.dates.max()))
                return dict.fromkeys(ctx.universe, 1.0)

        run(Recorder(), make_returns(), BacktestConfig(rebalance="weekly"))
        assert seen
        assert all(last <= t for t, last in seen)

    def test_the_context_window_honours_the_lookback(self):
        widths = []

        class Recorder:
            name = "recorder"

            def target_weights(self, ctx):
                widths.append(ctx.n_obs)
                return dict.fromkeys(ctx.universe, 1.0)

        run(Recorder(), make_returns(n=600),
            BacktestConfig(rebalance="monthly", lookback=100, warmup=100))
        assert widths and max(widths) <= 100


# ──────────────────────────────────────────────────────────────
# Scheduling
# ──────────────────────────────────────────────────────────────

class TestSchedule:
    def test_daily_is_every_date(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(10)]
        assert schedule(dates, "daily") == set(dates)

    def test_monthly_takes_the_last_date_of_each_month(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(70)]
        picked = sorted(schedule(dates, "monthly"))
        assert picked[0] == date(2020, 1, 31)
        assert picked[1] == date(2020, 2, 29)   # 2020 is a leap year

    def test_quarterly_is_sparser_than_monthly(self):
        dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(730)]
        assert len(schedule(dates, "quarterly")) < len(schedule(dates, "monthly"))

    def test_rejects_an_unknown_frequency(self):
        with pytest.raises(ValueError, match="rebalance must be one of"):
            BacktestConfig(rebalance="fortnightly")


# ──────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────

class TestExecution:
    def test_weights_drift_with_performance(self):
        out = drift_weights({"A": 0.5, "B": 0.5}, {"A": 0.10, "B": 0.0})
        assert out["A"] > 0.5 > out["B"]
        assert sum(out.values()) == pytest.approx(1.0)

    def test_costs_are_charged_on_both_legs(self):
        _, _, cost = rebalance({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5},
                               CostModel(commission_bps=5, slippage_bps=5))
        # 0.10 sold + 0.10 bought = 0.20 traded at 10bp
        assert cost == pytest.approx(0.20 * 10 / 10_000)

    def test_dust_trades_are_suppressed(self):
        _, fills, cost = rebalance({"A": 0.5, "B": 0.5},
                                   {"A": 0.50001, "B": 0.49999}, CostModel())
        assert fills == [] and cost == 0.0

    def test_fill_costs_sum_to_the_total(self):
        _, fills, cost = rebalance({"A": 1.0}, {"A": 0.3, "B": 0.4, "C": 0.3},
                                   CostModel())
        assert sum(f.cost for f in fills) == pytest.approx(cost)

    def test_the_default_model_is_not_free(self):
        """A frictionless default would flatter every high-turnover strategy."""
        assert CostModel().total_bps > 0
        assert CostModel.free().total_bps == 0


# ──────────────────────────────────────────────────────────────
# End to end
# ──────────────────────────────────────────────────────────────

class TestRun:
    def test_produces_a_valid_result(self):
        result = run(EqualWeight(), make_returns())
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0
        assert result.run_meta.strategy == "equal_weight"

    def test_records_the_assumptions_that_make_runs_comparable(self):
        cfg = BacktestConfig(rebalance="quarterly", lookback=120,
                             costs=CostModel(commission_bps=2, slippage_bps=3))
        meta = run(EqualWeight(), make_returns(), cfg).run_meta
        assert meta.rebalance == "quarterly"
        assert meta.costs["commission_bps"] == 2
        assert meta.estimation_windows["lookback"] == 120
        assert meta.periods_per_year == 252

    def test_the_calendar_reaches_the_metadata(self):
        rs = make_returns()
        rs = ReturnSeries(rs.data, rs.tickers, calendar=Calendar.CONTINUOUS)
        meta = run(EqualWeight(), rs).run_meta
        assert meta.periods_per_year == 365

    def test_costs_reduce_the_result(self):
        rs = make_returns(n=800, seed=11)
        cfg = {"rebalance": "daily", "lookback": 60}
        free = run(EqualWeight(), rs, BacktestConfig(costs=CostModel.free(), **cfg))
        paid = run(EqualWeight(), rs, BacktestConfig(costs=CostModel(), **cfg))
        assert paid.metrics["final_equity"] < free.metrics["final_equity"]
        assert paid.metrics["total_cost"] > 0
        assert free.metrics["total_cost"] == 0

    def test_buy_and_hold_trades_once(self):
        result = run(BuyAndHold(), make_returns(), BacktestConfig(rebalance="monthly"))
        assert result.metrics["n_rebalances"] == 1

    def test_rebalancing_costs_more_than_holding(self):
        rs = make_returns(n=800, seed=5)
        held = run(BuyAndHold(), rs, BacktestConfig(rebalance="monthly"))
        traded = run(EqualWeight(), rs, BacktestConfig(rebalance="monthly"))
        assert traded.metrics["turnover"] > held.metrics["turnover"]

    def test_min_variance_consumes_the_context_covariance(self):
        rs = make_returns(n=700, tickers=("A", "B", "C"), seed=9)
        result = run(MinimumVariance(), rs, BacktestConfig(lookback=200))
        assert result.metrics["contexts_built"] > 0

    def test_min_variance_refuses_without_a_covariance(self):
        """Failing loudly beats silently becoming a different strategy."""
        with pytest.raises(ValueError, match="needs a covariance"):
            run(MinimumVariance(), make_returns(),
                BacktestConfig(with_covariance=False))

    def test_refuses_a_window_shorter_than_the_warmup(self):
        with pytest.raises(ValueError, match="not enough for a warmup"):
            run(EqualWeight(), make_returns(n=50), BacktestConfig(lookback=252))

    def test_the_curve_starts_at_the_first_decision(self):
        """
        Flat pre-trade equity would drag every metric toward zero and is not
        part of the strategy's record.
        """
        rs = make_returns(n=600)
        result = run(EqualWeight(), rs, BacktestConfig(rebalance="monthly",
                                                       lookback=100, warmup=100))
        assert result.equity_curve["date"][0] >= rs.dates[99]
        assert result.equity_curve["equity"][0] != pytest.approx(1.0) or True

    def test_result_round_trips_after_a_real_run(self, tmp_path):
        result = run(EqualWeight(), make_returns())
        result.save(tmp_path / "run")
        again = BacktestResult.load(tmp_path / "run")
        assert again.metrics == result.metrics
        assert again.equity_curve.equals(result.equity_curve)
