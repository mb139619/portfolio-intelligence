"""
Walk-forward and tearsheet tests.

`TestSelectionIsBlind` is the one that matters. A walk-forward harness that
picks its winner using the test window produces a beautiful curve and means
nothing, and the failure is invisible in the output — the chart looks the same
either way. So the test builds a series where the right answer flips between
the two windows and asserts the harness picks the one that was right in
training, even though that is the worse choice.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.backtest import BacktestConfig, CostModel, EqualWeight, MinimumVariance
from src.backtest.result import BacktestResult
from src.backtest.walkforward import make_folds, walk_forward
from src.domain.returns import ReturnSeries
from src.export.tearsheet import SECTIONS, _assumptions, _warnings, build_payload


def make_returns(n=1200, tickers=("A", "B"), seed=0, start=date(2015, 1, 1)):
    rng = np.random.default_rng(seed)
    dates = [start + timedelta(days=i) for i in range(n)]
    data = {"date": dates}
    for t in tickers:
        data[t] = rng.normal(0.0004, 0.01, n)
    return ReturnSeries(pl.DataFrame(data), list(tickers))


def flipping_returns(n=1000, start=date(2015, 1, 1)):
    """
    A drifts up for the first half, B for the second.

    Whichever strategy wins the training window is therefore the wrong one for
    the test window, by construction. That is what makes the selection
    testable: a blind harness must pick the loser.
    """
    dates = [start + timedelta(days=i) for i in range(n)]
    half = n // 2
    a = [0.002] * half + [-0.001] * (n - half)
    b = [-0.001] * half + [0.002] * (n - half)
    return ReturnSeries(pl.DataFrame({"date": dates, "A": a, "B": b}), ["A", "B"])


class OnlyA:
    name = "only_a"

    def target_weights(self, ctx):
        return {"A": 1.0}


class OnlyB:
    name = "only_b"

    def target_weights(self, ctx):
        return {"B": 1.0}


# ──────────────────────────────────────────────────────────────
# The load-bearing test
# ──────────────────────────────────────────────────────────────

class TestSelectionIsBlind:
    def test_the_winner_is_chosen_on_training_data_only(self):
        rs = flipping_returns()
        cfg = BacktestConfig(rebalance="monthly", lookback=60, warmup=60,
                             costs=CostModel.free())
        result = walk_forward([OnlyA(), OnlyB()], rs, cfg,
                              n_folds=1, min_train=500)

        chosen = result.metrics["fold_selection"][0]["winner"]
        assert chosen == "only_a", (
            "the harness picked the strategy that wins the TEST window; the "
            "selection is seeing data it is judged on"
        )

    def test_and_therefore_loses_out_of_sample(self):
        """
        The consequence, stated as a test: choosing blind must be allowed to
        produce a bad result. A harness that never loses is cheating.
        """
        rs = flipping_returns()
        cfg = BacktestConfig(rebalance="monthly", lookback=60, warmup=60,
                             costs=CostModel.free())
        result = walk_forward([OnlyA(), OnlyB()], rs, cfg,
                              n_folds=1, min_train=500)
        assert result.metrics["final_equity"] < 1.0


# ──────────────────────────────────────────────────────────────
# Folds
# ──────────────────────────────────────────────────────────────

class TestMakeFolds:
    def test_test_windows_never_overlap_training(self):
        folds = make_folds(make_returns().dates.to_list(), n_folds=5)
        for f in folds:
            assert f.test_start > f.train_end

    def test_test_windows_are_contiguous_and_disjoint(self):
        """Gaps would silently drop periods; overlaps would score days twice."""
        folds = make_folds(make_returns().dates.to_list(), n_folds=5)
        for a, b in zip(folds, folds[1:]):
            assert a.test_end < b.test_start

    def test_expanding_training_always_starts_at_the_beginning(self):
        dates = make_returns().dates.to_list()
        folds = make_folds(dates, n_folds=4, scheme="expanding")
        assert all(f.train_start == dates[0] for f in folds)

    def test_rolling_training_moves_forward(self):
        dates = make_returns().dates.to_list()
        folds = make_folds(dates, n_folds=4, scheme="rolling", train_window=252)
        starts = [f.train_start for f in folds]
        assert starts == sorted(starts) and starts[0] < starts[-1]

    def test_rejects_an_unknown_scheme(self):
        with pytest.raises(ValueError, match="expanding.*rolling"):
            make_folds(make_returns().dates.to_list(), scheme="sideways")

    def test_rejects_a_window_too_short_to_split(self):
        with pytest.raises(ValueError, match="cannot support"):
            make_folds(make_returns(n=200).dates.to_list(), n_folds=5, min_train=252)


# ──────────────────────────────────────────────────────────────
# The stitched result
# ──────────────────────────────────────────────────────────────

class TestWalkForwardResult:
    @pytest.fixture(scope="class")
    def result(self):
        return walk_forward(
            [EqualWeight(), MinimumVariance()], make_returns(n=1500, seed=4),
            BacktestConfig(rebalance="monthly", lookback=120, costs=CostModel()),
            n_folds=4, min_train=300,
        )

    def test_carries_its_folds(self, result):
        assert result.folds and len(result.folds) == result.metrics["n_folds"]

    def test_the_curve_is_only_test_segments(self, result):
        """`out_of_sample()` should be the whole curve, because that is all it is."""
        assert len(result.out_of_sample()) == len(result.equity_curve)

    def test_records_the_choice_made_in_each_fold(self, result):
        for sel in result.metrics["fold_selection"]:
            assert sel["winner"] in {"equal_weight", "min_variance"}
            assert set(sel["candidates"]) == {"equal_weight", "min_variance"}

    def test_reports_selection_stability(self, result):
        assert 0.0 < result.metrics["selection_stability"] <= 1.0

    def test_round_trips_with_its_folds(self, result, tmp_path):
        result.save(tmp_path / "wf")
        again = BacktestResult.load(tmp_path / "wf")
        assert again.folds == result.folds
        assert again.metrics["fold_selection"] == result.metrics["fold_selection"]

    def test_needs_at_least_one_candidate(self):
        with pytest.raises(ValueError, match="at least one candidate"):
            walk_forward([], make_returns())


# ──────────────────────────────────────────────────────────────
# Tearsheet
# ──────────────────────────────────────────────────────────────

class TestTearsheet:
    @pytest.fixture(scope="class")
    def result(self):
        return walk_forward(
            [EqualWeight()], make_returns(n=1200, seed=8),
            BacktestConfig(rebalance="monthly", lookback=120, costs=CostModel()),
            n_folds=3, min_train=300,
        )

    def test_every_section_builds(self, result):
        payload = build_payload(result)
        failed = [s["id"] for s in payload["sections"] if s["error"]]
        assert not failed, f"sections failed: {failed}"

    def test_the_payload_matches_the_schema_the_spa_renders(self, result):
        payload = build_payload(result)
        assert set(payload) == {"meta", "sections"}
        assert len(payload["sections"]) == len(SECTIONS)
        for s in payload["sections"]:
            assert set(s) >= {"id", "title", "stats", "figures", "tables", "notes"}

    def test_it_serialises_without_non_finite_values(self, result):
        from src.export.encode import dumps
        dumps(build_payload(result))          # allow_nan=False under the hood

    def test_assumptions_and_warnings_do_not_repeat_each_other(self, result):
        """
        Emitting the same text in the banner and in the notes trains the reader
        to ignore the banner.
        """
        assert not set(_assumptions(result)) & set(_warnings(result))

    def test_an_in_sample_run_is_warned_about(self):
        from src.backtest import run

        plain = run(EqualWeight(), make_returns(n=800),
                    BacktestConfig(rebalance="monthly", lookback=120))
        assert any("IN SAMPLE" in w for w in _warnings(plain))

    def test_a_run_without_folds_says_so_rather_than_pretending(self):
        from src.backtest import run

        plain = run(EqualWeight(), make_returns(n=800),
                    BacktestConfig(rebalance="monthly", lookback=120))
        payload = build_payload(plain)
        wf = next(s for s in payload["sections"] if s["id"] == "walkforward")
        assert "no folds" in wf["error"].lower()
