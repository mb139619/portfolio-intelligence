"""
Backtesting engine.

Public surface is the three contracts. Define anything new against these
rather than reaching into engine internals:

    Context         what a strategy is allowed to know at time t
    Strategy        target weights in, nothing else
    BacktestResult  the serialisable output everything downstream consumes

The dependency arrow runs report -> backtest -> analytics -> store; nothing in
this package may be imported by the layers beneath it.
"""

from src.backtest.context import Context, LookAheadError, RegimeState
from src.backtest.engine import BacktestConfig, run, schedule
from src.backtest.execution import CostModel, drift_weights, rebalance
from src.backtest.result import BacktestResult, Fold, RunMeta, current_git_commit
from src.backtest.strategy import (
    BuyAndHold,
    EqualWeight,
    MinimumVariance,
    Strategy,
    normalise_weights,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BuyAndHold",
    "Context",
    "CostModel",
    "EqualWeight",
    "Fold",
    "LookAheadError",
    "MinimumVariance",
    "RegimeState",
    "RunMeta",
    "Strategy",
    "current_git_commit",
    "drift_weights",
    "normalise_weights",
    "rebalance",
    "run",
    "schedule",
]
