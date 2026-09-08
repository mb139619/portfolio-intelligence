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
from src.backtest.result import BacktestResult, Fold, RunMeta, current_git_commit
from src.backtest.strategy import BuyAndHold, EqualWeight, Strategy, normalise_weights

__all__ = [
    "BacktestResult",
    "BuyAndHold",
    "Context",
    "EqualWeight",
    "Fold",
    "LookAheadError",
    "RegimeState",
    "RunMeta",
    "Strategy",
    "current_git_commit",
    "normalise_weights",
]
