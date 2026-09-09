"""
Four custom strategies, written against nothing but the `Context` surface.

    python examples/custom_strategies.py

None of these import the store, the ingestion pipeline or anything else that
could reach past the decision date. That is the point: a strategy physically
cannot see the future here, so the interesting question stops being "is this
backtest honest?" and becomes "is this idea any good?".

Two of the four are deliberately unflattering. Momentum earns the highest
return in this window and pays for it with ten times the turnover of equal
weight; the drawdown-triggered de-risking rule produces a *worse* maximum
drawdown than doing nothing, because it sells low and is not back in for the
recovery. Both are the sort of result a backtester exists to produce.
"""

from __future__ import annotations

import numpy as np

from src.backtest import BacktestConfig, Context, CostModel, EqualWeight, run
from src.config import settings
from src.store.parquet_store import ParquetStore


class InverseVolatility:
    """
    Weight inversely to each asset's volatility — a poor man's risk parity.

    Reads the covariance the engine already estimated rather than computing
    one: `ctx.covariance` is a `CovarianceResult` from the risk engine, fitted
    on the lookback window ending at `ctx.t`.
    """

    name = "inverse_vol"

    def target_weights(self, ctx: Context) -> dict[str, float]:
        vols = ctx.covariance.volatilities            # annualised, sqrt of diagonal
        # The engine normalises, so returning unnormalised scores is fine.
        return {t: 1.0 / v for t, v in zip(ctx.universe, vols) if v > 0}


class Momentum:
    """
    Hold the top-N by trailing return, skipping the most recent month.

    The skip is the conventional 12-1 construction: the most recent month
    tends to reverse, so including it dilutes the signal. Note what this
    strategy costs — see the turnover column.
    """

    name = "momentum"

    def __init__(self, top_n: int = 2, skip: int = 21) -> None:
        self.top_n, self.skip = top_n, skip

    @property
    def params(self) -> dict:
        """Recorded in RunMeta, so two runs of this can be told apart."""
        return {"top_n": self.top_n, "skip": self.skip}

    def target_weights(self, ctx: Context) -> dict[str, float]:
        scores = {}
        for t in ctx.universe:
            r = ctx.returns.to_numpy_series(t)
            if len(r) <= self.skip:
                continue
            scores[t] = float(np.prod(1 + r[: -self.skip]) - 1)
        if not scores:                                 # too little history yet
            return dict.fromkeys(ctx.universe, 1.0)
        winners = sorted(scores, key=scores.get, reverse=True)[: self.top_n]
        return dict.fromkeys(winners, 1.0)


class DeriskOnDrawdown:
    """
    Move to a defensive asset when the book is more than 10% underwater.

    Kept here because it does not work. It reduces neither volatility nor the
    maximum drawdown — it makes the drawdown worse, by selling after the fall
    and being absent for the recovery. A rule that sounds prudent and is not
    is exactly what a harness is for.
    """

    name = "derisk_dd"

    def __init__(self, defensive: str = "TLT", threshold: float = -0.10) -> None:
        self.defensive, self.threshold = defensive, threshold

    @property
    def params(self) -> dict:
        return {"defensive": self.defensive, "threshold": self.threshold}

    def target_weights(self, ctx: Context) -> dict[str, float]:
        n = ctx.returns.n_assets
        port = ctx.returns.to_numpy() @ np.full(n, 1.0 / n)
        cum = np.cumprod(1 + port)
        drawdown = cum[-1] / cum.max() - 1
        if drawdown < self.threshold and self.defensive in ctx.universe:
            return {self.defensive: 1.0}
        return dict.fromkeys(ctx.universe, 1.0)


class RegimeAware:
    """
    Hold the defensive asset while the market is in the stress regime.

    Requires `BacktestConfig(with_regime=True)` and a `market` series. The
    state on `ctx.regime` is derived from FILTERED probabilities and from a
    model refitted only on data published by `ctx.t` — using the smoothed
    estimate here would hand the strategy the answer.
    """

    name = "regime_aware"

    def __init__(self, defensive: str = "TLT") -> None:
        self.defensive = defensive

    def target_weights(self, ctx: Context) -> dict[str, float]:
        if ctx.regime is not None and ctx.regime.is_stress:
            if self.defensive in ctx.universe:
                return {self.defensive: 1.0}
        return dict.fromkeys(ctx.universe, 1.0)


def main() -> None:
    store = ParquetStore(settings.data_dir)
    rs = store.read_returns(["SPY", "TLT", "QQQ", "GLD", "EFA"], start="2015-01-01")
    config = BacktestConfig(rebalance="monthly", lookback=252, costs=CostModel())

    print(f"\nUniverse  {', '.join(rs.tickers)}")
    print(f"Window    {rs.dates.min()} -> {rs.dates.max()}  ({rs.n_obs} obs)")
    print(f"Costs     {config.costs.total_bps:.0f}bp round trip, "
          f"{config.rebalance} rebalance\n")

    header = (f"{'strategy':<16}{'CAGR':>8}{'vol':>8}{'ret/vol':>9}"
              f"{'maxDD':>9}{'turnover':>10}{'costs':>8}")
    print(header)
    print("-" * len(header))

    for strategy in [EqualWeight(), InverseVolatility(), Momentum(),
                     DeriskOnDrawdown()]:
        m = run(strategy, rs, config).metrics
        print(f"{strategy.name:<16}{m['annualized_return']:>7.2%}"
              f"{m['annualized_volatility']:>8.2%}{m['sharpe_ratio']:>9.2f}"
              f"{m['max_drawdown']:>9.2%}{m['turnover']:>10.2f}"
              f"{m['total_cost']:>8.2%}")

    print("\nret/vol uses a zero risk-free rate.")


if __name__ == "__main__":
    main()
