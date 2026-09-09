"""
Execution — turning target weights into trades, and trades into costs.

Kept apart from strategy logic so the same strategy can be re-run under
different cost assumptions without touching its code. That separation is the
point: "does this edge survive costs?" is a question you answer by varying one
side while holding the other fixed.

**There is no frictionless default.** `CostModel()` with no arguments still
charges. A zero-cost run has to be asked for explicitly, because a backtest
that silently assumes free trading flatters every high-turnover strategy, and
high turnover is precisely what naive strategies produce.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    """
    Linear transaction costs in basis points of traded notional.

    Defaults are deliberately unremarkable rather than optimistic: 5bp
    commission and 5bp slippage is a plausible retail-to-small-institution
    round number for liquid ETFs. They are an assumption, recorded in
    `RunMeta`, not a measurement — a real desk would model spread, market
    impact and borrow separately.

    Linear cost ignores market impact, which grows super-linearly with size.
    That understates the cost of a large book and is stated rather than
    hidden: this model is honest for research on liquid instruments and wrong
    for anything that moves a market.
    """

    commission_bps: float = 5.0
    slippage_bps: float = 5.0

    @property
    def total_bps(self) -> float:
        return self.commission_bps + self.slippage_bps

    def charge(self, traded_notional: float) -> float:
        """
        Cost as a fraction of portfolio value.

        `traded_notional` is Σ|Δw| — moving 60/40 to 50/50 sells 10% and buys
        10%, so 0.20 of the book changes hands and both legs pay.
        """
        return traded_notional * self.total_bps / 10_000.0

    def to_dict(self) -> dict:
        return {
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "model": "linear",
        }

    @classmethod
    def free(cls) -> CostModel:
        """Frictionless. Only for isolating the cost drag, never as a baseline."""
        return cls(commission_bps=0.0, slippage_bps=0.0)


@dataclass(frozen=True)
class Fill:
    ticker: str
    delta_weight: float
    cost: float


def drift_weights(
    weights: dict[str, float], returns: dict[str, float]
) -> dict[str, float]:
    """
    Carry weights forward through one period of returns.

    Holding an asset that outperforms leaves you with more of it: weights
    change without trading, which is the whole reason rebalancing costs
    anything. A backtest that resets to target every bar without charging for
    it is measuring a portfolio nobody could have held.
    """
    if not weights:
        return {}
    grown = {t: w * (1.0 + returns.get(t, 0.0)) for t, w in weights.items()}
    total = sum(grown.values())
    if abs(total) < 1e-12:
        return dict(weights)
    return {t: v / total for t, v in grown.items()}


def rebalance(
    current: dict[str, float],
    target: dict[str, float],
    costs: CostModel,
    min_trade: float = 1e-4,
) -> tuple[dict[str, float], list[Fill], float]:
    """
    Move from `current` to `target`, charging for the distance travelled.

    Returns (achieved weights, fills, total cost as a fraction of book value).

    `min_trade` suppresses trades below 1bp of the book. Without it, floating
    point noise in an optimiser's output generates thousands of microscopic
    fills whose costs accumulate into a real and entirely artificial drag.
    """
    universe = set(current) | set(target)
    fills: list[Fill] = []
    traded = 0.0

    for t in sorted(universe):
        delta = target.get(t, 0.0) - current.get(t, 0.0)
        if abs(delta) < min_trade:
            continue
        traded += abs(delta)
        fills.append(Fill(ticker=t, delta_weight=delta, cost=0.0))

    total_cost = costs.charge(traded)

    # Attribute the cost across fills in proportion to size, so the trades
    # table sums to the reported total rather than approximately.
    if fills and traded > 0:
        fills = [
            Fill(f.ticker, f.delta_weight, total_cost * abs(f.delta_weight) / traded)
            for f in fills
        ]

    achieved = dict(current)
    for f in fills:
        achieved[f.ticker] = achieved.get(f.ticker, 0.0) + f.delta_weight
    achieved = {t: w for t, w in achieved.items() if abs(w) > 1e-12}

    return achieved, fills, total_cost
