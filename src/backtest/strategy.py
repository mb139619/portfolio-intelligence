"""
Strategy — signal only.

A strategy answers one question: given what is knowable at `t`, what should the
book look like? It returns **target weights, never orders**. Turning weights
into trades and trades into fills is the execution layer's job, and keeping
that boundary sharp is what lets the same strategy be evaluated under different
cost assumptions without touching its logic.

Strategies are also stateless with respect to scheduling. The engine decides
*when* to ask; the strategy decides only *what*. A strategy that wants to
rebalance monthly does not implement a calendar — the engine is configured to
call it monthly. This keeps the rebalance schedule an experimental variable
rather than something buried in strategy code.

Portfolio optimisers wrap trivially: a risk-parity strategy is a call to the
optimiser on `ctx.covariance` and nothing else.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from src.backtest.context import Context


@runtime_checkable
class Strategy(Protocol):
    """The whole interface. Anything satisfying this can be backtested."""

    name: str

    def target_weights(self, ctx: Context) -> dict[str, float]:
        """
        Desired portfolio weights at `ctx.t`.

        May return a subset of the universe; absent assets are treated as zero.
        Need not sum to exactly 1 — the engine normalises — but must not
        contain NaN or infinity, which silently poison a portfolio return.
        """
        ...


def normalise_weights(
    raw: dict[str, float],
    universe: list[str],
    *,
    strategy_name: str = "strategy",
    allow_short: bool = False,
    gross_target: float = 1.0,
) -> dict[str, float]:
    """
    Validate and normalise what a strategy returned.

    Applied by the engine to every strategy output, so a badly behaved strategy
    fails loudly at its own boundary instead of producing a plausible-looking
    equity curve that is quietly wrong.

    **Normalisation is on GROSS exposure**, Σ|w|, scaled to `gross_target`.
    For a long-only book gross equals net, so this is exactly the old
    "scale to sum 1" and every long-only result is unchanged.

    Normalising on net exposure — the sum — is correct only for a fully
    invested long book and silently catastrophic otherwise: a 60/40 long/short
    pair sums to 0.2, so dividing by it turned a 1x book into a 5x levered one,
    and an exactly neutral book divided by zero. Gross is the invariant that
    means the same thing for every mandate.

    A gross_target of 1.0 means the capital at risk equals the capital. A
    strategy wanting a levered book has to say so.
    """
    unknown = [t for t in raw if t not in universe]
    if unknown:
        raise ValueError(
            f"{strategy_name} returned weights for {unknown}, which are not in "
            f"the context universe. A strategy may only trade what it can see."
        )

    bad = {t: w for t, w in raw.items() if not math.isfinite(w)}
    if bad:
        raise ValueError(f"{strategy_name} returned non-finite weights: {bad}")

    if not allow_short:
        negative = {t: w for t, w in raw.items() if w < 0}
        if negative:
            raise ValueError(
                f"{strategy_name} returned negative weights {negative} but the "
                f"run is long-only. Enable shorting explicitly if intended."
            )

    gross = sum(abs(w) for w in raw.values())
    if gross < 1e-12:
        raise ValueError(
            f"{strategy_name} returned an empty book (gross exposure "
            f"{gross:.2e}); there is nothing to normalise."
        )

    scale = gross_target / gross
    return {t: w * scale for t, w in raw.items() if w != 0.0}


class EqualWeight:
    """
    1/N over whatever is investable at `t`.

    The reference baseline, and deliberately the first strategy implemented.
    Equal weight is hard to beat after costs and it exercises every part of the
    harness — universe changes, rebalancing, turnover, costs — without any
    signal to hide behind. If the engine is wrong, it is most visible here.
    """

    name = "equal_weight"

    def target_weights(self, ctx: Context) -> dict[str, float]:
        universe = ctx.universe
        w = 1.0 / len(universe)
        return dict.fromkeys(universe, w)


class BuyAndHold:
    """
    Hold the initial allocation and never trade again.

    Useful as a control: the difference between this and a rebalanced strategy
    on the same universe is exactly the rebalancing premium net of its cost,
    which is a number worth seeing rather than assuming.
    """

    name = "buy_and_hold"

    def __init__(self, initial: dict[str, float] | None = None) -> None:
        self.initial = initial

    def target_weights(self, ctx: Context) -> dict[str, float]:
        if self.initial is not None:
            return {t: w for t, w in self.initial.items() if t in ctx.universe}
        # No target given: adopt whatever is currently held, and on the first
        # bar (nothing held yet) fall back to equal weight.
        if ctx.current_weights:
            return dict(ctx.current_weights)
        return dict.fromkeys(ctx.universe, 1.0 / len(ctx.universe))


class MinimumVariance:
    """
    Minimum-variance weights from the context's own covariance estimate.

    The point of this one is what it does *not* contain. There is no
    re-estimation, no matrix algebra, no calendar handling: the risk engine
    produced `ctx.covariance` and the optimiser consumes it unchanged. If
    wrapping an optimiser had required more than this, the boundary between
    the two layers would be in the wrong place.

    Needs `BacktestConfig.with_covariance` (the default). It fails loudly
    rather than silently falling back to equal weight, because a strategy that
    quietly becomes a different strategy is worse than one that stops.
    """

    name = "min_variance"

    def __init__(self, max_weight: float | None = None) -> None:
        self.max_weight = max_weight

    @property
    def params(self) -> dict:
        return {"max_weight": self.max_weight}

    def target_weights(self, ctx: Context) -> dict[str, float]:
        from src.analytics.optimization import Constraints, min_variance

        if ctx.covariance is None:
            raise ValueError(
                f"{self.name} needs a covariance estimate; run with "
                f"BacktestConfig(with_covariance=True)."
            )
        result = min_variance(
            ctx.covariance, Constraints(max_weight=self.max_weight)
        )
        if not result.success:
            # The optimiser returns its starting point on failure, which is
            # equal weight. Accepting that silently would turn this into a
            # different strategy on exactly the windows where the covariance
            # was worst -- the ones whose behaviour matters most.
            raise ValueError(
                f"{self.name}: minimum-variance solve failed at {ctx.t} "
                f"({result.message or 'no message'}). Check the covariance "
                f"estimate for that window."
            )
        return result.weights_dict()


class RiskParity:
    """
    Equal risk contribution from the context's covariance.

    Where `MinimumVariance` concentrates — it will happily drop assets to zero
    — this one holds everything and sizes each position so that all contribute
    the same amount of risk. Both consume `ctx.covariance` unchanged; the
    difference is entirely in the objective.
    """

    name = "risk_parity"

    def __init__(self, max_weight: float | None = None) -> None:
        self.max_weight = max_weight

    @property
    def params(self) -> dict:
        return {"max_weight": self.max_weight}

    def target_weights(self, ctx: Context) -> dict[str, float]:
        from src.analytics.optimization import Constraints, risk_parity

        if ctx.covariance is None:
            raise ValueError(f"{self.name} needs a covariance estimate")
        result = risk_parity(ctx.covariance, Constraints(max_weight=self.max_weight))
        if not result.success:
            raise ValueError(
                f"{self.name}: risk parity did not converge at {ctx.t} "
                f"({result.message}). The contributions are not equal, so this "
                f"is not the strategy it claims to be."
            )
        return result.weights_dict()


class HierarchicalRiskParity:
    """
    HRP from the context's covariance.

    Never inverts a matrix, which is what makes it hold up when the covariance
    is poorly conditioned — exactly the windows where minimum variance produces
    its most confident and least reliable answers.
    """

    name = "hrp"

    def __init__(self, linkage_method: str = "single",
                 max_weight: float | None = None) -> None:
        self.linkage_method = linkage_method
        self.max_weight = max_weight

    @property
    def params(self) -> dict:
        return {"linkage_method": self.linkage_method,
                "max_weight": self.max_weight}

    def target_weights(self, ctx: Context) -> dict[str, float]:
        from src.analytics.optimization import Constraints, hrp

        if ctx.covariance is None:
            raise ValueError(f"{self.name} needs a covariance estimate")
        return hrp(
            ctx.covariance, Constraints(max_weight=self.max_weight),
            linkage_method=self.linkage_method,
        ).weights_dict()
