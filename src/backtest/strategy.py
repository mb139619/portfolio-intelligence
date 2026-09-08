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
) -> dict[str, float]:
    """
    Validate and normalise what a strategy returned.

    Applied by the engine to every strategy output, so a badly behaved strategy
    fails loudly at its own boundary instead of producing a plausible-looking
    equity curve that is quietly wrong.

    Rejects assets outside the universe (a strategy cannot trade what the
    context does not contain), non-finite values, and a zero-sum book. Scales
    the rest to sum to 1.
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

    total = sum(raw.values())
    if abs(total) < 1e-12:
        raise ValueError(
            f"{strategy_name} returned weights summing to {total:.2e}; there is "
            f"no book to normalise."
        )

    return {t: w / total for t, w in raw.items() if w != 0.0}


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
