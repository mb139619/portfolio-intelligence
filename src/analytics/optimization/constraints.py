"""
Portfolio constraints — declared once, honoured by every optimiser.

Before this existed each optimiser grew its own keyword arguments: `min_variance`
took `long_only` and `max_weight`, the frontier took `long_only` alone, and
anything new would have invented a third spelling. That is the failure mode this
type prevents. Constraints are a property of the *mandate*, not of the objective
you happen to be optimising, so the same `Constraints` object can be handed to
minimum variance, risk parity and HRP and mean the same thing in all three.

Feasibility is checked up front and refuses with an explanation. An optimiser
handed an impossible problem does not usually crash — it returns its starting
point with `success=False`, and callers that forget to check that get a
plausible-looking portfolio which is really just the initial guess.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Constraints:
    """
    The feasible set.

    long_only  : forbid negative weights. Also acts as an implicit regulariser
                 (Jagannathan & Ma 2003) — the non-negativity bound absorbs
                 estimation error that would otherwise show up as large
                 offsetting long/short legs.
    min_weight : per-asset floor. Applies to every asset, so a positive floor
                 forces a fully populated book.
    max_weight : per-asset cap. The usual concentration limit.
    budget     : what the weights must sum to. 1.0 is fully invested; 0.0 would
                 be a long/short book with no net exposure.
    """

    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float | None = None
    budget: float = 1.0

    def __post_init__(self) -> None:
        if self.max_weight is not None and self.max_weight <= 0:
            raise ValueError(f"max_weight must be positive, got {self.max_weight}")
        if self.max_weight is not None and self.min_weight > self.max_weight:
            raise ValueError(
                f"min_weight {self.min_weight} exceeds max_weight {self.max_weight}"
            )
        if self.long_only and self.min_weight < 0:
            raise ValueError(
                f"min_weight {self.min_weight} is negative but long_only is set; "
                f"pass long_only=False to allow shorts."
            )

    # ------------------------------------------------------------------

    def validate(self, n: int) -> None:
        """
        Refuse an infeasible problem before an optimiser quietly fails on it.

        With `n` assets the caps must be able to reach the budget from below
        and the floors must not overshoot it from above.
        """
        if n < 1:
            raise ValueError("need at least one asset")
        if self.max_weight is not None and self.max_weight * n < self.budget - 1e-9:
            raise ValueError(
                f"infeasible: max_weight={self.max_weight} × {n} assets = "
                f"{self.max_weight * n:.3f} < budget {self.budget}. "
                f"Raise the cap to at least {self.budget / n:.3f}."
            )
        if self.min_weight * n > self.budget + 1e-9:
            raise ValueError(
                f"infeasible: min_weight={self.min_weight} × {n} assets = "
                f"{self.min_weight * n:.3f} > budget {self.budget}."
            )

    def bounds(self, n: int) -> list[tuple[float | None, float | None]]:
        """Per-asset (lower, upper) pairs in the form scipy.optimize expects."""
        if self.long_only:
            upper = self.max_weight if self.max_weight is not None else self.budget
            return [(self.min_weight, upper)] * n
        lower = self.min_weight if self.min_weight != 0.0 else None
        return [(lower, self.max_weight)] * n

    def budget_constraint(self) -> dict:
        """The equality constraint tying the weights to the budget."""
        import numpy as np

        return {
            "type": "eq",
            "fun": lambda w: w.sum() - self.budget,
            "jac": lambda w: np.ones(len(w)),
        }

    def start(self, n: int):
        """A feasible starting point: equal weight, clipped into the box."""
        import numpy as np

        w = np.full(n, self.budget / n)
        lo = self.min_weight
        hi = self.max_weight if self.max_weight is not None else np.inf
        w = np.clip(w, lo, hi)
        total = w.sum()
        return w * (self.budget / total) if total else w

    @property
    def is_default(self) -> bool:
        """Long-only, fully invested, no bounds — the common case."""
        return (self.long_only and self.min_weight == 0.0
                and self.max_weight is None and self.budget == 1.0)

    def describe(self) -> str:
        parts = ["long-only" if self.long_only else "long/short"]
        if self.min_weight:
            parts.append(f"w ≥ {self.min_weight:.1%}")
        if self.max_weight is not None:
            parts.append(f"w ≤ {self.max_weight:.1%}")
        parts.append(f"Σw = {self.budget:g}")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "long_only": self.long_only,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "budget": self.budget,
        }


LONG_ONLY = Constraints()
LONG_SHORT = Constraints(long_only=False)
