"""Result object shared by all portfolio optimisers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class OptimizationResult:
    method: str
    tickers: list[str]
    weights: np.ndarray            # aligned to `tickers`, sums to 1
    expected_volatility: float     # annualised √(wᵀΣw)
    success: bool
    message: str = ""
    n_iter: int | None = None

    def weights_dict(self) -> dict[str, float]:
        return dict(zip(self.tickers, self.weights.tolist()))

    @property
    def gross_exposure(self) -> float:
        return float(np.abs(self.weights).sum())

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())

    @property
    def effective_n_positions(self) -> float:
        """Inverse Herfindahl of the weights (1 = single bet, N = equal weight)."""
        s = float(np.sum(self.weights ** 2))
        return 1.0 / s if s > 0 else 0.0

    def to_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({
            "ticker": self.tickers,
            "weight": self.weights.tolist(),
        }).sort("weight", descending=True)

    def summary(self) -> str:
        lines = [
            f"-- Optimisation: {self.method} "
            f"({'OK' if self.success else 'FAILED'}) --",
            f"  expected vol (ann.)  {self.expected_volatility:>8.2%}",
            f"  effective positions  {self.effective_n_positions:>8.2f}"
            f"   (gross {self.gross_exposure:.2f}, net {self.net_exposure:.2f})",
            f"  {'ticker':<10} {'weight':>8}",
        ]
        for row in self.to_dataframe().iter_rows(named=True):
            lines.append(f"  {row['ticker']:<10} {row['weight']:>8.2%}")
        if not self.success and self.message:
            lines.append(f"  ! {self.message}")
        return "\n".join(lines)
