"""
Domain objects — pure Python dataclasses. Zero external dependencies.
These are the nouns of the system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AssetClass(str, Enum):
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CASH = "cash"
    ALTERNATIVE = "alternative"


class Currency(str, Enum):
    USD = "USD"
    EUR = "EUR"
    GBP = "GBP"
    JPY = "JPY"


@dataclass(frozen=True)
class Asset:
    ticker: str
    name: str
    asset_class: AssetClass
    currency: Currency = Currency.USD
    region: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.ticker} ({self.name})"


@dataclass
class Position:
    asset: Asset
    weight: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(
                f"Weight for {self.asset.ticker} must be in [0, 1], got {self.weight}"
            )


@dataclass
class Portfolio:
    name: str
    positions: list[Position] = field(default_factory=list)
    currency: Currency = Currency.USD
    description: str = ""

    def __post_init__(self) -> None:
        if self.positions:
            self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(p.weight for p in self.positions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Portfolio weights must sum to 1.0, got {total:.6f}")

    @property
    def tickers(self) -> list[str]:
        return [p.asset.ticker for p in self.positions]

    @property
    def weights(self) -> dict[str, float]:
        return {p.asset.ticker: p.weight for p in self.positions}

    @property
    def assets(self) -> list[Asset]:
        return [p.asset for p in self.positions]

    def get_position(self, ticker: str) -> Optional[Position]:
        for p in self.positions:
            if p.asset.ticker == ticker:
                return p
        return None

    @classmethod
    def equal_weight(cls, name: str, assets: list[Asset]) -> "Portfolio":
        w = 1.0 / len(assets)
        return cls(name, [Position(a, w) for a in assets])

    def __repr__(self) -> str:
        lines = [f"Portfolio: {self.name}"]
        for p in sorted(self.positions, key=lambda x: x.weight, reverse=True):
            lines.append(f"  {p.asset.ticker:8s}  {p.weight:.1%}")
        return "\n".join(lines)
