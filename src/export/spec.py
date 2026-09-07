"""
Portfolio specification — the input contract for the dashboard build.

The whole point of the JSON file is that the universe is declared once, in one
place, and everything downstream (ingestion, analytics, the rendered pages) is
derived from it. Change the tickers there, rebuild, and the dashboard follows.

Validation is strict and happens here, before anything is downloaded: a typo in
a weight should fail in a millisecond with a clear message, not twenty seconds
into a Yahoo fetch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.domain.portfolio import Asset, AssetClass, Portfolio, Position


@dataclass(frozen=True)
class BuildSettings:
    """Analytic parameters. Every one of these is a real modelling choice."""

    rolling_window: int = 63          # trading days for rolling %RC and vol
    beta_window: int = 252            # window for rolling factor betas
    corr_window: int = 63             # window for average correlation
    n_clusters: int = 3               # correlation clusters
    pca_method: str = "covariance"    # "covariance" | "correlation"
    regime_states: int = 2            # HMM states
    regime_search_reps: int = 20      # EM restarts — guards against local optima
    var_confidence: float = 0.99
    evt_threshold_quantile: float = 0.95
    frontier_points: int = 50
    cloud_points: int = 1500          # random portfolios behind the frontier
    hac_lags: int | None = 5       # Newey-West lags; None = classical OLS SE

    @classmethod
    def from_dict(cls, d: dict) -> BuildSettings:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(
                f"Unknown settings: {sorted(unknown)}. "
                f"Valid keys: {sorted(known)}"
            )
        return cls(**d)


@dataclass(frozen=True)
class PositionSpec:
    ticker: str
    weight: float
    name: str = ""
    asset_class: str = "equity"

    def to_asset(self) -> Asset:
        try:
            ac = AssetClass(self.asset_class)
        except ValueError:
            valid = [c.value for c in AssetClass]
            raise ValueError(
                f"{self.ticker}: unknown asset_class {self.asset_class!r}; "
                f"valid values are {valid}"
            ) from None
        return Asset(self.ticker, self.name or self.ticker, ac)


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    positions: list[PositionSpec]
    start: str = "2018-01-01"
    end: str | None = None
    cov_method: str = "ledoit_wolf_cc"
    description: str = ""
    rates: list[str] = field(default_factory=lambda: ["USD_FEDFUNDS"])
    factor_datasets: list[str] = field(default_factory=lambda: ["FF5", "MOM"])
    # Factors are fetched from far earlier than the portfolio window on purpose.
    # Factor-based stress replay is the one method that reaches crises predating
    # the holdings themselves, and the regime model estimates far better
    # transition probabilities on decades than on a few years. It costs nothing:
    # the whole factor history is a single file of a few hundred KB.
    factor_start: str = "1990-01-01"
    settings: BuildSettings = field(default_factory=BuildSettings)

    # --- derived views ---

    @property
    def tickers(self) -> list[str]:
        return [p.ticker for p in self.positions]

    @property
    def weights(self) -> dict[str, float]:
        return {p.ticker: p.weight for p in self.positions}

    @property
    def asset_classes(self) -> dict[str, str]:
        return {p.ticker: p.asset_class for p in self.positions}

    def to_portfolio(self) -> Portfolio:
        return Portfolio(
            name=self.name,
            positions=[Position(p.to_asset(), p.weight) for p in self.positions],
            description=self.description,
        )

    def equal_weights(self) -> dict[str, float]:
        n = len(self.positions)
        return {p.ticker: 1.0 / n for p in self.positions}

    # --- loading ---

    @classmethod
    def load(cls, path: str | Path) -> PortfolioSpec:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Portfolio config not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path} is not valid JSON: {e}") from None
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> PortfolioSpec:
        if "positions" not in raw or not raw["positions"]:
            raise ValueError("Config must define a non-empty 'positions' list")

        positions = []
        seen: set[str] = set()
        for i, p in enumerate(raw["positions"]):
            if "ticker" not in p or "weight" not in p:
                raise ValueError(
                    f"positions[{i}] must have both 'ticker' and 'weight'"
                )
            ticker = str(p["ticker"]).strip().upper()
            if not ticker:
                raise ValueError(f"positions[{i}] has an empty ticker")
            if ticker in seen:
                raise ValueError(f"Duplicate ticker in positions: {ticker}")
            seen.add(ticker)
            positions.append(PositionSpec(
                ticker=ticker,
                weight=float(p["weight"]),
                name=str(p.get("name", "")),
                asset_class=str(p.get("asset_class", "equity")),
            ))

        # Fail here rather than deep inside Portfolio, so the message names the file.
        total = sum(p.weight for p in positions)
        if abs(total - 1.0) > 1e-6:
            detail = ", ".join(f"{p.ticker}={p.weight:g}" for p in positions)
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}  ({detail})"
            )

        spec = cls(
            name=str(raw.get("name", "Portfolio")),
            positions=positions,
            start=str(raw.get("start", "2018-01-01")),
            end=raw.get("end"),
            cov_method=str(raw.get("cov_method", "ledoit_wolf_cc")),
            description=str(raw.get("description", "")),
            rates=list(raw.get("rates", ["USD_FEDFUNDS"])),
            factor_datasets=list(raw.get("factor_datasets", ["FF5", "MOM"])),
            factor_start=str(raw.get("factor_start", "1990-01-01")),
            settings=BuildSettings.from_dict(raw.get("settings", {})),
        )
        # Surfaces an unknown asset_class immediately instead of at render time.
        spec.to_portfolio()
        return spec

    def describe(self) -> str:
        lines = [f"{self.name} — {len(self.positions)} positions, "
                 f"from {self.start}" + (f" to {self.end}" if self.end else "")]
        for p in sorted(self.positions, key=lambda x: x.weight, reverse=True):
            lines.append(f"  {p.ticker:<6} {p.weight:>7.1%}  {p.name}")
        return "\n".join(lines)
