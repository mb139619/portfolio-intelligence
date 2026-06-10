"""
Historical stress — replay a crisis window.

Two methods:
  asset_replay  : apply the current weights to the actual asset returns over the
                  window. Intuitive, but needs the assets to have existed then.
  factor_replay : apply the current factor betas to the factor returns over the
                  window. Works for ANY era (Fama-French history reaches 1926),
                  so it covers crises that predate your ETFs. Captures only the
                  systematic part (by construction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl
from loguru import logger

from src.domain.returns import ReturnSeries
from src.analytics.stress.scenarios import HistoricalScenario


@dataclass
class StressResult:
    scenario: str
    method: str
    total_pnl: float                      # portfolio return over the window (decimal)
    contributions: dict[str, float]       # per asset / per factor (approx, see note)
    window: Optional[str] = None
    max_drawdown: Optional[float] = None
    note: str = ""

    def summary(self) -> str:
        lines = [
            f"-- Stress: {self.scenario} ({self.method}) --",
            f"  Window         {self.window}",
            f"  Portfolio PnL  {self.total_pnl:>8.2%}",
        ]
        if self.max_drawdown is not None:
            lines.append(f"  Max drawdown   {self.max_drawdown:>8.2%}")
        lines.append(f"  {'driver':<10} {'contribution':>14}")
        for k, v in sorted(self.contributions.items(), key=lambda kv: kv[1]):
            lines.append(f"  {k:<10} {v:>14.2%}")
        if self.note:
            lines.append(f"  note: {self.note}")
        return "\n".join(lines)


def run_historical_asset(
    weights: dict[str, float],
    rs: ReturnSeries,
    scenario: HistoricalScenario,
) -> StressResult:
    """
    Replay the portfolio over a crisis window using actual asset returns.
    Renormalises across whatever assets have data in the window (and warns
    if any are missing), so a partially-covered window still produces a result.
    """
    window_rs = rs.trim(start=scenario.start, end=scenario.end)
    if window_rs.n_obs == 0:
        raise ValueError(
            f"No asset data in window {scenario.start}→{scenario.end}. "
            f"Use factor-based replay for pre-history crises."
        )

    # Assets actually present with data in the window
    present = [t for t in weights if t in window_rs.tickers]
    missing = [t for t in weights if t not in present]
    if missing:
        logger.warning(
            f"Scenario {scenario.name}: no data for {missing} in window; "
            f"renormalising over {present}."
        )

    sub = window_rs.select(present)
    w = np.array([weights[t] for t in present])
    w = w / w.sum()  # renormalise

    R = sub.to_numpy()                          # (T, k)
    port_daily = R @ w
    total = float(np.prod(1 + port_daily) - 1)

    # Approximate per-asset contributions: weight × asset compounded return.
    asset_cum = np.prod(1 + R, axis=0) - 1
    contributions = {t: float(w[i] * asset_cum[i]) for i, t in enumerate(present)}

    # Max drawdown inside the window
    cum = np.cumprod(1 + port_daily)
    peak = np.maximum.accumulate(cum)
    mdd = float(np.min(cum / peak - 1))

    note = ""
    if missing:
        note = f"renormalised over {present}; contributions are approximate (compounding)."
    else:
        note = "contributions approximate (compounding/rebalancing)."

    return StressResult(
        scenario=scenario.name, method="asset_replay",
        total_pnl=total, contributions=contributions,
        window=f"{scenario.start} → {scenario.end}",
        max_drawdown=mdd, note=note,
    )


def run_historical_factor(
    model,
    factor_wide: pl.DataFrame,
    scenario: HistoricalScenario,
) -> StressResult:
    """
    Replay a crisis using the portfolio's current factor betas applied to the
    factor returns over the window. Works for any era covered by the factor
    data. Captures the SYSTEMATIC PnL only (no idiosyncratic, no alpha).

    model       : a FactorModel (provides betas + factor_names)
    factor_wide : wide factor DataFrame (date | Mkt-RF | ... )
    """
    win = factor_wide.filter(
        (pl.col("date") >= pl.lit(scenario.start).str.to_date())
        & (pl.col("date") <= pl.lit(scenario.end).str.to_date())
    ).sort("date")

    if win.is_empty():
        raise ValueError(
            f"No factor data in window {scenario.start}→{scenario.end}."
        )

    factors = model.factor_names
    F = win.select(factors).to_numpy()          # (T, K)
    b = model.beta_vector()

    systematic_daily = F @ b                    # (T,)
    total = float(np.prod(1 + systematic_daily) - 1)

    # Additive per-factor contribution (arithmetic sum over the window)
    contributions = {
        f: float(np.sum(F[:, i] * b[i])) for i, f in enumerate(factors)
    }

    cum = np.cumprod(1 + systematic_daily)
    peak = np.maximum.accumulate(cum)
    mdd = float(np.min(cum / peak - 1))

    return StressResult(
        scenario=scenario.name, method="factor_replay",
        total_pnl=total, contributions=contributions,
        window=f"{scenario.start} → {scenario.end}",
        max_drawdown=mdd,
        note="systematic PnL only (current betas × historical factor moves); "
             "per-factor contributions are arithmetic (sum to ~total). "
             "CAVEAT: betas are full-sample and assumed constant — in real "
             "crises betas and correlations typically rise (correlation "
             "breakdown), so this likely UNDERSTATES the true loss.",
    )
