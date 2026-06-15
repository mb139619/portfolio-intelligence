"""
Data quality checks.

Free price data (Yahoo et al.) fails in subtle ways that silently corrupt every
downstream analytic. This module DETECTS and REPORTS problems — it does not
auto-correct them. The philosophy is the same discipline a real risk desk uses:
surface where to look, attach a severity, and let a human decide.

Each check is a pure function over a price DataFrame and returns a list of
QualityFinding. The report (report.py) aggregates them per ticker.

Checks:
  - calendar gaps        : trading days missing for one ticker vs the panel
  - return outliers       : |daily return| beyond a robust threshold (likely a
                            bad tick or an unadjusted split)
  - stale prices          : identical price repeated for N+ days (suspended /
                            illiquid → artificially low volatility)
  - short history         : fewer observations than a usable minimum
  - non-positive / null   : adjusted close that is null or <= 0
  - large unexplained jump : price jump with no plausible split ratio
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import polars as pl


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# Absolute daily-return magnitudes that are "suspicious" per asset class.
# A 10% daily move is routine for crypto, a red flag for a government-bond ETF.
# (warn, critical) bands on |daily return|.
ASSET_CLASS_OUTLIER_BANDS: dict[str, tuple[float, float]] = {
    "fixed_income": (0.05, 0.12),   # govvies/IG barely move; >12% is almost surely a data error
    "equity":       (0.20, 0.40),   # single-stock/ETF; >40% likely an unadjusted split
    "commodity":    (0.20, 0.50),
    "fx":           (0.08, 0.20),
    "crypto":       (0.40, 0.80),   # genuinely volatile; only extreme moves are suspect
    "alternative":  (0.25, 0.50),
    "unknown":      (0.25, 0.50),   # conservative default
}


def outlier_bands_for(asset_class: str | None) -> tuple[float, float]:
    """(warning, critical) absolute-return bands for an asset class."""
    if asset_class is None:
        return ASSET_CLASS_OUTLIER_BANDS["unknown"]
    return ASSET_CLASS_OUTLIER_BANDS.get(str(asset_class).lower(),
                                         ASSET_CLASS_OUTLIER_BANDS["unknown"])


@dataclass
class QualityFinding:
    ticker: str
    check: str
    severity: Severity
    message: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper():8}] {self.ticker} · {self.check}: {self.message}"


# ──────────────────────────────────────────────────────────────
# Individual checks
# ──────────────────────────────────────────────────────────────

def check_missing_values(df: pl.DataFrame, ticker: str, col: str = "adj_close") -> list[QualityFinding]:
    """Null or non-positive adjusted prices."""
    out = []
    n_null = df.filter(pl.col(col).is_null()).height
    n_nonpos = df.filter(pl.col(col) <= 0).height
    if n_null:
        out.append(QualityFinding(
            ticker, "missing_values", Severity.CRITICAL,
            f"{n_null} null {col} values", {"count": n_null},
        ))
    if n_nonpos:
        out.append(QualityFinding(
            ticker, "non_positive", Severity.CRITICAL,
            f"{n_nonpos} non-positive {col} values", {"count": n_nonpos},
        ))
    return out


def check_short_history(df: pl.DataFrame, ticker: str, min_obs: int = 252) -> list[QualityFinding]:
    """Series too short for a typical (1-year) estimation window."""
    n = df.height
    if n < min_obs:
        return [QualityFinding(
            ticker, "short_history", Severity.WARNING,
            f"only {n} observations (< {min_obs})", {"n_obs": n, "min_obs": min_obs},
        )]
    return []


def check_return_outliers(
    df: pl.DataFrame, ticker: str, col: str = "adj_close",
    asset_class: str | None = None,
    mad_multiple: float = 8.0,
    message_sample: int = 3,
) -> list[QualityFinding]:
    """
    Flag suspicious daily returns. Two layers, combined:

      1. A robust statistical band (median ± k·MAD) catches anything far from a
         series' own typical behaviour.
      2. Absolute asset-class bands (see ASSET_CLASS_OUTLIER_BANDS) assign
         severity by economic plausibility: a 10% day is routine for crypto but
         a red flag for a government-bond ETF.

    The finding lists the actual DATES of the flagged returns so they can be
    investigated (an unadjusted split, a bad tick, a real shock).
    """
    sub = df.sort("date")
    prices = sub[col].to_numpy()
    dates = sub["date"].to_list()
    if len(prices) < 3:
        return []
    rets = prices[1:] / prices[:-1] - 1
    ret_dates = dates[1:]   # return at t is dated t

    warn_band, crit_band = outlier_bands_for(asset_class)

    # Robust statistical band
    med = np.median(rets)
    mad = np.median(np.abs(rets - med)) * 1.4826
    stat_band = mad_multiple * mad if mad > 0 else np.inf

    abs_rets = np.abs(rets)
    # An observation is flagged if it breaches the warning band OR the stat band
    mask = (abs_rets > warn_band) | (np.abs(rets - med) > stat_band)
    if not mask.any():
        return []

    flagged_idx = np.where(mask)[0]
    worst = float(abs_rets[flagged_idx].max())
    # critical if any flagged move breaches the asset-class critical band
    sev = Severity.CRITICAL if (abs_rets[flagged_idx] > crit_band).any() else Severity.WARNING

    # Build a dated list of ALL flagged returns, worst first
    order = flagged_idx[np.argsort(-abs_rets[flagged_idx])]
    events = [
        {"date": str(ret_dates[i]), "return": round(float(rets[i]), 4)}
        for i in order
    ]
    sample = events[:message_sample]
    sample_str = ", ".join(f"{e['date']} ({e['return']:+.1%})" for e in sample)
    more = "" if len(events) <= message_sample else f", +{len(events) - message_sample} more"

    return [QualityFinding(
        ticker, "return_outliers", sev,
        f"{len(flagged_idx)} outlier returns (worst |r| = {worst:.1%}, "
        f"class={asset_class or 'unknown'}, crit>{crit_band:.0%}); "
        f"dates: {sample_str}{more}",
        {
            "count": int(len(flagged_idx)),
            "worst_abs_return": worst,
            "asset_class": asset_class or "unknown",
            "warn_band": warn_band,
            "crit_band": crit_band,
            "events": events,   # ALL flagged dates, worst first
        },
    )]


def check_stale_prices(
    df: pl.DataFrame, ticker: str, col: str = "adj_close", min_run: int = 5,
) -> list[QualityFinding]:
    """
    Runs of identical prices (zero return) of length >= min_run — suspended or
    illiquid names whose flat stretch artificially deflates volatility.
    """
    prices = df.sort("date")[col].to_numpy()
    if len(prices) < min_run:
        return []
    longest = run = 1
    for i in range(1, len(prices)):
        run = run + 1 if prices[i] == prices[i - 1] else 1
        longest = max(longest, run)
    if longest >= min_run:
        return [QualityFinding(
            ticker, "stale_prices", Severity.WARNING,
            f"flat price run of {longest} days", {"longest_run": longest},
        )]
    return []


def check_calendar_gaps(
    df: pl.DataFrame, ticker: str, panel_dates: pl.Series,
    max_report: int = 5,
) -> list[QualityFinding]:
    """
    Trading days present in the panel (union of all tickers) but missing for
    this ticker — the source of silent misalignment when series are joined.
    """
    have = set(df["date"].to_list())
    missing = [d for d in panel_dates.to_list() if d not in have]
    if not missing:
        return []
    sev = Severity.WARNING if len(missing) > 5 else Severity.INFO
    sample = missing[:max_report]
    return [QualityFinding(
        ticker, "calendar_gaps", sev,
        f"{len(missing)} panel dates missing for this ticker",
        {"count": len(missing), "sample": [str(d) for d in sample]},
    )]
