"""
Performance analytics — pure functions.
Input: np.ndarray of returns. Output: scalar metrics or DataFrames.
No state, no side effects, fully testable in isolation.

Conventions (interview-defensible):
  - Sharpe and Sortino use the *arithmetic* mean of daily EXCESS returns,
    annualised — the textbook definition (Sharpe 1966/1994).
  - The risk-free can be a SCALAR annual rate (converted to a daily rate) OR a
    DAILY SERIES aligned to the returns. Passing the real risk-free series is
    strongly preferred over a constant when rates move across the sample.
  - `annualized_return` (geometric, a.k.a. CAGR) is reported separately as a
    performance descriptor; it is NOT used inside the Sharpe numerator.
  - Reported VaR/CVaR are 1-day at the stated confidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float          # geometric (CAGR)
    annualized_volatility: float
    annualized_excess_return: float   # arithmetic mean excess × ppy (Sharpe numerator)
    downside_deviation: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    skewness: float
    excess_kurtosis: float
    var_95: float
    cvar_95: float
    n_observations: int
    hit_rate: float
    rf_kind: str = "scalar"           # "scalar" or "series" — what the Sharpe used

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

    def summary(self) -> str:
        return "\n".join([
            "-- Performance Metrics --------------------",
            f"  Total Return         {self.total_return:>10.2%}",
            f"  Ann. Return (CAGR)   {self.annualized_return:>10.2%}",
            f"  Ann. Volatility      {self.annualized_volatility:>10.2%}",
            f"  Ann. Excess Return   {self.annualized_excess_return:>10.2%}",
            f"  Sharpe Ratio         {self.sharpe_ratio:>10.3f}   (rf: {self.rf_kind})",
            f"  Sortino Ratio        {self.sortino_ratio:>10.3f}",
            f"  Calmar Ratio         {self.calmar_ratio:>10.3f}",
            f"  Max Drawdown         {self.max_drawdown:>10.2%}",
            f"  VaR (95%, 1-day)     {self.var_95:>10.2%}",
            f"  CVaR (95%, 1-day)    {self.cvar_95:>10.2%}",
            f"  Hit Rate             {self.hit_rate:>10.2%}",
            "-------------------------------------------",
        ])


# ──────────────────────────────────────────────────────────────
# Excess returns: the foundation of risk-adjusted metrics
# ──────────────────────────────────────────────────────────────

def to_daily_rf(rf, n: int, ppy: int = 252) -> np.ndarray:
    """
    Normalise a risk-free input to a daily array of length n.
    rf may be:
      - a scalar ANNUAL rate (e.g. 0.04)         → constant daily rate rf/ppy
      - a daily array/series aligned to returns  → used as-is
    """
    if np.isscalar(rf):
        return np.full(n, float(rf) / ppy)
    rf = np.asarray(rf, dtype=float)
    if len(rf) != n:
        raise ValueError(
            f"risk-free series length {len(rf)} != returns length {n}; "
            f"align them on date first (see align_risk_free)."
        )
    return rf


def excess_returns(r: np.ndarray, rf, ppy: int = 252) -> np.ndarray:
    """Daily excess returns r_t - rf_t."""
    return r - to_daily_rf(rf, len(r), ppy)


def align_risk_free(
    returns: pl.Series,
    dates: pl.Series,
    rf_wide: pl.DataFrame,
    rf_col: str = "RF",
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Inner-join a return series with a daily risk-free column on date.
    Returns (aligned_returns, aligned_daily_rf, dates). The RF column is assumed
    to already be a DAILY rate in decimal (e.g. the French 'RF' factor).
    """
    ret_df = pl.DataFrame({"date": dates, "ret": returns})
    if rf_col not in rf_wide.columns:
        raise ValueError(f"'{rf_col}' not in rf_wide columns: {rf_wide.columns}")
    merged = ret_df.join(
        rf_wide.select(["date", rf_col]), on="date", how="inner"
    ).drop_nulls().sort("date")
    return (
        merged["ret"].to_numpy(),
        merged[rf_col].to_numpy(),
        merged["date"].to_list(),
    )


# ──────────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────────

def total_return(r: np.ndarray) -> float:
    return float(np.prod(1 + r) - 1)


def annualized_return(r: np.ndarray, ppy: int = 252) -> float:
    """Geometric annualised return (CAGR)."""
    n = len(r)
    if n == 0:
        return 0.0
    return float(np.prod(1 + r) ** (ppy / n) - 1)


def annualized_volatility(r: np.ndarray, ppy: int = 252) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(ppy))


def downside_deviation(r: np.ndarray, ppy: int = 252) -> float:
    """
    Annualised downside deviation of (already-excess) returns below zero.
    Pass excess returns in; the MAR is therefore the risk-free rate.
    """
    neg = np.minimum(r, 0.0)
    return float(np.sqrt(np.mean(neg ** 2)) * np.sqrt(ppy))


def sharpe_ratio(r: np.ndarray, rf=0.04, ppy: int = 252) -> float:
    """
    Standard Sharpe: arithmetic mean of daily excess returns, annualised,
    divided by the annualised volatility of excess returns.
    rf may be a scalar annual rate or a daily series (see to_daily_rf).
    """
    ex = excess_returns(r, rf, ppy)
    sd = np.std(ex, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(ex) / sd * np.sqrt(ppy))


def sortino_ratio(r: np.ndarray, rf=0.04, ppy: int = 252) -> float:
    """Sortino: annualised mean excess return over annualised downside deviation."""
    ex = excess_returns(r, rf, ppy)
    dd = downside_deviation(ex, ppy)
    if dd == 0:
        return 0.0
    return float(np.mean(ex) * ppy / dd)


def drawdown_series(r: np.ndarray) -> np.ndarray:
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    return cum / peak - 1


def max_drawdown(r: np.ndarray) -> float:
    return float(np.min(drawdown_series(r)))


def max_drawdown_duration(r: np.ndarray) -> int:
    dd = drawdown_series(r)
    in_dd = dd < 0
    max_dur = cur = 0
    for v in in_dd:
        cur = cur + 1 if v else 0
        max_dur = max(max_dur, cur)
    return max_dur


def calmar_ratio(r: np.ndarray, ppy: int = 252) -> float:
    mdd = abs(max_drawdown(r))
    if mdd == 0:
        return 0.0
    return float(annualized_return(r, ppy) / mdd)


def historical_var(r: np.ndarray, confidence: float = 0.95) -> float:
    """1-day historical VaR (positive number = loss)."""
    return float(-np.percentile(r, (1 - confidence) * 100))


def historical_cvar(r: np.ndarray, confidence: float = 0.95) -> float:
    """1-day historical CVaR / Expected Shortfall (positive = loss)."""
    var = historical_var(r, confidence)
    tail = r[r <= -var]
    return var if len(tail) == 0 else float(-np.mean(tail))


def compute_metrics(r: np.ndarray, rf=0.04, ppy: int = 252,
                    var_conf: float = 0.95) -> PerformanceMetrics:
    """
    Full metric set. `rf` may be a scalar annual rate or a daily series aligned
    to `r` (preferred — use align_risk_free to build it).
    """
    from scipy import stats
    mask = ~np.isnan(r)
    r = r[mask]
    rf_is_series = not np.isscalar(rf)
    if rf_is_series:
        rf = np.asarray(rf)[mask]

    ex = excess_returns(r, rf, ppy)

    return PerformanceMetrics(
        total_return=total_return(r),
        annualized_return=annualized_return(r, ppy),
        annualized_volatility=annualized_volatility(r, ppy),
        annualized_excess_return=float(np.mean(ex) * ppy),
        downside_deviation=downside_deviation(ex, ppy),
        sharpe_ratio=sharpe_ratio(r, rf, ppy),
        sortino_ratio=sortino_ratio(r, rf, ppy),
        calmar_ratio=calmar_ratio(r, ppy),
        max_drawdown=max_drawdown(r),
        max_drawdown_duration=max_drawdown_duration(r),
        skewness=float(stats.skew(r)),
        excess_kurtosis=float(stats.kurtosis(r)),
        var_95=historical_var(r, var_conf),
        cvar_95=historical_cvar(r, var_conf),
        n_observations=len(r),
        hit_rate=float(np.mean(r > 0)),
        rf_kind="series" if rf_is_series else "scalar",
    )


def rolling_metric(rs: ReturnSeries, ticker: str, window: int = 252,
                   ppy: int = 252, rf=0.04) -> pl.DataFrame:
    r = rs.to_numpy_series(ticker)
    dates = rs.dates.to_list()
    rows = []
    for i in range(window, len(r) + 1):
        w = r[i - window:i]
        rows.append({
            "date": dates[i - 1],
            "rolling_sharpe": sharpe_ratio(w, rf, ppy),
            "rolling_vol": annualized_volatility(w, ppy),
            "rolling_max_dd": max_drawdown(w),
        })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
