"""
Performance analytics — pure functions.
Input: np.ndarray of returns. Output: scalar metrics or DataFrames.
No state, no side effects, fully testable in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries


@dataclass
class PerformanceMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
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

    def to_dict(self) -> dict:
        return {k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}

    def summary(self) -> str:
        return "\n".join([
            "-- Performance Metrics --------------------",
            f"  Total Return        {self.total_return:>10.2%}",
            f"  Ann. Return         {self.annualized_return:>10.2%}",
            f"  Ann. Volatility     {self.annualized_volatility:>10.2%}",
            f"  Sharpe Ratio        {self.sharpe_ratio:>10.3f}",
            f"  Sortino Ratio       {self.sortino_ratio:>10.3f}",
            f"  Calmar Ratio        {self.calmar_ratio:>10.3f}",
            f"  Max Drawdown        {self.max_drawdown:>10.2%}",
            f"  VaR (95%)           {self.var_95:>10.2%}",
            f"  CVaR (95%)          {self.cvar_95:>10.2%}",
            f"  Hit Rate            {self.hit_rate:>10.2%}",
            "-------------------------------------------",
        ])


def total_return(r: np.ndarray) -> float:
    return float(np.prod(1 + r) - 1)


def annualized_return(r: np.ndarray, ppy: int = 252) -> float:
    n = len(r)
    if n == 0:
        return 0.0
    return float(np.prod(1 + r) ** (ppy / n) - 1)


def annualized_volatility(r: np.ndarray, ppy: int = 252) -> float:
    return float(np.std(r, ddof=1) * np.sqrt(ppy))


def downside_deviation(r: np.ndarray, mar: float = 0.0, ppy: int = 252) -> float:
    excess = r - mar / ppy
    neg = np.where(excess < 0, excess, 0.0)
    return float(np.sqrt(np.mean(neg ** 2)) * np.sqrt(ppy))


def sharpe_ratio(r: np.ndarray, rfr: float = 0.04, ppy: int = 252) -> float:
    vol = annualized_volatility(r, ppy)
    if vol == 0:
        return 0.0
    return float((annualized_return(r, ppy) - rfr) / vol)


def sortino_ratio(r: np.ndarray, rfr: float = 0.04, ppy: int = 252) -> float:
    dd = downside_deviation(r, mar=rfr, ppy=ppy)
    if dd == 0:
        return 0.0
    return float((annualized_return(r, ppy) - rfr) / dd)


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
    return float(-np.percentile(r, (1 - confidence) * 100))


def historical_cvar(r: np.ndarray, confidence: float = 0.95) -> float:
    var = historical_var(r, confidence)
    tail = r[r <= -var]
    return var if len(tail) == 0 else float(-np.mean(tail))


def compute_metrics(r: np.ndarray, rfr: float = 0.04, ppy: int = 252,
                    var_conf: float = 0.95) -> PerformanceMetrics:
    from scipy import stats
    r = r[~np.isnan(r)]
    return PerformanceMetrics(
        total_return=total_return(r),
        annualized_return=annualized_return(r, ppy),
        annualized_volatility=annualized_volatility(r, ppy),
        downside_deviation=downside_deviation(r, mar=rfr, ppy=ppy),
        sharpe_ratio=sharpe_ratio(r, rfr, ppy),
        sortino_ratio=sortino_ratio(r, rfr, ppy),
        calmar_ratio=calmar_ratio(r, ppy),
        max_drawdown=max_drawdown(r),
        max_drawdown_duration=max_drawdown_duration(r),
        skewness=float(stats.skew(r)),
        excess_kurtosis=float(stats.kurtosis(r)),
        var_95=historical_var(r, var_conf),
        cvar_95=historical_cvar(r, var_conf),
        n_observations=len(r),
        hit_rate=float(np.mean(r > 0)),
    )


def rolling_metric(rs: ReturnSeries, ticker: str, window: int = 252,
                   ppy: int = 252, rfr: float = 0.04) -> pl.DataFrame:
    r = rs.to_numpy_series(ticker)
    dates = rs.dates.to_list()
    rows = []
    for i in range(window, len(r) + 1):
        w = r[i - window:i]
        rows.append({
            "date": dates[i - 1],
            "rolling_sharpe": sharpe_ratio(w, rfr, ppy),
            "rolling_vol": annualized_volatility(w, ppy),
            "rolling_max_dd": max_drawdown(w),
        })
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))
