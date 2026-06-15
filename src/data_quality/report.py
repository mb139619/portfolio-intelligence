"""
Data quality report.

Runs every check across every ticker in the store and aggregates the findings
into a per-ticker report with a severity roll-up. Call this after ingestion as
a gate before trusting the data downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from loguru import logger

from src.store.parquet_store import ParquetStore
from src.data_quality.checks import (
    QualityFinding, Severity,
    check_missing_values, check_short_history, check_return_outliers,
    check_stale_prices, check_calendar_gaps,
)


@dataclass
class QualityReport:
    findings: list[QualityFinding]
    tickers: list[str]

    # --- roll-ups ---

    @property
    def n_critical(self) -> int:
        return sum(f.severity == Severity.CRITICAL for f in self.findings)

    @property
    def n_warning(self) -> int:
        return sum(f.severity == Severity.WARNING for f in self.findings)

    def for_ticker(self, ticker: str) -> list[QualityFinding]:
        return [f for f in self.findings if f.ticker == ticker]

    def critical_tickers(self) -> list[str]:
        return sorted({f.ticker for f in self.findings if f.severity == Severity.CRITICAL})

    def clean_tickers(self) -> list[str]:
        flagged = {f.ticker for f in self.findings}
        return [t for t in self.tickers if t not in flagged]

    def to_dataframe(self) -> pl.DataFrame:
        if not self.findings:
            return pl.DataFrame(
                schema={"ticker": pl.Utf8, "check": pl.Utf8,
                        "severity": pl.Utf8, "message": pl.Utf8}
            )
        return pl.DataFrame({
            "ticker": [f.ticker for f in self.findings],
            "check": [f.check for f in self.findings],
            "severity": [f.severity.value for f in self.findings],
            "message": [f.message for f in self.findings],
        })

    def summary(self) -> str:
        lines = [
            "-- Data Quality Report --------------------",
            f"  Tickers checked   {len(self.tickers)}",
            f"  Findings          {len(self.findings)} "
            f"({self.n_critical} critical, {self.n_warning} warning)",
            f"  Clean tickers     {len(self.clean_tickers())}/{len(self.tickers)}",
        ]
        crit = self.critical_tickers()
        if crit:
            lines.append(f"  Needs attention   {', '.join(crit)}")
        lines.append("-------------------------------------------")
        for f in sorted(self.findings, key=lambda x: (x.severity != Severity.CRITICAL, x.ticker)):
            lines.append(f"  {f}")
        return "\n".join(lines)


def run_quality_report(
    store: ParquetStore,
    tickers: list[str] | None = None,
    asset_classes: dict[str, str] | None = None,
    min_obs: int = 252,
    stale_min_run: int = 5,
) -> QualityReport:
    """
    Run all checks across the given tickers (default: everything in the store).

    asset_classes : optional {ticker: asset_class} map (e.g. "equity",
                    "fixed_income", "crypto"). Drives the outlier severity bands
                    so a 10% move is judged in context. Unmapped tickers use a
                    conservative default.
    """
    tickers = tickers or store.available_tickers()
    asset_classes = asset_classes or {}
    if not tickers:
        return QualityReport(findings=[], tickers=[])

    # Build the panel calendar = union of all dates across tickers
    panel_dates: set = set()
    frames: dict[str, pl.DataFrame] = {}
    for t in tickers:
        path = store.prices_dir / f"{t}.parquet"
        if not path.exists():
            continue
        df = pl.read_parquet(path)   # raw long format: date, ticker, OHLCV, adj_close
        if df.is_empty():
            continue
        frames[t] = df
        panel_dates.update(df["date"].to_list())
    panel = pl.Series("date", sorted(panel_dates))

    findings: list[QualityFinding] = []
    for t, df in frames.items():
        findings += check_missing_values(df, t)
        findings += check_short_history(df, t, min_obs=min_obs)
        findings += check_return_outliers(df, t, asset_class=asset_classes.get(t))
        findings += check_stale_prices(df, t, min_run=stale_min_run)
        findings += check_calendar_gaps(df, t, panel)

    report = QualityReport(findings=findings, tickers=list(frames.keys()))
    logger.info(
        f"Quality report: {len(findings)} findings "
        f"({report.n_critical} critical) across {len(frames)} tickers"
    )
    return report
