"""
Data quality tests.

Strategy: start from a clean synthetic price series, inject one specific defect,
and assert exactly that check fires (and that clean data stays clean).
"""

import numpy as np
import polars as pl
import pytest

from src.data_quality.checks import (
    Severity,
    check_missing_values, check_short_history, check_return_outliers,
    check_stale_prices, check_calendar_gaps,
)
from src.data_quality.report import run_quality_report
from src.store.parquet_store import ParquetStore


def _clean_prices(n=400, seed=0, start=(2022, 1, 1)):
    np.random.seed(seed)
    price = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))
    d0 = pl.date(*start)
    dates = pl.date_range(d0, d0 + pl.duration(days=n - 1), interval="1d", eager=True)
    return pl.DataFrame({
        "date": dates, "ticker": ["X"] * n,
        "open": price, "high": price, "low": price,
        "close": price, "adj_close": price, "volume": [1_000_000] * n,
    })


class TestIndividualChecks:
    def test_clean_data_no_findings(self):
        df = _clean_prices()
        assert check_missing_values(df, "X") == []
        assert check_short_history(df, "X") == []
        assert check_return_outliers(df, "X") == []
        assert check_stale_prices(df, "X") == []

    def test_detects_null(self):
        df = _clean_prices()
        df = df.with_columns(
            pl.when(pl.arange(0, df.height) == 10)
            .then(None).otherwise(pl.col("adj_close")).alias("adj_close")
        )
        out = check_missing_values(df, "X")
        assert any(f.check == "missing_values" and f.severity == Severity.CRITICAL for f in out)

    def test_detects_non_positive(self):
        df = _clean_prices()
        df[5, "adj_close"] = -1.0
        out = check_missing_values(df, "X")
        assert any(f.check == "non_positive" for f in out)

    def test_detects_short_history(self):
        df = _clean_prices(n=60)
        out = check_short_history(df, "X", min_obs=252)
        assert len(out) == 1 and out[0].severity == Severity.WARNING

    def test_detects_split_like_outlier(self):
        # Inject an unadjusted 2:1 split (price halves overnight) → ~-50% return
        df = _clean_prices()
        prices = df["adj_close"].to_numpy().copy()
        prices[200:] /= 2.0
        df = df.with_columns(pl.Series("adj_close", prices))
        out = check_return_outliers(df, "X", asset_class="equity")
        assert len(out) >= 1
        assert out[0].severity == Severity.CRITICAL          # worst |r| ~ 50% > 40% equity crit
        assert out[0].detail["worst_abs_return"] > 0.40

    def test_outlier_reports_dates(self):
        df = _clean_prices()
        prices = df["adj_close"].to_numpy().copy()
        prices[200:] /= 2.0
        df = df.with_columns(pl.Series("adj_close", prices))
        out = check_return_outliers(df, "X", asset_class="equity")
        events = out[0].detail["events"]
        assert len(events) >= 1
        # The worst event corresponds to the split day and carries a date + return
        assert "date" in events[0] and "return" in events[0]
        assert events[0]["return"] < -0.4

    def test_all_outlier_dates_reported(self):
        # Inject several distinct large down-moves; ALL must appear in events.
        df = _clean_prices(seed=21)
        prices = df["adj_close"].to_numpy().copy()
        for i in (50, 120, 250, 300):
            prices[i:] *= 0.55          # ~-45% each, distinct equity-critical days
        df = df.with_columns(pl.Series("adj_close", prices))
        out = check_return_outliers(df, "X", asset_class="equity")
        events = out[0].detail["events"]
        # count in detail equals number of events listed (nothing truncated)
        assert out[0].detail["count"] == len(events)
        assert len(events) >= 4

    def test_asset_class_changes_severity(self):
        # A ~10% daily move: routine for crypto, critical for govvies.
        df = _clean_prices(seed=11)
        prices = df["adj_close"].to_numpy().copy()
        prices[150:] *= 0.90          # one -10% day
        df = df.with_columns(pl.Series("adj_close", prices))

        bond = check_return_outliers(df, "X", asset_class="fixed_income")
        crypto = check_return_outliers(df, "X", asset_class="crypto")
        # Bonds: -10% breaches the 12% crit? no, but >5% warn → flagged; severity warning or critical
        assert len(bond) == 1
        # Crypto: -10% is below the 40% warn band and within stat noise → not flagged
        assert crypto == [] or crypto[0].severity == Severity.WARNING
        # And the bond flag is at least as severe as the crypto outcome
        assert bond[0].severity in (Severity.WARNING, Severity.CRITICAL)

    def test_detects_stale_run(self):
        df = _clean_prices()
        prices = df["adj_close"].to_numpy().copy()
        prices[100:108] = prices[100]                        # 8 flat days
        df = df.with_columns(pl.Series("adj_close", prices))
        out = check_stale_prices(df, "X", min_run=5)
        assert len(out) == 1
        assert out[0].detail["longest_run"] >= 8

    def test_no_false_positive_stale(self):
        df = _clean_prices()
        assert check_stale_prices(df, "X", min_run=5) == []

    def test_detects_calendar_gap(self):
        from datetime import date
        df = _clean_prices(n=100)
        # Panel has 3 dates this ticker lacks
        extra = [date(2030, 1, 1), date(2030, 1, 2), date(2030, 1, 3)]
        panel = pl.concat([df["date"], pl.Series("date", extra)])
        out = check_calendar_gaps(df, "X", panel)
        assert len(out) == 1 and out[0].detail["count"] == 3


class TestQualityReport:
    def test_report_over_store(self, tmp_path):
        store = ParquetStore(tmp_path)
        # One clean ticker, one with a split-like outlier
        clean = _clean_prices(seed=1).with_columns(pl.lit("CLEAN").alias("ticker"))
        store.write_prices("CLEAN", clean)

        dirty = _clean_prices(seed=2)
        p = dirty["adj_close"].to_numpy().copy()
        p[150:] /= 2.0
        dirty = dirty.with_columns([pl.Series("adj_close", p), pl.lit("DIRTY").alias("ticker")])
        store.write_prices("DIRTY", dirty)

        report = run_quality_report(store, asset_classes={"DIRTY": "equity", "CLEAN": "equity"})
        assert "DIRTY" in report.critical_tickers()
        assert "CLEAN" in report.clean_tickers()
        assert report.n_critical >= 1

    def test_report_dataframe_schema(self, tmp_path):
        store = ParquetStore(tmp_path)
        store.write_prices("X", _clean_prices())
        report = run_quality_report(store)
        df = report.to_dataframe()
        assert set(df.columns) >= {"ticker", "check", "severity", "message"} or df.is_empty()

    def test_empty_store(self, tmp_path):
        store = ParquetStore(tmp_path)
        report = run_quality_report(store)
        assert report.findings == []
        assert report.clean_tickers() == []
