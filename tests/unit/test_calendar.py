"""
Calendar awareness (Milestone B).

The tests that matter most here are the negative ones: that weekends are never
manufactured, that a mixed universe is intersected rather than filled, and that
an analytic with no meaning for a 24/7 asset refuses to run instead of
returning a number. Those are the failure modes that would otherwise be
invisible — a forward-filled series produces perfectly plausible-looking output.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from src.analytics.calendar_policy import (
    CalendarPolicy,
    UnsupportedCalendarError,
    describe,
    require_trading_days,
)
from src.analytics.factors.prepare import FF5_FACTORS, align_factors
from src.analytics.risk.covariance import estimate_covariance
from src.data_quality.checks import check_observation_gaps
from src.domain.calendar import Calendar, resolve
from src.domain.portfolio import Asset, AssetClass
from src.domain.returns import ReturnSeries
from src.ingestion.prices import PRICE_REGISTRY, available_crypto
from src.ingestion.prices import resolve as resolve_source
from src.store.parquet_store import ParquetStore

# ──────────────────────────────────────────────────────────────
# Calendar primitives
# ──────────────────────────────────────────────────────────────

def test_periods_per_year():
    assert Calendar.TRADING_DAYS.periods_per_year == 252
    assert Calendar.CONTINUOUS.periods_per_year == 365


def test_max_normal_gap_reflects_weekends():
    # A long weekend is routine on an exchange; any gap is suspicious on a
    # venue that never closes.
    assert Calendar.TRADING_DAYS.max_normal_gap_days == 4
    assert Calendar.CONTINUOUS.max_normal_gap_days == 1


def test_calendar_from_asset_class():
    assert Calendar.for_asset_class(AssetClass.CRYPTO) is Calendar.CONTINUOUS
    assert Calendar.for_asset_class(AssetClass.EQUITY) is Calendar.TRADING_DAYS
    assert Calendar.for_asset_class(AssetClass.FIXED_INCOME) is Calendar.TRADING_DAYS


def test_resolve_picks_the_most_restrictive():
    # Intersecting continuous with trading days leaves only trading days, so
    # annualising the result at 365 would overstate volatility.
    mixed = [Calendar.CONTINUOUS, Calendar.TRADING_DAYS]
    assert resolve(mixed) is Calendar.TRADING_DAYS
    assert resolve([Calendar.CONTINUOUS, Calendar.CONTINUOUS]) is Calendar.CONTINUOUS
    assert resolve([]) is Calendar.TRADING_DAYS


# ──────────────────────────────────────────────────────────────
# Domain model
# ──────────────────────────────────────────────────────────────

def test_asset_derives_calendar_from_class():
    btc = Asset("BTC-USD", "Bitcoin", AssetClass.CRYPTO)
    spy = Asset("SPY", "S&P 500", AssetClass.EQUITY)
    assert btc.calendar is Calendar.CONTINUOUS
    assert btc.periods_per_year == 365
    assert spy.calendar is Calendar.TRADING_DAYS
    assert spy.periods_per_year == 252


def test_explicit_calendar_overrides_the_asset_class_default():
    # A crypto product that only trades on an exchange schedule (a futures ETF,
    # say) must be able to say so.
    fund = Asset("BITO", "BTC futures ETF", AssetClass.CRYPTO,
                 calendar=Calendar.TRADING_DAYS)
    assert fund.calendar is Calendar.TRADING_DAYS


def _series(n: int, calendar: Calendar, tickers=("A", "B")) -> ReturnSeries:
    rng = np.random.default_rng(0)
    start = dt.date(2020, 1, 1)
    data = {"date": [start + dt.timedelta(days=i) for i in range(n)]}
    for t in tickers:
        data[t] = rng.normal(0, 0.01, n)
    return ReturnSeries(pl.DataFrame(data), list(tickers), calendar=calendar)


def test_return_series_exposes_periods_per_year():
    assert _series(50, Calendar.CONTINUOUS).periods_per_year == 365
    assert _series(50, Calendar.TRADING_DAYS).periods_per_year == 252


def test_calendar_survives_select_and_trim():
    # Losing the calendar on a slice is the subtle version of this bug: every
    # rolling window would silently revert to 252.
    rs = _series(50, Calendar.CONTINUOUS)
    assert rs.select(["A"]).calendar is Calendar.CONTINUOUS
    assert rs.trim(start="2020-01-10").calendar is Calendar.CONTINUOUS


# ──────────────────────────────────────────────────────────────
# Annualisation derives from the data
# ──────────────────────────────────────────────────────────────

def test_covariance_annualises_on_the_series_calendar():
    cont = _series(400, Calendar.CONTINUOUS)
    trad = ReturnSeries(cont.data, cont.tickers, calendar=Calendar.TRADING_DAYS)

    c_cont = estimate_covariance(cont, method="sample")
    c_trad = estimate_covariance(trad, method="sample")

    assert c_cont.ppy == 365
    assert c_trad.ppy == 252
    # Same underlying returns, so the ratio is exactly the annualisation factor.
    ratio = c_cont.volatilities[0] / c_trad.volatilities[0]
    assert ratio == pytest.approx(np.sqrt(365 / 252), rel=1e-9)


def test_explicit_ppy_still_wins():
    rs = _series(300, Calendar.CONTINUOUS)
    assert estimate_covariance(rs, method="sample", ppy=252).ppy == 252


def test_sample_covariance_handles_a_single_asset():
    # np.cov collapses a one-column input to a 0-d scalar; without the reshape
    # every consumer calling np.diag on the result raises.
    rs = _series(100, Calendar.CONTINUOUS, tickers=("A",))
    cov = estimate_covariance(rs, method="sample")
    assert cov.matrix.shape == (1, 1)
    assert cov.volatilities.shape == (1,)


# ──────────────────────────────────────────────────────────────
# Price registry
# ──────────────────────────────────────────────────────────────

def test_registry_routes_crypto_to_ccxt():
    src = resolve_source("BTC-USD")
    assert src.backend == "ccxt"
    assert src.exchange == "binance"
    assert src.code == "BTC/USDT"
    assert src.calendar is Calendar.CONTINUOUS
    assert src.asset_class == "crypto"


def test_registry_falls_back_to_yahoo_for_unknown_tickers():
    # The fallback is what keeps the universe open: any Yahoo symbol works
    # without a code change.
    src = resolve_source("VWCE.DE")
    assert src.backend == "yahoo"
    assert src.code == "VWCE.DE"
    assert src.calendar is Calendar.TRADING_DAYS


def test_logical_ticker_is_filesystem_safe():
    # The store writes one Parquet file per ticker, so an exchange symbol
    # containing a slash must never reach it.
    for ticker, source in PRICE_REGISTRY.items():
        assert "/" not in ticker
        if source.backend == "ccxt":
            assert "/" in source.code      # the exchange symbol does have one


def test_available_crypto_lists_registered_pairs():
    assert "BTC-USD" in available_crypto()
    assert "ETH-USD" in available_crypto()


# ──────────────────────────────────────────────────────────────
# Store: intersection, never forward-fill
# ──────────────────────────────────────────────────────────────

def _write_prices(store: ParquetStore, ticker: str, days: list[dt.date],
                  calendar: Calendar) -> None:
    n = len(days)
    df = pl.DataFrame({
        "date": days,
        "ticker": [ticker] * n,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(100, 110, n),
        "low": np.linspace(100, 110, n),
        "close": np.linspace(100, 110, n),
        "adj_close": np.linspace(100, 110, n),
        "volume": [1000] * n,
    })
    store.write_prices(ticker, df, upsert=False)
    store.write_asset_meta(ticker, calendar=calendar)


def test_mixed_calendars_intersect_and_never_fill(tmp_path):
    store = ParquetStore(tmp_path)
    start = dt.date(2024, 1, 1)
    all_days = [start + dt.timedelta(days=i) for i in range(60)]
    weekdays = [d for d in all_days if d.weekday() < 5]

    _write_prices(store, "BTC", all_days, Calendar.CONTINUOUS)
    _write_prices(store, "SPY", weekdays, Calendar.TRADING_DAYS)

    rs = store.read_returns(["BTC", "SPY"])

    # The intersection keeps weekdays only — no weekend was invented for SPY.
    assert all(d.weekday() < 5 for d in rs.dates.to_list())
    assert rs.n_obs == len(weekdays) - 1          # one lost to differencing
    # ...and the combined series annualises on the restrictive calendar.
    assert rs.calendar is Calendar.TRADING_DAYS
    assert rs.periods_per_year == 252


def test_pure_crypto_keeps_its_own_calendar(tmp_path):
    store = ParquetStore(tmp_path)
    start = dt.date(2024, 1, 1)
    all_days = [start + dt.timedelta(days=i) for i in range(60)]
    _write_prices(store, "BTC", all_days, Calendar.CONTINUOUS)
    _write_prices(store, "ETH", all_days, Calendar.CONTINUOUS)

    rs = store.read_returns(["BTC", "ETH"])
    assert rs.calendar is Calendar.CONTINUOUS
    assert rs.periods_per_year == 365
    # Weekends are kept: they are information for a 24/7 asset.
    assert any(d.weekday() >= 5 for d in rs.dates.to_list())


def test_unknown_ticker_defaults_to_trading_days(tmp_path):
    # Price files written before the metadata existed must behave exactly as
    # they did before — this is what keeps the change backward compatible.
    store = ParquetStore(tmp_path)
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(30)]
    n = len(days)
    store.write_prices("OLD", pl.DataFrame({
        "date": days, "ticker": ["OLD"] * n,
        "open": np.linspace(1, 2, n), "high": np.linspace(1, 2, n),
        "low": np.linspace(1, 2, n), "close": np.linspace(1, 2, n),
        "adj_close": np.linspace(1, 2, n), "volume": [1] * n,
    }), upsert=False)
    assert store.calendar_for("OLD") is Calendar.TRADING_DAYS
    assert store.read_returns(["OLD"]).calendar is Calendar.TRADING_DAYS


def test_asset_meta_upserts(tmp_path):
    store = ParquetStore(tmp_path)
    store.write_asset_meta("X", Calendar.TRADING_DAYS, "equity", "yahoo")
    store.write_asset_meta("X", Calendar.CONTINUOUS, "crypto", "ccxt:binance")
    meta = store.read_asset_meta()
    assert len(meta) == 1
    assert meta["X"]["calendar"] == "continuous"
    assert store.calendar_for("X") is Calendar.CONTINUOUS


def test_asset_meta_is_not_mistaken_for_a_ticker(tmp_path):
    # It lives outside prices/ precisely so the DuckDB prices() glob and
    # available_tickers() never pick it up.
    store = ParquetStore(tmp_path)
    _write_prices(store, "BTC", [dt.date(2024, 1, 1)], Calendar.CONTINUOUS)
    assert store.available_tickers() == ["BTC"]
    assert store.assets_path.parent != store.prices_dir


# ──────────────────────────────────────────────────────────────
# Cross-calendar policy
# ──────────────────────────────────────────────────────────────

def test_require_trading_days_rejects_continuous():
    with pytest.raises(UnsupportedCalendarError, match="continuous"):
        require_trading_days(Calendar.CONTINUOUS, "The factor engine", "because.")
    # ...and is a no-op on the supported calendar
    require_trading_days(Calendar.TRADING_DAYS, "The factor engine", "because.")


def test_factor_engine_refuses_a_continuous_series():
    """
    A beta of BTC on HML has no referent. The engine must say so rather than
    return a number someone will later paste into a report.
    """
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(100)]
    rng = np.random.default_rng(0)
    returns = pl.Series("r", rng.normal(0, 0.02, len(days)))
    factors = pl.DataFrame(
        {"date": days,
         **{f: rng.normal(0, 0.01, len(days)) for f in FF5_FACTORS},
         "RF": np.full(len(days), 0.0001)}
    )

    with pytest.raises(UnsupportedCalendarError):
        align_factors(returns, pl.Series("date", days), factors, FF5_FACTORS,
                      calendar=Calendar.CONTINUOUS)

    # The same call on the default trading-day calendar still works.
    aligned = align_factors(returns, pl.Series("date", days), factors, FF5_FACTORS)
    assert aligned.n_obs == len(days)


def test_describe_states_what_happened():
    assert "365" in describe(CalendarPolicy.NATIVE, Calendar.CONTINUOUS)
    assert "252" in describe(CalendarPolicy.INTERSECTION, Calendar.TRADING_DAYS)


# ──────────────────────────────────────────────────────────────
# Calendar-aware data quality
# ──────────────────────────────────────────────────────────────

def _price_frame(days: list[dt.date]) -> pl.DataFrame:
    return pl.DataFrame({
        "date": days,
        "adj_close": np.linspace(100, 110, len(days)),
    })


def test_weekend_gaps_are_normal_on_an_exchange_calendar():
    days = [d for d in
            (dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40))
            if d.weekday() < 5]
    assert check_observation_gaps(_price_frame(days), "SPY",
                                  Calendar.TRADING_DAYS) == []


def test_the_same_gaps_are_anomalies_on_a_continuous_calendar():
    # Identical data, different calendar: a weekend-shaped hole in a 24/7 series
    # is an outage, and this is the check that only exists because the calendar
    # is explicit.
    days = [d for d in
            (dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(40))
            if d.weekday() < 5]
    findings = check_observation_gaps(_price_frame(days), "BTC-USD",
                                      Calendar.CONTINUOUS)
    assert len(findings) == 1
    assert findings[0].check == "observation_gaps"
    assert findings[0].detail["count"] == 5      # five weekends in 40 days


def test_a_real_outage_is_caught_on_a_continuous_calendar():
    days = ([dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(10)]
            + [dt.date(2024, 1, 20) + dt.timedelta(days=i) for i in range(10)])
    findings = check_observation_gaps(_price_frame(days), "BTC-USD",
                                      Calendar.CONTINUOUS)
    assert findings[0].detail["gaps"][0]["days"] == 10


def test_no_gaps_reported_for_a_complete_continuous_series():
    days = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(60)]
    assert check_observation_gaps(_price_frame(days), "BTC-USD",
                                  Calendar.CONTINUOUS) == []
