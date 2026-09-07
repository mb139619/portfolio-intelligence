"""
Cross-calendar policy.

Every analytic that can see more than one asset has to answer the same
question: what happens when those assets do not trade on the same days? Leaving
that implicit is how a platform ends up silently forward-filling weekends. So
each analytic declares one of three policies, and this module enforces it.

  NATIVE
      Runs on the asset's own frequency, whatever that is. Correct for
      single-asset-class work — tail risk, regime detection, realised
      volatility. A crypto weekend is information, not noise, and dropping it
      would throw away 28% of the sample.

  INTERSECTION
      Keep only the dates on which every asset traded. Required for anything
      that RELATES series to each other: correlations, the MST, factor betas,
      covariance. `ParquetStore.read_returns` already applies this when it
      reads a mixed universe, so by the time an analytic sees the data the
      intersection has happened and the calendar has been resolved to the most
      restrictive one.

  UNSUPPORTED
      Refuse, loudly. The Fama-French engine on a crypto book is the canonical
      case: the factors are constructed from US equity portfolios and a beta of
      BTC on HML is a number without a referent. A clear error beats a
      meaningless coefficient that someone will later put in a report.

The asymmetry worth remembering: NATIVE and INTERSECTION are both defensible
defaults depending on the question, but they are not interchangeable. Realised
volatility computed on the intersection of BTC with equities is not BTC's
volatility — it is the volatility of BTC observed on weekdays, which is a
different and less useful quantity.
"""

from __future__ import annotations

from enum import Enum

from src.domain.calendar import Calendar


class CalendarPolicy(str, Enum):
    NATIVE = "native"
    INTERSECTION = "intersection"
    UNSUPPORTED = "unsupported"


class UnsupportedCalendarError(ValueError):
    """An analytic was asked to run on a calendar it has no meaning for."""


def require_trading_days(
    calendar: Calendar,
    analytic: str,
    reason: str,
) -> None:
    """
    Guard for analytics whose policy is UNSUPPORTED outside exchange hours.

    Raises rather than warns on purpose. A warning in a log is not seen by the
    person who reads the resulting chart six months later; an exception is.
    """
    if calendar is not Calendar.TRADING_DAYS:
        raise UnsupportedCalendarError(
            f"{analytic} does not support the '{calendar}' calendar. {reason} "
            f"If you want this analysis on the exchange-traded part of the "
            f"book, build a ReturnSeries from those assets alone."
        )


def describe(policy: CalendarPolicy, calendar: Calendar) -> str:
    """One-line statement of what an analytic did with the calendar."""
    if policy is CalendarPolicy.NATIVE:
        return (f"Computed on the asset's native {calendar} calendar "
                f"({calendar.periods_per_year} observations/year).")
    if policy is CalendarPolicy.INTERSECTION:
        return (f"Computed on dates common to every asset, annualised at "
                f"{calendar.periods_per_year}/year ({calendar}).")
    return f"Not supported for the {calendar} calendar."
