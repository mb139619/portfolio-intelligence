"""
Trading calendars.

The calendar is an attribute of the DATA, never an assumption in code. An
equity trades ~252 days a year; a crypto pair trades all 365. Every
annualisation factor, and every judgement about whether a gap in a series is
normal or an anomaly, derives from this.

The rule this module exists to enforce
-------------------------------------
**Never forward-fill one calendar onto another.** Filling equity weekends with
Friday's close manufactures two artificial zero returns per week. That is not a
cosmetic problem:

  * volatility is deflated by roughly 15% — the zeros enter the variance but
    carry no information;
  * correlations compress toward zero, because the artificial zeros are shared
    across every equity and dilute the real co-movement;
  * worst, a Markov-switching model learns a spurious low-volatility "weekend
    regime" and reports it as a market state.

The honest alternative is to intersect: keep only the dates on which every
asset genuinely traded, and say so. `ParquetStore.read_returns` already does
exactly that (an outer join followed by `drop_nulls`), so mixing a crypto pair
with equities yields the equity trading calendar and never a fabricated price.
"""

from __future__ import annotations

from enum import Enum


class Calendar(str, Enum):
    """How often an asset can possibly trade."""

    TRADING_DAYS = "trading_days"   # exchange hours, ~252 observations a year
    CONTINUOUS = "continuous"       # 24/7 markets, 365 observations a year

    @property
    def periods_per_year(self) -> int:
        """Annualisation factor. This is the only place 252 and 365 are decided."""
        return 252 if self is Calendar.TRADING_DAYS else 365

    @property
    def max_normal_gap_days(self) -> int:
        """
        The largest gap between consecutive observations that is still routine.

        Under TRADING_DAYS a three-day Friday-to-Monday gap is every weekend and
        four days is an ordinary long weekend, so anything up to four is
        unremarkable. Under CONTINUOUS there is no weekend: a gap of more than
        one day means an exchange outage or a delisted pair, and should be
        reported.
        """
        return 4 if self is Calendar.TRADING_DAYS else 1

    @classmethod
    def for_asset_class(cls, asset_class) -> Calendar:
        """
        Default calendar implied by an asset class. Crypto is continuous;
        everything else follows exchange hours. Explicit values on the domain
        object always win over this default.

        Accepts either an AssetClass member or a plain string. Note the
        `.value`: a `str`-based Enum still stringifies as "AssetClass.CRYPTO"
        rather than "crypto", so comparing str(...) directly silently never
        matches.
        """
        raw = getattr(asset_class, "value", asset_class)
        return cls.CONTINUOUS if str(raw).lower() == "crypto" else cls.TRADING_DAYS

    def __str__(self) -> str:
        return self.value


def resolve(calendars: list[Calendar]) -> Calendar:
    """
    The calendar a set of assets share once aligned on common dates.

    Intersecting a continuous series with a trading-day one leaves only the
    trading days, so the most restrictive calendar wins. Annualising the
    combined series at 365 would then overstate volatility by ~20%, which is
    why this is resolved from the data rather than assumed.
    """
    if not calendars:
        return Calendar.TRADING_DAYS
    return (Calendar.TRADING_DAYS if any(c is Calendar.TRADING_DAYS for c in calendars)
            else Calendar.CONTINUOUS)
