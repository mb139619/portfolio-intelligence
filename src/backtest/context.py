"""
Context — the only window a strategy has onto data.

This is where look-ahead prevention actually lives. Not in a code-review
convention, not in a comment telling the author to be careful: a `Context`
that contains data past its own timestamp cannot be constructed. The check
runs in `__post_init__`, so *every* Context in the system is truncated,
including ones a test builds by hand.

A strategy receives one of these and nothing else. It never gets a full-history
frame, never holds a reference to the raw dataset, and has no route back to the
store. There is no discipline to remember because there is no way to reach the
future.

Two things this module takes seriously that most backtesters do not
-------------------------------------------------------------------

**Publication lag is not the observation date.** Truncating at `t` is necessary
and not sufficient. The Fama-French library publishes weeks late — every
dashboard build in this repo reports 30-40 excluded trading days — so a factor
value *dated* before `t` may not have been *knowable* at `t`. Revised macro
series (CPI, GDP) have the same shape: today's number for 2019 is not the
number anyone had in 2019. `Context` records the lag that was applied so the
tearsheet can state it, and `available_through` reports the effective data
frontier, which is generally earlier than `t`.

**Derived quantities are stale on purpose.** Covariance, regime state and tail
metrics are expensive; refitting an HMM at every bar makes a backtest unusable,
and quietly reusing one fitted on the whole sample is precisely the bug this
class exists to prevent. So they are recomputed on the rebalance schedule and
carry `derived_as_of`, the date they were last computed from. Stale and honest
beats fresh and forward-looking, and the staleness is visible rather than
implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from src.analytics.risk.covariance import CovarianceResult
from src.analytics.risk.tail import EVTResult
from src.domain.calendar import Calendar
from src.domain.returns import ReturnSeries


class LookAheadError(AssertionError):
    """
    Raised when data beyond the decision date reaches a strategy.

    Deliberately an AssertionError: this is a broken invariant, not a
    recoverable condition. Nothing should ever catch it.
    """


@dataclass(frozen=True)
class RegimeState:
    """
    A point-in-time view of the market regime.

    Carries the fitting window rather than just the label, because a regime
    label with no provenance cannot be distinguished from one leaked out of a
    full-sample fit.
    """

    state: int
    label: str
    probability: float
    fitted_through: date          # last observation the fit was allowed to see
    n_states: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(
                f"regime probability must be in [0, 1], got {self.probability}"
            )
        if not 0 <= self.state < self.n_states:
            raise ValueError(
                f"state {self.state} outside 0..{self.n_states - 1}"
            )

    @property
    def is_stress(self) -> bool:
        """States are canonicalised by ascending volatility, so the last is stress."""
        return self.state == self.n_states - 1


@dataclass(frozen=True)
class Context:
    """
    Everything a strategy is allowed to know at time `t`.

    Note what is absent: no store, no ingestion pipeline, no full price history,
    no way to ask for another date. The surface is the whole guarantee.
    """

    t: date
    returns: ReturnSeries
    current_weights: dict[str, float]
    cash: float = 0.0

    # Derived quantities, refreshed on the rebalance schedule rather than
    # per bar. None means "not computed for this run", not "zero".
    covariance: CovarianceResult | None = None
    regime: RegimeState | None = None
    tail: EVTResult | None = None
    derived_as_of: date | None = None

    # Provenance, carried into RunMeta and the tearsheet header.
    publication_lag_days: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.returns.n_obs == 0:
            raise ValueError(f"Context at {self.t} has no observations")

        last = self.returns.dates.max()
        if last > self.t:
            raise LookAheadError(
                f"Context at t={self.t} carries returns through {last}. "
                f"The engine must truncate before constructing a Context; a "
                f"strategy receiving this could trade on unpublished data."
            )

        if self.derived_as_of is not None and self.derived_as_of > self.t:
            raise LookAheadError(
                f"Context at t={self.t} carries derived quantities computed "
                f"through {self.derived_as_of}. Covariance, regime and tail "
                f"estimates must be fitted on data at or before t."
            )

        if self.regime is not None and self.regime.fitted_through > self.t:
            raise LookAheadError(
                f"Context at t={self.t} carries a regime fitted through "
                f"{self.regime.fitted_through}."
            )

        if self.publication_lag_days < 0:
            raise ValueError(
                f"publication_lag_days must be >= 0, got {self.publication_lag_days}"
            )

        missing = [k for k in self.current_weights if k not in self.returns.tickers]
        if missing:
            raise ValueError(
                f"current_weights references {missing}, absent from the "
                f"context universe {self.returns.tickers}"
            )

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def universe(self) -> list[str]:
        """Assets with usable history at `t`. The only universe a strategy sees."""
        return list(self.returns.tickers)

    @property
    def n_obs(self) -> int:
        return self.returns.n_obs

    @property
    def calendar(self) -> Calendar:
        return self.returns.calendar

    @property
    def periods_per_year(self) -> int:
        return self.returns.periods_per_year

    @property
    def available_through(self) -> date:
        """
        The effective data frontier: `t` less the publication lag.

        Usually earlier than `t`, and that gap is the honest answer to "what
        did this strategy actually know?".
        """
        return self.t - timedelta(days=self.publication_lag_days)

    @property
    def derived_staleness_days(self) -> int | None:
        """How old the covariance/regime/tail estimates are, in calendar days."""
        if self.derived_as_of is None:
            return None
        return (self.t - self.derived_as_of).days

    def weight(self, ticker: str) -> float:
        """Current weight of one asset; 0.0 if not held."""
        return self.current_weights.get(ticker, 0.0)

    def __repr__(self) -> str:
        stale = self.derived_staleness_days
        return (
            f"Context(t={self.t}, universe={len(self.universe)}, "
            f"obs={self.n_obs}, calendar={self.calendar}, "
            f"derived_staleness={stale if stale is not None else 'n/a'}d)"
        )
