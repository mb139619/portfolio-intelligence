"""
The backtest loop.

The whole engine is one pass over dates, and its correctness rests almost
entirely on the order of operations *within* a single bar. Getting that order
wrong is the most common way a backtest quietly earns a return it could not
have earned, and it survives review because the code looks obviously right.

The order here is:

    1. the book earns today's return, using the weights it held COMING INTO
       today — decided at some earlier bar;
    2. those weights drift, because the assets moved by different amounts;
    3. only then, if today is a rebalance date, the strategy is asked for a new
       target using data through today. The new weights earn from TOMORROW.

Step 3 last is the load-bearing part. Deciding on today's close and then
applying today's return to the new weights would use today's return twice: once
to choose, once to profit. That is a one-bar look-ahead, it is worth a large and
entirely fictional amount of alpha, and no amount of date-truncation catches it
because every date involved is legitimately in the past.

Costs are charged at the moment of trading, so a strategy that rebalances into a
position it immediately exits pays for the round trip. There is no frictionless
default: `CostModel()` charges unless explicitly told otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import polars as pl
from loguru import logger

from src.analytics.performance import compute_metrics
from src.analytics.risk.covariance import estimate_covariance
from src.backtest.context import Context, RegimeState
from src.backtest.execution import CostModel, drift_weights, rebalance
from src.backtest.result import BacktestResult, Fold, RunMeta
from src.backtest.strategy import Strategy, normalise_weights
from src.domain.returns import ReturnSeries

REBALANCE_FREQUENCIES = ("daily", "weekly", "monthly", "quarterly", "annual")


@dataclass(frozen=True)
class BacktestConfig:
    """Everything about a run that is not the strategy itself."""

    rebalance: str = "monthly"
    lookback: int = 252
    warmup: int | None = None          # defaults to `lookback`
    costs: CostModel = field(default_factory=CostModel)
    allow_short: bool = False
    initial_capital: float = 1.0
    # Σ|w| the book is scaled to. 1.0 means capital at risk equals
    # capital; a long/short strategy wanting 100% long against 100% short
    # asks for 2.0 explicitly rather than getting leverage by accident.
    gross_target: float = 1.0

    # Derived quantities. Covariance is cheap; the regime fit is not, so it
    # gets its own, slower schedule — see CLAUDE.md §3.
    with_covariance: bool = True
    cov_method: str = "ledoit_wolf_cc"
    with_regime: bool = False
    regime_refit: str = "quarterly"
    regime_states: int = 2
    regime_search_reps: int = 5        # 20 costs 5x more for no measured gain

    # Days before an observation dated D is knowable. Applied to the market
    # series behind the regime fit, which comes from the Fama-French library.
    publication_lag_days: int = 0

    def __post_init__(self) -> None:
        if self.rebalance not in REBALANCE_FREQUENCIES:
            raise ValueError(
                f"rebalance must be one of {REBALANCE_FREQUENCIES}, "
                f"got {self.rebalance!r}"
            )
        if self.regime_refit not in REBALANCE_FREQUENCIES:
            raise ValueError(f"regime_refit must be one of {REBALANCE_FREQUENCIES}")
        if self.lookback < 2:
            raise ValueError(f"lookback must be at least 2, got {self.lookback}")

    @property
    def effective_warmup(self) -> int:
        return self.warmup if self.warmup is not None else self.lookback


def _period_key(d: date, freq: str):
    if freq == "daily":
        return d
    if freq == "weekly":
        return d.isocalendar()[:2]
    if freq == "monthly":
        return (d.year, d.month)
    if freq == "quarterly":
        return (d.year, (d.month - 1) // 3)
    return d.year


def schedule(dates: list[date], freq: str) -> set[date]:
    """
    The dates on which a decision is taken: the LAST observation of each period.

    Deciding at the period end rather than the start means the decision uses a
    complete period of information, and it matches how a monthly rebalance is
    actually run.
    """
    if freq == "daily":
        return set(dates)
    last: dict = {}
    for d in dates:
        last[_period_key(d, freq)] = d
    return set(last.values())


class LookAheadGuard:
    """
    Counts what the engine handed to strategies, so the invariant can be
    asserted after a run rather than merely intended before one.
    """

    def __init__(self) -> None:
        self.contexts = 0
        self.max_date_seen: date | None = None

    def record(self, ctx: Context) -> None:
        self.contexts += 1
        last = ctx.returns.dates.max()
        if last > ctx.t:                      # belt and braces: Context also checks
            raise AssertionError(f"context at {ctx.t} saw {last}")
        if self.max_date_seen is None or last > self.max_date_seen:
            self.max_date_seen = last


def run(
    strategy: Strategy,
    returns: ReturnSeries,
    config: BacktestConfig | None = None,
    *,
    market: tuple[np.ndarray, list] | None = None,
    folds: list[Fold] | None = None,
) -> BacktestResult:
    """
    Run `strategy` over `returns` and produce a self-contained result.

    `market` is an optional (values, dates) pair for the broad-market series the
    regime model is fitted on — the Fama-French Mkt-RF factor in practice. It is
    passed in rather than fetched because the engine, like everything below it,
    never touches the store.
    """
    config = config or BacktestConfig()
    tickers = list(returns.tickers)
    dates: list[date] = returns.dates.to_list()
    R = returns.to_numpy()

    warmup = config.effective_warmup
    if len(dates) <= warmup:
        raise ValueError(
            f"{len(dates)} observations is not enough for a warmup of {warmup}. "
            f"Shorten the lookback or lengthen the window."
        )

    rebal_dates = schedule(dates, config.rebalance)
    regime_dates = schedule(dates, config.regime_refit) if config.with_regime else set()
    guard = LookAheadGuard()

    weights: dict[str, float] = {}
    equity = config.initial_capital
    prev_equity = equity
    started = False

    eq_rows, pos_rows, trade_rows = [], [], []
    regime_state: RegimeState | None = None

    for i, t in enumerate(dates):
        r_t = {tk: float(R[i, j]) for j, tk in enumerate(tickers)}

        # ---- 1. yesterday's book earns today's return --------------------
        if weights:
            port_ret = sum(weights.get(tk, 0.0) * r for tk, r in r_t.items())
            equity *= 1.0 + port_ret
            # ---- 2. and drifts, because the legs moved differently -------
            weights = drift_weights(weights, r_t)

        cost_today = 0.0

        # ---- 3. only now may the strategy look at today ------------------
        if i >= warmup - 1 and t in rebal_dates:
            if config.with_regime and market is not None and (
                regime_state is None or t in regime_dates
            ):
                regime_state = _fit_regime_at(t, market, config)

            ctx = _build_context(
                i, t, returns, R, tickers, weights, config, regime_state
            )
            guard.record(ctx)

            raw = strategy.target_weights(ctx)
            target = normalise_weights(
                raw, ctx.universe,
                strategy_name=strategy.name, allow_short=config.allow_short,
                gross_target=config.gross_target,
            )
            weights, fills, cost_today = rebalance(weights, target, config.costs)
            equity *= 1.0 - cost_today
            for f in fills:
                trade_rows.append({
                    "date": t, "ticker": f.ticker,
                    "delta_weight": f.delta_weight, "cost": f.cost,
                })
            started = True

        # Nothing is recorded before the first decision: a stretch of flat
        # pre-trade equity would drag every metric toward zero and is not part
        # of the strategy's record.
        if started:
            # Gross and net are recorded per bar rather than derived later:
            # they drift between rebalances, so a summary computed from the
            # target weights alone would describe a book nobody held.
            eq_rows.append({
                "date": t, "equity": equity,
                "ret": equity / prev_equity - 1.0 if prev_equity else 0.0,
                "gross": sum(abs(w) for w in weights.values()),
                "net": sum(weights.values()),
            })
            prev_equity = equity
            for tk, w in weights.items():
                pos_rows.append({"date": t, "ticker": tk, "weight": w})

    if not eq_rows:
        raise ValueError(
            "No rebalance date fell after the warmup; the run produced nothing."
        )

    equity_curve = pl.DataFrame(eq_rows)
    trades = pl.DataFrame(
        trade_rows,
        schema={"date": pl.Date, "ticker": pl.Utf8,
                "delta_weight": pl.Float64, "cost": pl.Float64},
    )
    positions = pl.DataFrame(
        pos_rows,
        schema={"date": pl.Date, "ticker": pl.Utf8, "weight": pl.Float64},
    )

    r = equity_curve["ret"].to_numpy()
    perf = compute_metrics(r, rf=0.0, ppy=returns.periods_per_year)
    metrics = perf.to_dict() | {
        "total_cost": float(trades["cost"].sum()) if len(trades) else 0.0,
        "turnover": float(trades["delta_weight"].abs().sum()) if len(trades) else 0.0,
        "n_rebalances": len(set(trades["date"].to_list())) if len(trades) else 0,
        "final_equity": float(equity_curve["equity"][-1]),
        "contexts_built": guard.contexts,
        "avg_gross_exposure": float(equity_curve["gross"].mean()),
        "max_gross_exposure": float(equity_curve["gross"].max()),
        "avg_net_exposure": float(equity_curve["net"].mean()),
        "min_net_exposure": float(equity_curve["net"].min()),
        "pct_days_with_shorts": float(
            (positions["weight"] < 0).mean() if len(positions) else 0.0
        ),
    }

    meta = RunMeta.create(
        strategy=strategy.name,
        universe=tickers,
        start=eq_rows[0]["date"],
        end=eq_rows[-1]["date"],
        strategy_params=getattr(strategy, "params", {}),
        calendar=str(returns.calendar),
        periods_per_year=returns.periods_per_year,
        publication_lag_days=config.publication_lag_days,
        rebalance=config.rebalance,
        costs=config.costs.to_dict(),
        estimation_windows={
            "lookback": config.lookback,
            "warmup": warmup,
            "cov_method": config.cov_method if config.with_covariance else None,
            "regime_refit": config.regime_refit if config.with_regime else None,
        },
    )

    logger.info(
        f"{strategy.name}: {len(equity_curve)} days, {metrics['n_rebalances']} "
        f"rebalances, turnover {metrics['turnover']:.2f}, "
        f"cost {metrics['total_cost']:.2%}, final equity {equity:.4f}"
    )
    return BacktestResult(equity_curve, positions, trades, metrics, meta, folds)


def _build_context(i, t, returns, R, tickers, weights, config, regime_state):
    """Slice the history at `t` and wrap it. The slice is the guarantee."""
    lo = max(0, i - config.lookback + 1)
    window = ReturnSeries(
        data=returns.data.slice(lo, i - lo + 1),
        tickers=tickers,
        is_log=returns.is_log,
        calendar=returns.calendar,
    )
    cov = (
        estimate_covariance(window, method=config.cov_method,
                            ppy=returns.periods_per_year)
        if config.with_covariance else None
    )
    return Context(
        t=t,
        returns=window,
        current_weights=weights,
        covariance=cov,
        regime=regime_state,
        derived_as_of=t,
        publication_lag_days=config.publication_lag_days,
    )


def _fit_regime_at(t, market, config) -> RegimeState | None:
    """
    Refit the regime model on market data knowable at `t`.

    Two truncations, both necessary. The publication lag comes off first —
    Fama-French factors dated within ~45 days of `t` had not been released — and
    then the FILTERED probability is read, never the smoothed one: the smoother
    conditions on the whole series and would hand the strategy the answer.
    """
    from src.analytics.regime.hmm import fit_regimes

    values, mkt_dates = market
    cutoff = t - timedelta(days=config.publication_lag_days)
    usable = [k for k, d in enumerate(mkt_dates) if d <= cutoff]
    if len(usable) < 250:
        return None

    hi = usable[-1] + 1
    model = fit_regimes(
        np.asarray(values[:hi]), mkt_dates[:hi],
        n_states=config.regime_states, search_reps=config.regime_search_reps,
    )
    idx = hi - 1
    probs = model.filtered_probs[idx]
    state = int(probs.argmax())
    return RegimeState(
        state=state,
        label=model.labels[state],
        probability=float(probs[state]),
        fitted_through=mkt_dates[idx],
        n_states=model.n_states,
    )
