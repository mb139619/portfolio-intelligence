"""
Walk-forward evaluation.

Splitting a backtest into train and test windows only means something if the
split *changes what happens*. Labelling one half of a single run "out of
sample" while the strategy was chosen by looking at the whole thing is
decoration, and it is the most common way a walk-forward chart lies.

So this harness makes the choice inside each fold: candidates are ranked on the
training window, the winner alone is carried into the test window, and only the
test segments are stitched into the reported curve. The selection therefore
never sees the data it is judged on. If one strategy wins every fold, that is
worth knowing; if the winner changes constantly, the selection rule is picking
up noise and the stitched result will show it.

Two schemes, and the difference matters:

  * **expanding** — training always starts at the beginning and grows. Uses all
    available history and implicitly assumes the distant past still informs the
    present.
  * **rolling** — training is a fixed window that slides. Adapts to a changing
    regime and throws away history to do it.

Neither is correct in general. Running both and finding they disagree is a
finding about the strategy, not a problem with the harness.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date

import numpy as np
import polars as pl
from loguru import logger

from src.backtest.engine import BacktestConfig, run
from src.backtest.result import BacktestResult, Fold, RunMeta
from src.backtest.strategy import Strategy
from src.domain.returns import ReturnSeries


def make_folds(
    dates: Sequence[date],
    n_folds: int = 5,
    scheme: str = "expanding",
    min_train: int = 252,
    train_window: int | None = None,
) -> list[Fold]:
    """
    Split a date range into contiguous, non-overlapping test windows.

    `min_train` observations are reserved before the first test window so the
    first fold is not judging a strategy that has barely any history. Under
    "rolling", `train_window` defaults to `min_train`.

    Folds are contiguous and cover the post-warmup range exactly once: no gaps
    (which would quietly drop the hardest periods) and no overlaps (which would
    score the same days twice).
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError(f"scheme must be 'expanding' or 'rolling', got {scheme!r}")
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    dates = list(dates)
    if len(dates) <= min_train + n_folds:
        raise ValueError(
            f"{len(dates)} observations cannot support {n_folds} folds after a "
            f"{min_train}-observation warmup. Shorten min_train or use fewer folds."
        )

    testable = dates[min_train:]
    edges = np.linspace(0, len(testable), n_folds + 1).astype(int)
    window = train_window or min_train

    folds: list[Fold] = []
    for i in range(n_folds):
        lo, hi = edges[i], edges[i + 1] - 1
        if hi < lo:
            continue
        test_start, test_end = testable[lo], testable[hi]
        train_end_idx = min_train + lo - 1
        train_end = dates[train_end_idx]
        train_start = (
            dates[0] if scheme == "expanding"
            else dates[max(0, train_end_idx - window + 1)]
        )
        folds.append(Fold(len(folds), train_start, train_end, test_start, test_end))
    return folds


def _segment(result: BacktestResult, start: date, end: date) -> pl.DataFrame:
    """The equity curve rows inside a window, rebased so the segment starts at 1."""
    seg = result.equity_curve.filter(
        (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    if seg.is_empty():
        return seg
    return seg.with_columns(
        (pl.col("equity") / pl.col("equity").first()).alias("equity")
    )


def _score(result: BacktestResult) -> float:
    """
    Selection criterion: return over volatility.

    Return alone would pick whatever was most levered to the last rally, and
    the point of choosing on a training window is to choose on something that
    might persist.
    """
    vol = result.metrics.get("annualized_volatility") or 0.0
    if vol <= 0:
        return float("-inf")
    return result.metrics.get("annualized_return", 0.0) / vol


def walk_forward(
    candidates: Sequence[Strategy] | Callable[[], Sequence[Strategy]],
    returns: ReturnSeries,
    config: BacktestConfig | None = None,
    *,
    n_folds: int = 5,
    scheme: str = "expanding",
    min_train: int = 252,
    train_window: int | None = None,
    score: Callable[[BacktestResult], float] = _score,
    market: tuple[np.ndarray, list] | None = None,
) -> BacktestResult:
    """
    Choose among `candidates` on each training window, evaluate out of sample.

    Returns a single `BacktestResult` whose equity curve is the chained test
    segments — the honest record — with the fold boundaries and the per-fold
    winner recorded in `metrics["fold_selection"]`.

    With one candidate this degenerates to an ordinary run with fold labels,
    which is a legitimate way to use it: the point is then simply that the
    reported curve contains no warm-up period.
    """
    config = config or BacktestConfig()
    pool = list(candidates() if callable(candidates) else candidates)
    if not pool:
        raise ValueError("walk_forward needs at least one candidate strategy")

    dates = returns.dates.to_list()
    folds = make_folds(dates, n_folds, scheme, min_train, train_window)

    eq_parts, pos_parts, trade_parts = [], [], []
    selection, used_folds, equity = [], [], 1.0

    for fold in folds:
        # --- choose on the training window only ---------------------------
        train = returns.trim(start=str(fold.train_start), end=str(fold.train_end))
        scored: list[tuple[float, Strategy]] = []
        for strategy in pool:
            try:
                scored.append((score(run(strategy, train, config, market=market)),
                               strategy))
            except ValueError as e:
                logger.debug(f"fold {fold.index}: {strategy.name} unusable — {e}")
        if not scored:
            logger.warning(f"fold {fold.index}: no candidate could be trained; skipped")
            continue
        best_score, winner = max(scored, key=lambda kv: kv[0])

        # --- evaluate it on the test window --------------------------------
        # The run starts at train_start so the strategy enters the test window
        # with history behind it, and only the test rows are kept.
        full = returns.trim(start=str(fold.train_start), end=str(fold.test_end))
        try:
            out = run(winner, full, config, market=market)
        except ValueError as e:
            logger.warning(
                f"fold {fold.index}: {winner.name} failed out of sample "
                f"— {e}"
            )
            continue

        seg = _segment(out, fold.test_start, fold.test_end)
        if seg.is_empty():
            continue

        seg = seg.with_columns((pl.col("equity") * equity).alias("equity"))
        equity = float(seg["equity"][-1])
        eq_parts.append(seg)

        pos_parts.append(out.positions.filter(
            (pl.col("date") >= fold.test_start) & (pl.col("date") <= fold.test_end)))
        trade_parts.append(out.trades.filter(
            (pl.col("date") >= fold.test_start) & (pl.col("date") <= fold.test_end)))

        used_folds.append(fold)
        selection.append({
            "fold": fold.index,
            "winner": winner.name,
            "train_score": round(best_score, 4),
            "candidates": {s.name: round(sc, 4) for sc, s in scored},
            "test_start": fold.test_start.isoformat(),
            "test_end": fold.test_end.isoformat(),
        })

    if not eq_parts:
        raise ValueError("Walk-forward produced no out-of-sample segments.")

    equity_curve = pl.concat(eq_parts).sort("date")
    positions = pl.concat(pos_parts).sort("date") if pos_parts else pl.DataFrame()
    trades = pl.concat(trade_parts).sort("date") if trade_parts else pl.DataFrame()

    from src.analytics.performance import compute_metrics

    r = equity_curve["ret"].to_numpy()
    perf = compute_metrics(r, rf=0.0, ppy=returns.periods_per_year)
    winners = [s["winner"] for s in selection]
    metrics = perf.to_dict() | {
        "total_cost": float(trades["cost"].sum()) if len(trades) else 0.0,
        "turnover": float(trades["delta_weight"].abs().sum()) if len(trades) else 0.0,
        "n_rebalances": len(set(trades["date"].to_list())) if len(trades) else 0,
        "final_equity": float(equity_curve["equity"][-1]),
        "n_folds": len(selection),
        "fold_selection": selection,
        # A winner that never changes is a stable rule; one that changes every
        # fold is a selection rule fitting noise.
        "selection_stability": (
            round(max(winners.count(w) for w in set(winners)) / len(winners), 3)
            if winners else 0.0
        ),
    }

    meta = RunMeta.create(
        strategy=f"walk_forward[{'|'.join(sorted({s.name for s in pool}))}]",
        universe=list(returns.tickers),
        start=equity_curve["date"][0],
        end=equity_curve["date"][-1],
        strategy_params={"scheme": scheme, "n_folds": n_folds,
                         "min_train": min_train, "train_window": train_window},
        calendar=str(returns.calendar),
        periods_per_year=returns.periods_per_year,
        publication_lag_days=config.publication_lag_days,
        rebalance=config.rebalance,
        costs=config.costs.to_dict(),
        estimation_windows={"lookback": config.lookback,
                            "warmup": config.effective_warmup},
    )

    logger.info(
        f"walk-forward ({scheme}, {len(selection)} folds): winners "
        f"{winners}, out-of-sample equity {equity:.4f}"
    )
    # Only the folds that actually produced a segment: a fold whose candidates
    # all failed must not appear as a boundary on a chart it contributed
    # nothing to.
    return BacktestResult(equity_curve, positions, trades, metrics, meta, used_folds)
