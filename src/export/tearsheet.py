"""
Tearsheet — a second section producer for the existing export pipeline.

    python -m src.export.tearsheet runs/equal_weight --serve

Deliberately not a new reporting stack. `src/export/` already does exactly what
a tearsheet needs: a producer emits a JSON payload, a renderer that computes
nothing draws it, and `--standalone` inlines everything into one file. The SPA
shell knows the payload schema and nothing about finance, so it renders these
sections as readily as the portfolio ones.

The rule that keeps it honest: **this module computes nothing.** Every number
comes out of `BacktestResult.metrics`, every chart from `viz.plots`. A missing
metric is added to the result object, never calculated here — otherwise the
report and the saved run drift apart and only the report is ever read.

Reads a run from disk rather than taking one in memory, which is the point of
`BacktestResult.save()`: a result can be re-rendered months later, from the
commit that produced it, without rerunning anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.backtest.result import BacktestResult
from src.export import encode
from src.viz import plots

ROOT = Path(__file__).resolve().parents[2]


# Assumptions and warnings are kept apart on purpose. Assumptions describe how
# the run was configured and belong beside the numbers they qualify; warnings
# are reasons to distrust the run and belong at the top, where they cannot be
# scrolled past. Emitting the same text in both places, as an earlier version
# did, trains the reader to ignore the banner.

def _assumptions(result: BacktestResult) -> list[str]:
    """How the run was configured. Shown as section notes."""
    meta = result.run_meta
    bps = (meta.costs.get("commission_bps", 0)
           + meta.costs.get("slippage_bps", 0))
    notes = [
        f"Rebalanced {meta.rebalance}; costs {bps:.0f}bp round trip; "
        f"annualised at {meta.periods_per_year}/yr ({meta.calendar}).",
    ]
    if meta.publication_lag_days:
        notes.append(
            f"Inputs were lagged by {meta.publication_lag_days} days before the "
            f"strategy saw them, so the run uses what was published by each "
            f"decision date rather than what was merely dated before it."
        )
    else:
        notes.append(
            "No publication lag was applied: every input is assumed knowable on "
            "the date it carries. Fine for prices, wrong for factor data."
        )
    return notes


def _warnings(result: BacktestResult) -> list[str]:
    """Reasons to distrust the run. Shown in the banner."""
    out = []
    if not result.run_meta.is_reproducible:
        out.append(
            "The working tree was modified when this ran, so the result cannot "
            "be reproduced from its commit. Treat it as provisional."
        )
    if not result.folds:
        out.append(
            "No walk-forward folds: every number here is IN SAMPLE. The strategy "
            "was evaluated on the same data any choices about it were made on."
        )
    stability = result.metrics.get("selection_stability")
    if stability is not None and stability < 0.7:
        out.append(
            f"Walk-forward selection stability is {stability:.0%}: the winning "
            f"strategy changed across folds, so the stitched curve blends "
            f"different rules rather than testing one."
        )
    return out


def build_overview(result: BacktestResult) -> dict:
    m = result.metrics
    meta = result.run_meta
    oos = " (out of sample)" if result.folds else " (in sample)"
    shape = "long/short" if result.is_long_short else "long-only"

    return encode.section(
        "overview", "Performance",
        subtitle=f"{meta.strategy} · {shape} · {meta.start} → {meta.end}{oos}",
        stats=[
            encode.stat("Ann. return", m.get("annualized_return"), "percent",
                        "Geometric"),
            encode.stat("Ann. volatility", m.get("annualized_volatility"), "percent"),
            encode.stat("Return / volatility", m.get("sharpe_ratio"), "ratio",
                        "Zero risk-free rate"),
            encode.stat("Sortino", m.get("sortino_ratio"), "ratio"),
            encode.stat("Max drawdown", m.get("max_drawdown"), "percent", tone="bad"),
            encode.stat("Calmar", m.get("calmar_ratio"), "ratio"),
            encode.stat("Hit rate", m.get("hit_rate"), "percent",
                        "Share of positive days"),
            encode.stat("Final equity", m.get("final_equity"), "ratio",
                        "Growth of 1"),
            encode.stat("Observations", m.get("n_observations"), "integer"),
            encode.stat("Commit", (meta.git_commit or "unknown")[:12], "text",
                        "dirty tree" if meta.git_dirty else "clean tree",
                        tone="bad" if meta.git_dirty else "good"),
        ],
        figures=[
            encode.figure(
                plots.plot_backtest_equity(result), "equity", "Equity and drawdown",
                "Shaded bands mark alternate walk-forward test windows."
                if result.folds else
                "No folds: this curve is in sample end to end.",
            ),
        ],
        notes=_assumptions(result),
    )


def build_risk(result: BacktestResult) -> dict:
    r = result.returns
    m = result.metrics
    ppy = result.run_meta.periods_per_year
    window = min(252, max(20, len(r) // 4))

    figures = [
        encode.figure(
            plots.plot_rolling_sharpe(r, result.dates, window=window, ppy=ppy),
            "rolling_sharpe", "Was the edge steady?",
            "A good full-sample number built from one stretch is a fact about "
            "that stretch.",
        ),
    ]
    if len(r) > 30:
        figures.append(encode.figure(
            plots.plot_var_distribution(r, confidence=0.99),
            "distribution", "Return distribution and VaR",
            "The gap between the Gaussian marker and the historical one is the "
            "cost of assuming normality.",
        ))

    return encode.section(
        "risk", "Risk",
        subtitle="What the headline numbers average away",
        stats=[
            encode.stat("VaR 95% (1d)", m.get("var_95"), "percent", "Historical"),
            encode.stat("CVaR 95% (1d)", m.get("cvar_95"), "percent",
                        "Mean loss beyond VaR"),
            encode.stat("Skewness", m.get("skewness"), "number"),
            encode.stat("Excess kurtosis", m.get("excess_kurtosis"), "number",
                        "0 = Gaussian tails"),
            encode.stat("Drawdown duration", m.get("max_drawdown_duration"),
                        "integer", "Longest run below the previous peak"),
            encode.stat("Downside deviation", m.get("downside_deviation"), "percent"),
        ],
        figures=figures,
    )


def build_trading(result: BacktestResult) -> dict:
    m = result.metrics
    trades = result.trades

    tables = []
    if len(trades):
        by_asset = (
            trades.group_by("ticker")
            .agg([pl.col("delta_weight").abs().sum().alias("turnover"),
                  pl.col("cost").sum().alias("cost"),
                  pl.len().alias("fills")])
            .sort("turnover", descending=True)
        )
        tables.append(encode.table(
            "by_asset", "Trading by asset", by_asset,
            formats={"turnover": "number", "cost": "percent", "fills": "integer"},
            note="Turnover is traded notional as a fraction of book value, "
                 "summed across the run.",
        ))

    cost_drag = None
    ann = m.get("annualized_return")
    if ann is not None and m.get("total_cost") and m.get("n_observations"):
        years = m["n_observations"] / result.run_meta.periods_per_year
        cost_drag = m["total_cost"] / years if years > 0 else None

    stats = [
        encode.stat("Total turnover", m.get("turnover"), "number",
                    "Traded notional, whole run"),
        encode.stat("Total cost", m.get("total_cost"), "percent", tone="bad"),
        encode.stat("Cost drag", cost_drag, "percent", "Per year", tone="bad"),
        encode.stat("Rebalances", m.get("n_rebalances"), "integer"),
    ]
    # Exposure tiles only earn their place on a long/short book. On a
    # fully invested long-only run gross is 1 and net is 1 by construction,
    # and a tile that always reads the same number teaches the reader to stop
    # looking at tiles.
    if result.is_long_short:
        stats += [
            encode.stat("Avg gross exposure", m.get("avg_gross_exposure"),
                        "number", "Σ|w| — capital actually at risk"),
            encode.stat("Max gross exposure", m.get("max_gross_exposure"),
                        "number", "Peak between rebalances",
                        tone="bad" if (m.get("max_gross_exposure") or 0) > 2
                        else "neutral"),
            encode.stat("Avg net exposure", m.get("avg_net_exposure"), "number",
                        "Σw — directional tilt; ~0 is market-neutral"),
            encode.stat("Days holding shorts", m.get("pct_days_with_shorts"),
                        "percent"),
        ]

    return encode.section(
        "trading", "Trading",
        subtitle=("What the strategy paid to exist"
                  + (" — and how much of it was short"
                     if result.is_long_short else "")),
        stats=stats,
        figures=[
            encode.figure(plots.plot_turnover(trades), "turnover",
                          "Turnover and cumulative cost"),
            encode.figure(plots.plot_exposure(result.positions), "exposure",
                          "What was actually held",
                          "Often less diversified than the description implies."),
        ],
        tables=tables,
        notes=[
            "Read this page before the performance page. A strategy whose return "
            "edge is smaller than its trading bill has not found anything.",
        ] + ([
            "Gross exposure is what is actually at risk; net is the directional "
            "tilt. They drift between rebalances, so these are measured per bar "
            "rather than read off the target weights.",
            "Borrow cost is NOT modelled. On a book that holds shorts "
            f"{(m.get('pct_days_with_shorts') or 0):.0%} of the time, the "
            "financing charge is a real cost this backtest omits, so the "
            "result here is optimistic by that amount.",
        ] if result.is_long_short else []),
    )


def build_walkforward(result: BacktestResult) -> dict:
    if not result.folds:
        return encode.section(
            "walkforward", "Walk-forward",
            error="This run has no folds, so there is no out-of-sample record. "
                  "Re-run through walk_forward() to produce one.",
        )

    selection = result.metrics.get("fold_selection") or []
    rows = []
    for fold in result.folds:
        pick = next((s for s in selection if s.get("fold") == fold.index), {})
        rows.append({
            "fold": fold.index,
            "train_start": str(fold.train_start),
            "train_end": str(fold.train_end),
            "test_start": str(fold.test_start),
            "test_end": str(fold.test_end),
            "winner": pick.get("winner", "—"),
            "train_score": pick.get("train_score"),
        })

    winners = [r["winner"] for r in rows]
    stability = result.metrics.get("selection_stability")

    notes = [
        "Candidates are ranked on each training window and only the winner is "
        "carried into the test window, so the selection never sees the data it "
        "is judged on. The curve is the chained test segments.",
    ]
    if stability is not None and stability < 0.7:
        notes.append(
            f"The winner changed across folds (stability {stability:.0%}). A "
            f"selection rule that keeps changing its mind is usually fitting "
            f"noise, and the stitched result carries that instability."
        )

    return encode.section(
        "walkforward", "Walk-forward",
        subtitle=f"{len(result.folds)} folds, selection made in sample only",
        stats=[
            encode.stat("Folds", len(result.folds), "integer"),
            encode.stat("Selection stability", stability, "percent",
                        "Share of folds won by the most frequent choice",
                        tone="bad" if (stability or 1) < 0.7 else "neutral"),
            encode.stat("Most chosen", max(set(winners), key=winners.count)
                        if winners else "—", "text"),
        ],
        tables=[encode.table("folds", "Fold boundaries and winners", rows,
                             formats={"fold": "integer", "train_score": "number"})],
        notes=notes,
    )


SECTIONS = [
    ("overview", "Performance", build_overview),
    ("risk", "Risk", build_risk),
    ("trading", "Trading", build_trading),
    ("walkforward", "Walk-forward", build_walkforward),
]


def build_payload(result: BacktestResult) -> dict:
    """Assemble the tearsheet payload from a saved run."""
    import datetime as dt

    from loguru import logger

    sections = []
    for sid, title, fn in SECTIONS:
        try:
            sections.append(fn(result))
        except Exception as e:                       # noqa: BLE001
            logger.opt(exception=True).error(f"section {sid} failed: {e}")
            sections.append(encode.section(sid, title,
                                           error=f"{type(e).__name__}: {e}"))

    meta = result.run_meta
    return encode.safe({
        "meta": {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "portfolio": {
                "name": f"{meta.strategy}",
                "description": f"Backtest · {meta.rebalance} rebalance",
                "positions": [{"ticker": t, "name": t, "asset_class": "",
                               "weight": np.nan} for t in meta.universe],
                "cov_method": meta.estimation_windows.get("cov_method") or "—",
            },
            "coverage": {
                "start": meta.start, "end": meta.end,
                "observations": result.metrics.get("n_observations", 0),
            },
            "warnings": _warnings(result),
        },
        "sections": sections,
    })


def main(argv: list[str] | None = None) -> int:
    from src.export.__main__ import DEFAULT_OUT, serve, write_standalone
    from src.export.encode import dumps

    p = argparse.ArgumentParser(
        prog="python -m src.export.tearsheet",
        description="Render a saved backtest run as a static report.",
    )
    p.add_argument("run", help="directory written by BacktestResult.save()")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--serve", action="store_true")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--standalone", action="store_true")
    args = p.parse_args(argv)

    run_dir = Path(args.run)
    if not (run_dir / "run.json").exists():
        raise SystemExit(f"{run_dir} does not look like a saved run (no run.json).")

    result = BacktestResult.load(run_dir)
    payload_json = dumps(build_payload(result))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(payload_json, encoding="utf-8")
    print(f"Tearsheet for {result.run_meta.strategy} -> {out} "
          f"({len(payload_json.encode('utf-8')) / 1e6:.2f} MB)")

    if args.standalone:
        write_standalone(payload_json, out.with_suffix(".html"))
        print(f"Standalone -> {out.with_suffix('.html')}")
    if args.serve:
        serve(args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
