"""
Run and compare strategies from the command line.

    python -m src.backtest
    python -m src.backtest --rebalance quarterly --costs-bps 20
    python -m src.backtest --tickers SPY,TLT,GLD --start 2018-01-01
    python -m src.backtest --save runs/

The header prints the assumptions before the results on purpose. Two backtests
are only comparable if their costs, rebalance schedule and estimation windows
match, and putting those numbers next to the Sharpe makes the comparison
honest rather than implied.

Note this module reads `portfolio.json` with plain `json`, rather than
importing `export.spec.PortfolioSpec`. `export/` is the presentation layer and
sits above `backtest/` in the dependency order; reaching up for a convenience
would invert the arrow the architecture rests on. All it needs is a list of
tickers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger

from src.backtest.engine import REBALANCE_FREQUENCIES, BacktestConfig, run
from src.backtest.execution import CostModel
from src.backtest.strategy import BuyAndHold, EqualWeight, MinimumVariance
from src.config import settings
from src.store.parquet_store import ParquetStore

ROOT = Path(__file__).resolve().parents[2]

STRATEGIES = {
    "buy_and_hold": BuyAndHold,
    "equal_weight": EqualWeight,
    "min_variance": MinimumVariance,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.backtest",
        description="Run strategies over the local price store and compare them.",
    )
    p.add_argument("--tickers", default="",
                   help="comma-separated; defaults to the portfolio.json universe")
    p.add_argument("--config", default=str(ROOT / "portfolio.json"),
                   help="portfolio file to take the universe from")
    p.add_argument("--strategies", default=",".join(STRATEGIES),
                   help=f"comma-separated from: {', '.join(STRATEGIES)}")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--rebalance", default="monthly", choices=REBALANCE_FREQUENCIES)
    p.add_argument("--lookback", type=int, default=252,
                   help="estimation window for covariance, in observations")
    p.add_argument("--costs-bps", type=float, default=10.0,
                   help="round-trip cost in basis points (0 for frictionless)")
    p.add_argument("--save", default="", help="directory to write each run into")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def universe_from(args) -> list[str]:
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    path = Path(args.config)
    if not path.exists():
        raise SystemExit(f"No --tickers given and {path} does not exist.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [p["ticker"] for p in raw.get("positions", [])]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    tickers = universe_from(args)
    names = [s.strip() for s in args.strategies.split(",") if s.strip()]
    unknown = [n for n in names if n not in STRATEGIES]
    if unknown:
        raise SystemExit(f"Unknown strategies {unknown}. Choose from {list(STRATEGIES)}")

    store = ParquetStore(settings.data_dir)
    try:
        rs = store.read_returns(tickers, start=args.start, end=args.end)
    except ValueError as e:
        raise SystemExit(f"{e}\nRun `python -m src.export` first to populate the store.")

    half = args.costs_bps / 2.0
    config = BacktestConfig(
        rebalance=args.rebalance,
        lookback=args.lookback,
        costs=CostModel(commission_bps=half, slippage_bps=half),
    )

    print(f"\nUniverse    {', '.join(rs.tickers)}")
    print(f"Window      {rs.dates.min()} -> {rs.dates.max()}  "
          f"({rs.n_obs} obs, {rs.periods_per_year}/yr, {rs.calendar})")
    print(f"Rebalance   {config.rebalance}, lookback {config.lookback} obs")
    print(f"Costs       {args.costs_bps:.1f}bp round trip"
          + ("   [FRICTIONLESS - not a baseline]" if args.costs_bps == 0 else ""))

    header = (f"\n{'strategy':<16}{'CAGR':>8}{'vol':>8}{'ret/vol':>9}"
              f"{'maxDD':>9}{'turnover':>10}{'costs':>8}{'trades':>8}")
    print(header)
    print("-" * len(header.strip()))

    results = {}
    for name in names:
        strategy = STRATEGIES[name]()
        try:
            result = run(strategy, rs, config)
        except ValueError as e:
            print(f"{name:<16}  failed: {e}")
            continue
        results[name] = result
        m = result.metrics
        print(f"{name:<16}{m['annualized_return']:>7.2%}"
              f"{m['annualized_volatility']:>8.2%}{m['sharpe_ratio']:>9.2f}"
              f"{m['max_drawdown']:>9.2%}{m['turnover']:>10.2f}"
              f"{m['total_cost']:>8.2%}{m['n_rebalances']:>8d}")

    if results:
        # ret/vol, not Sharpe: the risk-free rate is zero here, and calling it
        # Sharpe would overstate what the number is.
        print("\nret/vol is computed against a zero risk-free rate.")
        meta = next(iter(results.values())).run_meta
        if not meta.is_reproducible:
            print("Working tree is dirty: this run cannot be reproduced from its commit.")

    if args.save:
        out = Path(args.save)
        for name, result in results.items():
            result.save(out / name)
        print(f"\nSaved {len(results)} run(s) under {out}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
