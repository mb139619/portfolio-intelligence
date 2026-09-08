# Backtesting Engine — Architecture Brief

> Append this section to the existing `CLAUDE.md` at repo root. It extends the
> current architecture; it does not replace any existing invariant.

## Scope and positioning

The repository now covers two surfaces: a **risk/analytics platform** and a
**strategy backtester**. The backtester is the primary deliverable going
forward; the risk engine is infrastructure it consumes.

Repo remains single. Do **not** split into a second repository. Module
boundaries, not repository boundaries, provide the separation.

Rename in README/docs to reflect both surfaces (e.g. "Portfolio Backtesting &
Risk Platform"). Package name and import paths stay unchanged to avoid churn.

## Module boundaries

```
data/        multi-source registry (Yahoo, FRED, ECB SDMX, French, CCXT)
risk/        HMM regimes, EVT/GPD, MST, HAC t-stats
optimize/    mean-variance, risk parity, HRP   (Phase 4)
backtest/    NEW — engine, strategies, execution, results
report/      NEW — tearsheet rendering
```

**Dependency direction is strictly one-way:**

```
report/  →  backtest/  →  optimize/  →  risk/  →  data/
```

`risk/`, `data/` and `optimize/` MUST NOT import from `backtest/` or
`report/`. If a change appears to require a reverse import, stop and flag it —
it means the abstraction is wrong.

## Core contracts

These three types are the load-bearing interfaces. Define them before writing
engine internals.

### `Context` — the only window onto data

```python
@dataclass(frozen=True)
class Context:
    t: pd.Timestamp
    prices: pd.DataFrame        # rows strictly <= t
    covariance: pd.DataFrame | None
    regime: int | None          # current HMM state, if available
    tail_metrics: TailMetrics | None
    current_weights: pd.Series
    cash: float
```

Look-ahead prevention is **structural, not by convention**. `Context` is
constructed by the engine with data already truncated at `t`. Strategies never
receive a full-history frame and never hold a reference to the raw dataset.

### `Strategy` — signal only

```python
class Strategy(Protocol):
    name: str
    def target_weights(self, ctx: Context) -> pd.Series: ...
```

- Returns **target weights**, never orders. Weights → trades → fills is the
  execution layer's job.
- Strategies are stateless with respect to scheduling. The engine decides
  *when* to call; the strategy decides only *what*.
- Phase 4 optimizers wrap trivially: a risk-parity strategy calls the
  optimizer on `ctx.covariance` and returns the result.

### `BacktestResult` — serializable output contract

```python
@dataclass
class BacktestResult:
    equity_curve: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict
    folds: list[Fold] | None      # walk-forward boundaries
    run_meta: RunMeta             # see below
```

Everything downstream (tearsheet, any future UI) consumes **only** this
object. Must round-trip to disk (Parquet/JSON) so a run can be reloaded and
re-rendered without recomputation.

`RunMeta` carries reproducibility: git commit hash, timestamp, universe,
date range, strategy params, cost/slippage assumptions, data snapshot ids.

## Execution model

Separate from strategy logic:

- Explicit transaction costs and slippage — no frictionless default.
- Rebalance scheduler external to the strategy (calendar- or drift-triggered).
- Turnover tracked and reported as a first-class metric.
- Partial/no-fill handling stubbed but explicit, not silently assumed away.

## Walk-forward

In-sample and out-of-sample must be distinguishable in the result object and
in every rendered output. A single undifferentiated equity curve is not an
acceptable deliverable. Record fold boundaries in `BacktestResult.folds`.

## Tearsheet

`report/` exposes `render(result: BacktestResult) -> str` (single-file HTML).

- **Pure function of `BacktestResult`.** No data access, no metric computation
  inside the renderer. A missing metric is added to `BacktestResult`, never
  calculated in the template.
- Jinja2 + matplotlib, figures inlined as base64. Single self-contained file,
  no external assets, openable offline.
- Required blocks: reproducibility header (`RunMeta`), equity curve with fold
  boundaries marked, drawdown, rolling Sharpe, return distribution, turnover
  and exposure over time, metrics table (Sharpe, Sortino, max DD, Calmar, hit
  rate, total costs).
- Risk-engine overlays where available: HMM regime band beneath the equity
  curve, EVT tail metrics in the metrics table.

## Calendar policy (inherited)

The existing Native / Intersection / Unsupported declaration applies unchanged
to backtests. A backtest spanning asset classes with differing calendars must
declare its policy in `RunMeta` and surface it in the tearsheet header.
Never forward-fill weekends.

## Invariants — stop and reconsider if violated

1. A strategy can access data beyond `ctx.t`.
2. `risk/`, `data/` or `optimize/` needs to import from `backtest/`.
3. The tearsheet computes a metric not present in `BacktestResult`.
4. Adding a strategy requires editing engine internals.
5. Backtest integration requires scattered edits across analytics modules
   (existing additive-only constraint).

## Build order

1. `Context`, `Strategy`, `BacktestResult` type definitions + tests
2. Engine loop with look-ahead assertions in the test suite
3. Execution layer (costs, slippage, turnover)
4. Two reference strategies: equal-weight baseline, risk parity from Phase 4
5. Walk-forward harness
6. Tearsheet renderer

Strategy sophistication is deliberately deferred. Backtest rigor is the
product; the strategy is the demonstration vehicle.
