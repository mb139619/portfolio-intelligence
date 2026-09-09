# Backtesting

How the engine is put together, and how to write and evaluate a strategy
against it.

---

## The shape of the thing

Three types carry everything. Learn these and the rest follows.

```
Context          what a strategy is allowed to know at time t
Strategy         target weights in, nothing else
BacktestResult   the serialisable output everything downstream consumes
```

The dependency arrow runs one way and never bends:

```
report/ → backtest/ → analytics/optimization/ → analytics/ → store/ ← ingestion/
                                                     ↓
                                                  domain/
```

`backtest/` may read the risk engine. The risk engine may not know `backtest/`
exists. If a change seems to need the reverse, the abstraction is wrong.

---

## The loop, and why the order matters

One pass over dates. Within each bar:

1. the book earns today's return, using the weights it held **coming into**
   today — decided at some earlier bar;
2. those weights **drift**, because the legs moved by different amounts;
3. only then, if today is a rebalance date, the strategy is asked for a new
   target using data through today. That target earns **from tomorrow**.

Step 3 being last is the load-bearing part. Deciding on today's close and then
applying today's return to the new weights uses that return twice — once to
choose, once to profit. Every date involved is legitimately in the past, so no
amount of date-truncation catches it, and it is worth an enormous and entirely
fictional amount of alpha.

**How we know it holds.** A test strategy buys whichever asset already had the
best return today, run against a rigged series where exactly one asset gains
10% a day and which one alternates. A leaking engine compounds 10% daily and
ends near 1e16; this one ends at exactly 1.0, because it always holds
yesterday's winner. On real market data the same strategy would earn 981%
annually under a leak and earns 5.6% — a ratio of 0.006. A second test asserts
the rigged fixture *would* pay a leak, so the first cannot pass for the wrong
reason.

---

## Writing a strategy

The whole interface:

```python
class MyStrategy:
    name = "my_strategy"

    def target_weights(self, ctx: Context) -> dict[str, float]:
        return dict.fromkeys(ctx.universe, 1.0)   # engine normalises
```

No base class to inherit, no registration. Anything with a `name` and that
method satisfies the `Strategy` protocol.

### What `Context` gives you

| | |
|---|---|
| `ctx.t` | the decision date |
| `ctx.returns` | a `ReturnSeries` ending at `t`, no longer than the lookback |
| `ctx.universe` | tickers with usable history at `t` |
| `ctx.covariance` | `CovarianceResult` from the risk engine, or `None` |
| `ctx.regime` | `RegimeState` from filtered probabilities, or `None` |
| `ctx.tail` | `EVTResult`, or `None` |
| `ctx.current_weights` | what is held right now, after drift |
| `ctx.calendar`, `ctx.periods_per_year` | from the data, never assumed |
| `ctx.available_through` | `t` minus the publication lag |
| `ctx.derived_staleness_days` | how old the covariance/regime estimates are |

Note what is **absent**: no store, no pipeline, no way to request another date.
The surface is the guarantee, and a test asserts it stays this small.

### Rules

- Return **target weights, never orders**. Execution is a separate layer, which
  is what lets the same strategy be re-run under different cost assumptions.
- Do not implement a schedule. The engine decides *when* to call; you decide
  only *what*. That keeps the rebalance frequency an experimental variable.
- You may return a subset of the universe; absent assets are zero.
- Weights need not sum to 1 — the engine normalises — but must be finite and
  must not name assets outside `ctx.universe`. Both fail loudly.
- Expose a `params` property if the strategy is configurable. It lands in
  `RunMeta`, so two runs of the same strategy can be told apart later.

### Using the risk engine

An optimiser wraps in three lines, which is the evidence the layer boundary is
in the right place:

```python
from src.analytics.optimization import min_variance

class MinVar:
    name = "min_var"
    def target_weights(self, ctx):
        return min_variance(ctx.covariance).weights_dict()
```

Nothing is re-estimated. `ctx.covariance` was fitted by the risk engine on the
lookback window ending at `ctx.t`.

---

## Running one

From the command line:

```bash
python -m src.backtest
python -m src.backtest --rebalance quarterly --costs-bps 20
python -m src.backtest --tickers SPY,TLT,GLD --start 2018-01-01
python -m src.backtest --save runs/
```

From Python:

```python
from src.backtest import run, BacktestConfig, CostModel
from src.store.parquet_store import ParquetStore
from src.config import settings

rs = ParquetStore(settings.data_dir).read_returns(["SPY", "TLT"], start="2015-01-01")
result = run(MyStrategy(), rs, BacktestConfig(rebalance="monthly", lookback=252))
print(result.metrics["annualized_return"], result.metrics["turnover"])
result.save("runs/mine")
```

### Configuration

| Key | Default | Meaning |
|-----|---------|---------|
| `rebalance` | `monthly` | `daily`/`weekly`/`monthly`/`quarterly`/`annual` |
| `lookback` | 252 | estimation window handed to the strategy, in observations |
| `warmup` | = lookback | bars before the first decision |
| `costs` | 5bp + 5bp | **not** frictionless; `CostModel.free()` is explicit |
| `with_covariance` | `True` | estimate Σ at each rebalance |
| `cov_method` | `ledoit_wolf_cc` | any estimator the risk engine supports |
| `with_regime` | `False` | refit the HMM; needs a `market` series |
| `regime_refit` | `quarterly` | its own, slower schedule — see below |
| `publication_lag_days` | 0 | days before an observation is knowable |
| `allow_short` | `False` | negative weights are rejected unless enabled |

**On the regime cost.** Measured on 9,212 observations: `fit_regimes` takes
21.4s at `search_reps=20` — 34 minutes across 96 monthly rebalances — and 4.3s
at `search_reps=5`, with no further gain below 5 because the floor is the
single EM fit rather than the restarts. Covariance re-estimation is negligible
by comparison (0.1s for 96). Hence the separate, slower schedule.

---

## Point-in-time, in two parts

Truncating at `t` is necessary and not sufficient.

**Publication lag.** A value *dated* before `t` was not necessarily *knowable*
at `t`. The Fama-French library publishes in monthly batches: measured against
this repo's own store, factor data runs 35 days behind the price data. Sources
declare their own lag — 0 for daily closes, 1 for FRED/ECB, a deliberately
conservative 45 for the French factors — and it is recorded in `RunMeta`.

This is not the same as **execution lag**: knowing today's close does not mean
you could have traded at it. Conflating them smuggles in a bar of hindsight.

**Filtered, not smoothed.** The regime model exposes both. A smoothed
probability at time `s` conditions on the whole series, because the Kim
smoother runs backwards; for describing history that is correct, and the
dashboard uses it. As a signal it is invalid — the model has already seen the
crash it is meant to anticipate. Anything under `backtest/` uses
`filtered_probs`, and `probabilities_at()` defaults to it. See METHODOLOGY
§10.1 for the measured divergence.

---

## Testing a strategy

Correctness of the *harness* is already covered — those tests live in
`tests/unit/test_backtest_engine.py` and are not your problem. What you should
check about a *strategy*:

**Compare against a baseline that costs nothing to run.** Equal weight is hard
to beat after costs. If a strategy does not beat it on `ret/vol`, the
sophistication is not paying for itself.

**Read the turnover column before the return column.** A strategy earning 2%
more with ten times the turnover has probably found a cost, not an edge.

**Vary the costs.** `--costs-bps 0 / 10 / 50`. An edge that evaporates between
10 and 50bp is not an edge.

**Vary the rebalance frequency and the lookback.** A result that only exists at
one setting is a result about that setting.

**Be suspicious of good numbers.** A Sharpe that jumps when you change
something is a bug hypothesis before it is a finding.

---

## Worked examples

`examples/custom_strategies.py` is runnable:

```bash
python examples/custom_strategies.py
```

Five assets, 2015-2026, monthly rebalance, 10bp round trip:

| strategy | CAGR | vol | ret/vol | maxDD | turnover | costs |
|---|---|---|---|---|---|---|
| equal_weight | 12.02% | 11.82% | **1.02** | −25.97% | 4.05 | 0.41% |
| inverse_vol | 11.11% | 11.07% | 1.01 | −25.31% | 5.15 | 0.51% |
| momentum | **14.02%** | 15.83% | 0.91 | −23.31% | 49.71 | 4.97% |
| derisk_dd | 10.25% | 12.24% | 0.86 | −32.01% | 13.19 | 1.32% |
| regime_aware | 5.90% | 13.22% | 0.50 | −33.98% | 33.14 | — |

Two of these are included precisely because they do not work.

**Momentum** earns the highest return and the *lowest* ret/vol, at ten times
the turnover of equal weight and 4.97% paid away in costs. The headline number
is the least informative column in its row.

**`derisk_dd`** moves to bonds after a 10% drawdown. It produces a *worse*
maximum drawdown than doing nothing — it sells low and is not back in for the
recovery. A rule that sounds prudent and is not.

**`regime_aware`** is the sharpest lesson. Holding bonds during the model's
stress regime earns 5.90% with a 34% drawdown. The regime signal that looks
compelling on the dashboard is descriptive: fitted on the whole sample and read
through smoothed probabilities. Once it must be filtered and lagged by the 45
days the factor data actually takes to publish, it stops being tradeable. The
distance between those two numbers is the reason this project exists.

---

## Not built yet

- **Walk-forward.** `BacktestResult.folds` and `out_of_sample()` exist and
  `Fold` refuses overlapping train/test windows, but nothing generates folds
  yet. Until then every result here is in-sample.
- **Tearsheet.** Will be a second section producer feeding the existing
  `src/export/` pipeline, not a new reporting stack.
- **Risk parity and HRP** (Milestone A), which become reference strategies.
