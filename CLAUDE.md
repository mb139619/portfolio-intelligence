# CLAUDE.md

Working brief for Claude Code on `portfolio-intelligence`.

Read this before writing code. When a request conflicts with the invariants
below, say so instead of silently working around them.

---

## 1. What this project is

A **strategy backtesting and research platform**, built on free public data,
whose distinguishing claim is *rigour rather than returns*. It answers whether a
strategy's result is real: was it available at the time, does it survive costs,
does it hold out of sample, and where does its risk actually come from.

The repository covers two surfaces:

- **`backtest/` + `report/`** — the primary deliverable. Point-in-time engine,
  execution modelling, walk-forward evaluation, tearsheets.
- **the risk engine** (`domain/`, `store/`, `ingestion/`, `analytics/`,
  `data_quality/`, `viz/`, `export/`) — infrastructure the backtester consumes.
  Factor exposures, latent structure, hidden concentration, correlation
  topology, tail behaviour, regimes.

The risk engine is not legacy. It is what makes the backtester worth more than
a for-loop over prices: a strategy result here arrives with its factor
attribution, its regime-conditional behaviour and its tail risk attached.

**Strategy sophistication is deliberately deferred.** Backtest rigour is the
product; the strategies are the demonstration vehicle. A well-instrumented
equal-weight baseline is worth more here than an elaborate signal with a
sloppy harness.

> **Scope history.** Earlier versions of this brief declared backtesting out of
> scope and destined for a separate repository. That was decided before the
> risk engine existed. It has since been reversed deliberately: the crypto
> milestone demonstrated that the module boundaries genuinely hold (see §4), so
> module boundaries — not repository boundaries — provide the separation. Do
> not split the repo.

---

## 2. Architecture invariants

These are load-bearing. Do not break them to make a feature fit.

1. **Separation of concerns** — data, analytics, backtesting and presentation
   are fully decoupled.
2. **Parquet-first storage** — data lives in Parquet; DuckDB is a stateless
   query engine over those files, never a persistent database.
3. **Pure analytics** — every analytic is a pure function: arrays in, results
   out. No hidden state, no side effects, no I/O inside analytics.
4. **Unidirectional dependencies** — strictly one way:

   ```
   report/ → backtest/ → analytics/optimization/ → analytics/ → store/ ← ingestion/
                                                        ↓
                                                     domain/
   ```

   `domain/`, `store/`, `ingestion/` and `analytics/` MUST NOT import from
   `backtest/` or `report/`. If a change appears to need a reverse import, stop
   and flag it: the abstraction is wrong.
5. **Explainable & decomposable** — every number traces back to its drivers.
6. **Point-in-time by construction** — see §3. This is the invariant the whole
   project is selling.

### Practical consequences

- Analytics never fetch data. They receive it.
- New data sources are **registry entries**, not new plumbing (see
  `ingestion/prices.py` and `ingestion/rates.py`). If adding a source requires
  touching analytics code, the abstraction has failed — stop and report it.
- No `if asset_class == "crypto"` branches inside analytics. Behaviour that
  varies by asset class belongs to metadata on the domain object.
- A new chart belongs in `viz/plots.py`, where it is unit-tested and both the
  notebook and the report can reuse it — never inline in a rendering layer.
- Presentation renders; it does not compute. If a page needs a number that does
  not exist yet, the number belongs in the payload's producer, not the template.

---

## 3. Point-in-time discipline

The single most important section in this document. A backtest that quietly
sees the future is worthless, and every plausible-looking equity curve should be
assumed guilty until the harness proves otherwise.

### Structural, not conventional

Look-ahead prevention is enforced by **construction**, not by reviewer
attention. The engine builds a `Context` whose data is already truncated at `t`
and hands that to the strategy. A strategy never receives a full-history frame
and never holds a reference to the raw dataset. There is no discipline to
remember because there is no way to reach the future.

The test suite asserts this directly: strategies are run against instrumented
contexts that fail if anything beyond `t` is touched.

### Publication lag is not the same as date

Truncating on the observation date is **not sufficient**, and this is the
subtle failure mode.

The Fama-French library publishes with weeks of lag — every dashboard build
reports it, typically 30-40 excluded trading days. A backtest standing at `t`
must see only what was *published* by `t`, not everything *dated* before `t`.
The same applies to any revised macro series (CPI, GDP) where the number
available at the time differs from today's revised value.

Every data source therefore declares its publication lag, and `Context`
applies it. Where the true lag is unknown, use a conservative constant and say
so in `RunMeta` rather than assuming zero.

### Recomputation policy

`Context` carries expensive derived quantities — covariance, regime state, tail
metrics. **When these are recomputed is a modelling decision and must be
explicit**, because it is where look-ahead most easily survives the truncation
discipline:

- Recomputed on the **rebalance schedule**, not every bar. A 2,000-day backtest
  refitting an HMM at every step is unusable, and quietly reusing one fitted on
  the full sample is exactly the bug this project exists to avoid.
- Each carries the **estimation window** that produced it, recorded in
  `RunMeta`.
- Between refits the strategy sees the last value computed strictly from data
  at or before its refit date. Stale is acceptable and honest; forward-looking
  is not.

---

## 4. Conventions

- Python, `src/` layout, editable install via `pyproject.toml`.
- **Polars, not pandas.** The store, `ReturnSeries` and every analytic are
  polars-first. The backtest layer follows: converting at each boundary would
  drift the project toward pandas without anyone deciding to.
- Tests with `pytest`, mirroring the `src/` structure.
- Type hints on all public functions.
- Methodology and model assumptions go in `docs/METHODOLOGY.md`; usage examples
  in `docs/USAGE.md`; the report layer in `docs/DASHBOARD.md`. Keep them
  current — a feature is not done until its assumptions and limitations are
  written down.
- Every model gets a short note on **where it stops being valid**. This matters
  as much as the implementation.
- CI runs `ruff` and `pytest` on 3.11 and 3.12. Both must stay green; the lint
  gate is only meaningful because it currently passes.

### Commands

The virtualenv is `.venv` and is not always on PATH; call it explicitly.

```bash
.venv/Scripts/python.exe -m pytest -q                  # full suite (~20s)
.venv/Scripts/python.exe -m ruff check src/ tests/     # line-length 88, E/F/I/UP
.venv/Scripts/python.exe -m src.export --serve         # build + open the dashboard
.venv/Scripts/python.exe -m src.export --skip-ingest --only risk   # fast iteration
```

The export CLI is documented in `docs/DASHBOARD.md`.

### Environment quirks

Do not spend time diagnosing these; they are known and external.

- **FRED times out** on this network (`USD_FEDFUNDS`, `USD_10Y`). Harmless: the
  risk-free rate used by the analytics comes from the French `RF` factor, not
  from FRED. ECB and the French library work fine.
- **Yahoo rate-limits** unpredictably. Ingestion is incremental, so re-running
  usually clears it.
- **The French library publishes with a lag**, so factor data ends weeks before
  price data. This is a data quirk *and* a modelling constraint — see §3.
- Console output is cp1252 on Windows; set `PYTHONIOENCODING=utf-8` before
  printing anything with arrows or Greek letters.
- `gh` is not installed; CI results must be checked in the browser.

---

## 5. Backtester architecture

### Core contracts

These three types are the load-bearing interfaces. Define them, and their
tests, before writing any engine internals.

**`Context` — the only window onto data.**

```python
@dataclass(frozen=True)
class Context:
    t: date
    returns: ReturnSeries          # rows strictly <= t, publication lag applied
    covariance: CovarianceResult | None
    regime: RegimeState | None
    tail: EVTResult | None
    current_weights: dict[str, float]
    cash: float
```

Constructed by the engine, never by a strategy. Reuses the existing domain and
analytics result types rather than introducing parallel ones.

**`Strategy` — signal only.**

```python
class Strategy(Protocol):
    name: str
    def target_weights(self, ctx: Context) -> dict[str, float]: ...
```

Returns **target weights, never orders**; weights → trades → fills is the
execution layer's job. Strategies are stateless with respect to scheduling: the
engine decides *when* to call, the strategy decides only *what*. Optimisers wrap
trivially — a risk-parity strategy calls the optimiser on `ctx.covariance` and
returns the result.

The universe is whatever `ctx.returns` contains at `t`; a strategy must handle
assets with no history yet and assets that have disappeared, and the engine
renormalises whatever it returns.

**`BacktestResult` — serialisable output contract.**

```python
@dataclass
class BacktestResult:
    equity_curve: pl.DataFrame
    positions: pl.DataFrame
    trades: pl.DataFrame
    metrics: dict
    folds: list[Fold] | None       # walk-forward boundaries
    run_meta: RunMeta
```

Everything downstream consumes **only** this object, and it must round-trip to
disk so a run can be reloaded and re-rendered without recomputation.

`RunMeta` carries reproducibility: git commit hash, timestamp, universe, date
range, strategy parameters, cost and slippage assumptions, estimation windows,
publication lags and calendar policy.

### Execution model

Kept strictly separate from strategy logic:

- Explicit transaction costs and slippage — **no frictionless default**.
- Rebalance scheduler external to the strategy (calendar- or drift-triggered).
- Turnover tracked and reported as a first-class metric.
- Partial and failed fills stubbed but explicit, never silently assumed away.

### Walk-forward

In-sample and out-of-sample must be distinguishable in the result object and in
every rendered output. A single undifferentiated equity curve is not an
acceptable deliverable. Fold boundaries live in `BacktestResult.folds` and are
drawn on the equity curve.

### Reporting — extend, do not duplicate

`src/export/` already implements the exact pattern a tearsheet needs: a Python
producer emits a JSON payload, and a renderer that computes nothing draws it,
with a `--standalone` mode inlining everything into one self-contained HTML
file. `encode.py` (payload schema, NaN/infinity handling) and the SPA shell in
`web/` are generic — the shell knows the schema and nothing about finance.

Therefore a tearsheet is **a second section producer feeding the existing
renderer**, not a new reporting stack. Do not introduce Jinja2 or matplotlib
alongside the existing Plotly pipeline.

- New charts go in `viz/plots.py`, like every other chart.
- Backtest-specific sections go in their own producer module, consuming only
  `BacktestResult`.
- A missing metric is added to `BacktestResult`; it is never computed in the
  rendering layer.
- Required blocks: reproducibility header from `RunMeta`, equity curve with
  fold boundaries marked, drawdown, rolling Sharpe, return distribution,
  turnover and exposure over time, and a metrics table (Sharpe, Sortino, max
  drawdown, Calmar, hit rate, total costs).
- Risk-engine overlays where available: HMM regime bands beneath the equity
  curve, EVT tail metrics in the table, factor attribution of strategy returns.

Comparing runs is a known extension: the SPA currently loads a single payload,
and a run index is additive to the same shell.

### Calendar policy (inherited)

The Native / Intersection / Unsupported declaration applies unchanged to
backtests. A backtest spanning asset classes with differing calendars declares
its policy in `RunMeta` and surfaces it in the tearsheet header. **Never
forward-fill weekends** — see §7.

### Stop and reconsider if any of these become true

1. A strategy can access data beyond `ctx.t`, by any route.
2. Truncation is applied on observation date without accounting for publication
   lag.
3. `domain/`, `store/`, `ingestion/` or `analytics/` needs to import from
   `backtest/` or `report/`.
4. The rendering layer computes a metric not present in `BacktestResult`.
5. Adding a strategy requires editing engine internals.
6. Backtest integration requires scattered edits across analytics modules.

### Build order

1. `Context`, `Strategy`, `BacktestResult` type definitions **plus their tests**
2. Engine loop, with look-ahead assertions in the test suite from the start
3. Execution layer (costs, slippage, turnover)
4. Two reference strategies: equal-weight baseline, and risk parity once
   Milestone A lands
5. Walk-forward harness
6. Tearsheet sections in the existing export pipeline

---

## 6. Current state

**Keep the markers below current.** A milestone that still asks for something
already built costs the next session either a duplicate implementation or an
hour of reading git log.

### Where the code is now

Ingestion, Parquet store, domain model, performance, risk decomposition,
covariance estimators, factor engine, PCA and hidden concentration, correlation
analytics, MST, stress testing, tail risk (EVT), regime detection. **204 tests
pass**, ruff is green, CI runs both on 3.11 and 3.12.

A static dashboard sits on top: `portfolio.json` declares the universe,
`src/export/` runs the analytics and writes `web/data/analysis.json`, and `web/`
is a dependency-free SPA that renders it. Nothing computes at view time, which
is what keeps hosting free and the universe unbounded.

### Milestone A — Portfolio optimization

**Done:** pluggable covariance estimation (`sample`, `ledoit_wolf`,
`ledoit_wolf_cc`, `ewma`, factor-implied); minimum variance (long-only QP and
closed-form long-short) and the efficient frontier; both surfaced on the
dashboard's portfolio construction page.

**Remaining:** risk parity, HRP, and the shared constraint interface. The
existing optimisers take `long_only` and `max_weight` as ad-hoc keyword
arguments, which is the per-optimizer hardcoding the constraint interface is
meant to remove — introducing it means refactoring `min_variance` and
`efficient_frontier` onto it, not bolting it onto the two new optimisers only.

Requirements: optimisers **consume** the existing risk engine and never
recompute what it already produces; HRP reuses `cluster_correlations`, which
already returns the quasi-diagonal order it needs; constraints are declared, not
hardcoded; covariance estimation stays a pluggable choice. Document the failure
modes in `METHODOLOGY.md` — mean-variance instability under estimation error,
why HRP is more robust, what risk parity assumes.

These optimisers are also the second reference strategy for the backtester, so
they are on the critical path.

### Milestone B — Crypto asset class

**Done and merged.** A CCXT/Binance OHLCV backend behind a new price registry,
and calendars made an explicit attribute of the data.

Delivered: `domain/calendar.py` (`TRADING_DAYS` / `CONTINUOUS`, carrying
`periods_per_year` and `max_normal_gap_days`); `ingestion/prices.py`, the price
registry prices never had, with unregistered tickers falling through to Yahoo so
the universe stays open; `ingestion/crypto.py` for paginated CCXT daily OHLCV;
`analytics/calendar_policy.py` (Native / Intersection / Unsupported, with
`align_factors` raising on a crypto book); calendar-aware data quality; and
annualisation derived from `rs.periods_per_year` wherever an analytic takes a
ReturnSeries.

**Checkpoint result — the abstraction held.** `fit_regimes` ran on BTC with no
edit to the regime module. `RegimeModel.summary()` did annualise with a literal
252 and reported BTC volatility as 26.1%/80.8% instead of 31.4%/97.2% — the
computation was portable, the reporting layer was not. Fixed with a `ppy` field;
no branching anywhere. This result is the evidence behind the single-repo
decision in §1.

**Still open:** `volatility_states` and the regime-conditional helpers take
array inputs with `ppy: int = 252` defaults, so a caller must pass the calendar
explicitly. The dashboard has no crypto-specific page — the factor, stress and
regime sections assume the FF factors apply.

### Milestone C — Backtesting engine

Architecture in §5, build order at the end of it.

**Done:** the three contracts (`Context`, `Strategy`, `BacktestResult`) with
their tests, and both point-in-time producers the contracts needed:

- Every data source declares a publication lag (`prices`, `rates` and a new
  `FACTOR_REGISTRY` in `french.py`, which previously had only a URL map). Before
  this, `Context.publication_lag_days` was validated but structurally always
  zero — a guarantee that looked enforced and was not.
- `RegimeModel` exposes `filtered_probs` / `filtered_states` alongside the
  smoothed ones. The smoothed estimate uses the whole series by construction and
  is correct for description; a strategy must use the filtered one. On the real
  market series the two disagree on 8.6% of days while the aggregate stress
  frequency differs by half a point, so the mistake is invisible in summary
  statistics. See METHODOLOGY §10.1.

**Also done:** the engine loop (bar ordering pinned by a test that a leak
fails by orders of magnitude), the execution layer, walk-forward with blind
per-fold selection, and the tearsheet as a second section producer feeding
`src/export/`.

**Next:** Milestone A's risk parity and HRP, which widen the walk-forward
candidate pool; and a run index so the SPA can compare runs rather than
render one. Note that `min_variance` already accepts a `CovarianceResult`, so
`min_variance(ctx.covariance)` is a reference strategy today — the engine is not
blocked on Milestone A's risk parity.

**Cost to plan around:** `fit_regimes` dominates everything else. Measured on
9,212 observations: 21.4s at `search_reps=20` (34 min for 96 monthly
rebalances), 4.3s at `search_reps=5` (6.8 min), and no further gain below 5 —
the floor is the single EM fit, not the restarts. Covariance re-estimation is
negligible by comparison (0.1s for 96). Refit quarterly rather than monthly.

---

## 7. Calendar handling

Make the calendar an explicit attribute of the data, never an assumption in
code. A hardcoded `252` anywhere is a bug.

- The domain model carries `TRADING_DAYS` (~252/yr) or `CONTINUOUS` (365/yr);
  annualisation factors derive from it.
- **Never forward-fill weekends to align calendars.** Filling equity weekends
  with Friday's price creates artificial zero returns, which deflate volatility
  by roughly 15%, compress correlations, and — worst — cause the HMM to learn a
  spurious low-volatility "weekend regime". `read_returns` intersects instead,
  and that behaviour is load-bearing.
- Each analytic declares its cross-calendar policy explicitly:
  - **Native** — runs on the asset's own frequency. Single-asset-class work:
    tail risk, regime detection, realized vol. Crypto weekends are information,
    not noise.
  - **Intersection** — keep only dates where all assets trade. Anything
    relating different calendars: correlations, MST, factor betas.
  - **Unsupported** — raise an explicit error. The Fama-French factor engine on
    crypto belongs here: a clear error beats a meaningless beta.
- Data quality checks are calendar-aware: a two-day gap is normal under
  `TRADING_DAYS`, an anomaly under `CONTINUOUS`.

### Documented in METHODOLOGY.md

- **Bar close misalignment** — Binance dailies close 00:00 UTC, US equities
  16:00 ET. The offset induces spurious lead-lag correlation. State the
  convention; being honest about it matters more than solving it.
- **Sample period asymmetry** — crypto has short history and roughly one full
  cycle. HMM results on decades of equities and a few years of BTC are not
  comparable.
- **Fat tails** — materially different GPD shape parameters in the EVT model.
  Report them; do not smooth them away.

---

## 8. Working style

- Read existing code before adding to it. Match what is there.
- Small, reviewable changes. Tests alongside, not after.
- Ask when the answer changes **what gets built** — stack, scope, a modelling
  choice with no defensible default. Otherwise pick the obvious option, state
  which one you picked, and keep going. Rounds of clarifying questions before
  any code is its own kind of failure.
- Verify, do not assume. A build that runs is not a build that is correct: read
  the numbers back and check they agree with each other. Several real bugs have
  been caught this way — metrics computed on a different window than the chart
  beside them, stress scenarios silently skipped because the factor history was
  too short, and outlier bands silently falling back to a default because a
  `str`-Enum did not stringify as its value.
- For a backtester specifically: **be suspicious of good results.** A Sharpe
  that jumps when a change lands is a bug hypothesis before it is a finding.
- When a request would violate §2, §3 or §5, say which invariant and why.
