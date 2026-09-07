# CLAUDE.md

Working brief for Claude Code on `portfolio-intelligence`.

Read this before writing code. When a request conflicts with the invariants
below, say so instead of silently working around them.

---

## 1. What this project is

A portfolio analytics and risk platform built on free, public data. It answers
*where risk and return come from* — factor exposures, latent structure, hidden
concentration, correlation topology, behaviour under stress.

It is **not** a performance tracker, **not** a signal generator, and **not** a
backtesting engine. Strategy simulation is deliberately out of scope and will
live in a separate repository that imports this one as a library.

If a request would introduce order management, execution simulation, or
stateful strategy loops, flag it as out of scope before implementing.

---

## 2. Architecture invariants

These are load-bearing. Do not break them to make a feature fit.

1. **Separation of concerns** — data, analytics and visualization are fully
   decoupled.
2. **Risk engine before dashboards** — the quantitative core comes first. This
   precondition is now met, and the dashboard was built on top of it. It still
   binds going forward: a new analytic ships with its model, its tests and its
   methodology note *before* it gets a page.
3. **Parquet-first storage** — data lives in Parquet; DuckDB is a stateless
   query engine over those files, never a persistent database.
4. **Pure analytics** — every analytic is a pure function: arrays in, results
   out. No hidden state, no side effects, no I/O inside analytics.
5. **Unidirectional dependencies** — `domain ← analytics`, `store ← ingestion`,
   `viz` depends only on analytics output, `export` depends on both and is
   depended on by neither. Never the reverse.
6. **Explainable & decomposable** — every number traces back to its drivers.

### Practical consequences

- Analytics never fetch data. They receive it.
- New data sources are **registry entries**, not new plumbing. If adding a
  source requires touching analytics code, the abstraction has failed — stop
  and report it rather than patching around it.
- No `if asset_class == "crypto"` branches inside analytics. Behaviour that
  varies by asset class belongs to metadata on the domain object.
- A new chart belongs in `viz/plots.py`, where it is unit-tested and the
  notebook can reuse it — never inline in the export layer.
- The dashboard renders; it does not compute. If a page needs a number that
  does not exist yet, the number is a new analytic, not a line in `build.py`.

---

## 3. Conventions

- Python, `src/` layout, editable install via `pyproject.toml`.
- Tests with `pytest`, mirroring the `src/` structure.
- Type hints on all public functions.
- Methodology and model assumptions go in `docs/METHODOLOGY.md`; usage examples
  in `docs/USAGE.md`. Keep both current — a feature is not done until its
  assumptions and limitations are written down.
- Every model gets a short note on **where it stops being valid**. This matters
  as much as the implementation.

### Commands

The virtualenv is `.venv` and is not always on PATH; call it explicitly.

```bash
.venv/Scripts/python.exe -m pytest -q                  # full suite (~20s)
.venv/Scripts/python.exe -m ruff check src/            # line-length 88, E/F/I/UP
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
  price data. Anything factor-based is fitted on a shorter window than the
  prices; say which window rather than hiding the gap.
- Console output is cp1252 on Windows; set `PYTHONIOENCODING=utf-8` before
  printing anything with arrows or Greek letters.

---

## 4. Current milestones

This section is a launchpad, not an inventory — but a launchpad has to know
where the ground is. **Keep the state markers below current.** A milestone that
still asks for something already built will cost the next session either a
duplicate implementation or an hour of reading git log to find out.

### Where the code is now

Phases 1-3 are done: ingestion, Parquet store, domain model, performance, risk
decomposition, covariance estimators, factor engine, PCA and hidden
concentration, correlation analytics, MST, stress testing, tail risk (EVT),
regime detection. 177 tests pass.

A static dashboard exists on top of all of it: `portfolio.json` declares the
universe, `src/export/` runs the analytics and writes `web/data/analysis.json`,
and `web/` is a dependency-free SPA that renders it. Nothing computes at view
time, which is what keeps hosting free and the universe unbounded. Adding a
page means adding a section in `src/export/build.py` and registering it in
`SECTIONS` — no JavaScript.

### Milestone A — Portfolio optimization (Phase 4)

Add mean-variance, risk parity, and Hierarchical Risk Parity.

**Done:** pluggable covariance estimation (`sample`, `ledoit_wolf`,
`ledoit_wolf_cc`, `ewma`, factor-implied) in `analytics/risk/covariance.py`;
minimum variance (long-only QP and closed-form long-short) and the efficient
frontier in `analytics/optimization/`; both surfaced on the dashboard's
portfolio construction page.

**Remaining:** risk parity, HRP, and the shared constraint interface — the
existing optimisers currently take `long_only` and `max_weight` as ad-hoc
keyword arguments, which is exactly the per-optimizer hardcoding the
requirement below forbids. Introducing the constraint interface means
refactoring `min_variance` and `efficient_frontier` onto it, not bolting it
onto the two new optimisers only.

Requirements:

- Optimizers **consume** the existing risk engine — covariance estimates, risk
  contributions, correlation clustering. Do not reimplement or recompute
  anything the risk engine already produces.
- HRP must reuse the existing correlation clustering rather than building its
  own linkage from scratch. `cluster_correlations` already returns the
  quasi-diagonal order HRP needs; use it.
- A shared constraint interface across all three optimizers (long-only, weight
  bounds, budget). Constraints are declared, not hardcoded per optimizer.
- Covariance estimation is a pluggable choice (sample, Ledoit-Wolf shrinkage,
  EWMA), not an assumption baked into the optimizer.
- Document the failure modes in `METHODOLOGY.md`: mean-variance instability
  under estimation error, why HRP is more robust, what risk parity assumes.
- Each new optimizer gets a strategy column on the dashboard's weights
  comparison table; that page is built to take them without restructuring.

### Milestone B — Crypto asset class

**Done, on branch `milestone-b-crypto`.** Add a CCXT/Binance OHLCV backend and
make the platform calendar-aware.

Delivered:

- `domain/calendar.py` — `Calendar.TRADING_DAYS` / `CONTINUOUS`, carrying
  `periods_per_year` and `max_normal_gap_days`. `Asset` derives its calendar
  from its asset class; `ReturnSeries` carries the calendar of the rows and
  propagates it through `select`/`trim`.
- `ingestion/prices.py` — the price registry that `rates.py` always had and
  prices never did. Routes a logical ticker to a backend; unregistered tickers
  fall through to Yahoo, which keeps the universe open.
- `ingestion/crypto.py` — paginated CCXT daily OHLCV.
- `analytics/calendar_policy.py` — Native / Intersection / Unsupported, with
  `align_factors` now raising `UnsupportedCalendarError` on a crypto book.
- Calendar-aware data quality: one panel per calendar (a shared panel would
  flag every equity as missing ~104 dates a year), plus `check_observation_gaps`.
- Annualisation derives from `rs.periods_per_year` wherever an analytic takes a
  ReturnSeries. Equity results are byte-identical; BTC annualises at 365.

**Checkpoint result — the abstraction held, with one caveat worth keeping.**
`fit_regimes` ran on BTC with no edit to the regime module: the estimation is
calendar-agnostic by construction. But `RegimeModel.summary()` annualised with a
literal `252`, so it *reported* BTC volatility as 26.1%/80.8% instead of
31.4%/97.2%. The computation was portable; the reporting layer was not. That is
a calendar-generalisation gap, not crypto special-casing, and the fix was to add
a `ppy` field — no branching anywhere.

**Still open:** `volatility_states` and the regime-conditional helpers take
array inputs with `ppy: int = 252` defaults, so a caller must pass the calendar
explicitly. The dashboard has no crypto page — `portfolio.json` accepts crypto
tickers, but the factor, stress and regime sections assume the FF factors apply.

Scope discipline — this must be **purely additive**:

- Work on a branch. Run the full test suite on the existing equity portfolio
  before and after; results must be identical.
- The change should consist of (a) a new registry backend and (b) a `calendar`
  attribute on the domain model. Nothing else.
- **Checkpoint:** if regime detection runs on BTC without editing the regime
  detection module, the abstraction holds. If it requires edits scattered
  through analytics, stop and report — that is a real finding, not a failure to
  push through.

#### Calendar handling

Make the calendar an explicit attribute of the data, never an assumption in
code. Hardcoded `252` anywhere is a bug.

- Domain model carries `TRADING_DAYS` (~252/yr) or `CONTINUOUS` (365/yr).
  Annualization factors derive from it.
- **Never forward-fill weekends to align calendars.** Filling equity weekends
  with Friday's price creates artificial zero returns, which deflate volatility
  by roughly 15%, compress correlations, and — worst — cause the HMM to learn a
  spurious low-volatility "weekend regime". This is the single most important
  rule in this milestone.
- Each analytic declares its cross-calendar policy explicitly:
  - **Native** — runs on the asset's own frequency. Use for single-asset-class
    work: tail risk, regime detection, realized vol. Crypto weekends are
    information, not noise.
  - **Intersection** — keep only dates where all assets trade. Use for anything
    relating different calendars: correlations, MST, factor betas.
  - **Unsupported** — raise an explicit error. The Fama-French factor engine on
    crypto belongs here: the factors have no meaning for that asset class. A
    clear error beats a meaningless beta.
- Data quality checks become calendar-aware: a two-day gap is normal under
  `TRADING_DAYS`, an anomaly under `CONTINUOUS` (exchange outage, delisted pair).

#### Document in METHODOLOGY.md

- **Bar close misalignment** — Binance dailies close 00:00 UTC, US equities
  16:00 ET. The ~5–8h offset induces spurious lead-lag correlation. State the
  chosen convention; being honest about it matters more than solving it.
- **Sample period asymmetry** — crypto has short history and roughly one full
  cycle. HMM results on 20 years of equities and 5 years of BTC are not
  comparable. Say so.
- **Fat tails** — expect materially different GPD shape parameters in the EVT
  tail model. Report them; do not smooth them away.

---

## 5. Working style

- Read existing code before adding to it. Match what is there.
- Small, reviewable changes. Tests alongside, not after.
- Ask when the answer changes **what gets built** — stack, scope, a modelling
  choice with no defensible default. Otherwise pick the obvious option, state
  which one you picked, and keep going. Rounds of clarifying questions before
  any code is its own kind of failure.
- Verify, do not assume. A build that runs is not a build that is correct: read
  the numbers back and check they agree with each other. Two real bugs in the
  dashboard work were found this way — metrics computed on a different window
  than the chart beside them, and stress scenarios silently skipped because the
  factor history was too short.
- When a request would violate section 2, say which invariant and why.
