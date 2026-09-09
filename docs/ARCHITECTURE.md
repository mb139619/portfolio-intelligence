# Architecture

A map of the whole system: what each layer is for, what the load-bearing
decisions are, and every assumption the results rest on.

The other documents go deep on their own subject —
[`METHODOLOGY.md`](METHODOLOGY.md) on the models,
[`BACKTESTING.md`](BACKTESTING.md) on the engine,
[`DASHBOARD.md`](DASHBOARD.md) on the report,
[`USAGE.md`](USAGE.md) on the risk-engine API. This one is the thing they hang
off.

---

## 1. The one claim

Everything here exists to support a single assertion: **a result produced by
this system could have been produced at the time it claims to describe.**

That is a narrower claim than "this strategy makes money" and a much harder one
to fake. Most of the structure below is there to make it checkable rather than
asserted — the truncation in `Context`, the publication lags on every source,
the filtered-not-smoothed regime probabilities, the blind walk-forward
selection, the git hash in `RunMeta`.

The corollary is worth stating plainly: **the risk engine's own numbers are
descriptive, not point-in-time.** The dashboard fits factor betas, PCA, regimes
and covariance on the full sample. That is correct for describing a history and
wrong for trading it, which is exactly why the backtester has its own path
through the same analytics.

---

## 2. The layers

```
                        ┌─────────────┐
                        │   export/   │  report layer: payload + SPA
                        └──────┬──────┘
                               ↓
                        ┌─────────────┐
                        │  backtest/  │  Context · Strategy · BacktestResult
                        └──────┬──────┘
                               ↓
                    ┌──────────────────────┐
                    │ analytics/optimization│  4 allocators, 1 Constraints
                    └──────────┬───────────┘
                               ↓
        ┌──────────────────────────────────────────┐
        │              analytics/                  │  pure functions
        │  performance · risk · factors · pca      │
        │  correlation · regime · stress           │
        └──────────────────┬───────────────────────┘
                           ↓
                    ┌─────────────┐        ┌──────────────┐
                    │   store/    │  ←──   │  ingestion/  │
                    └──────┬──────┘        └──────────────┘
                           ↓
                    ┌─────────────┐
                    │   domain/   │  Calendar · Asset · ReturnSeries
                    └─────────────┘
```

`viz/` sits beside `analytics/` and reads its outputs. `data_quality/` sits
beside `store/`.

**The rule: nothing below imports from above.** Verified, not just intended —
the actual import graph is:

| layer | imports |
|---|---|
| `domain` | *(nothing)* |
| `store` | domain |
| `ingestion` | config, domain, store |
| `analytics` | domain |
| `viz` | analytics |
| `data_quality` | domain, store |
| `backtest` | domain, store, analytics, config |
| `export` | everything — it is the top |

`analytics` importing **only** `domain` is the load-bearing one. It is what
makes every analytic callable on a truncated slice inside a backtest without
adapting anything, and it was tested for real twice: the crypto milestone ran
`fit_regimes` on BTC with no edit to the regime module, and the backtest engine
calls `estimate_covariance` on a rolling window with no point-in-time variant.

---

## 3. What each module is for

### `domain/` — 313 lines
The nouns, with zero external dependencies beyond polars/numpy.

- **`Calendar`** — `TRADING_DAYS` (252/yr) or `CONTINUOUS` (365/yr). Every
  annualisation factor derives from it. A hardcoded 252 anywhere is a bug.
- **`Asset`, `Position`, `Portfolio`** — weights validated to sum to 1.
- **`ReturnSeries`** — the type every analytic consumes. Carries the calendar
  of its *rows*, which is not the calendar of any one asset: a crypto pair read
  alone is continuous, the same pair joined to equities is trading-days,
  because the join keeps only the days both traded.

### `store/` — 342 lines
Parquet is the source of truth; DuckDB is a stateless query engine over it.
There is no database, no schema, no migrations.

`read_returns` is where the most important rule in the system lives: joining
assets with different calendars **intersects** on common dates and never
forward-fills. Filling equity weekends with Friday's close would deflate
volatility ~15%, compress correlations, and teach the HMM a spurious
low-volatility "weekend regime".

### `ingestion/` — 947 lines
Sources behind two registries. `RATE_REGISTRY` maps a logical rate to FRED or
the ECB; `PRICE_REGISTRY` maps a ticker to Yahoo or a CCXT venue, carrying the
trading calendar and the publication lag with it. Anything unregistered falls
through to Yahoo, so the universe stays open.

**Adding a source is a registry entry.** If it ever requires touching
`analytics/`, the abstraction has failed and that is the finding to report.

### `data_quality/` — 430 lines
Post-ingestion checks that detect and report, never auto-correct: gaps,
dated outliers with per-asset-class bands, stale prices, short history.
Calendar-aware — a two-day gap is normal under `TRADING_DAYS` and an anomaly
under `CONTINUOUS`.

### `analytics/` — 3,543 lines, all pure functions
Arrays in, results out. No state, no I/O, no fetching.

| module | answers |
|---|---|
| `performance` | return, volatility, Sharpe/Sortino/Calmar, drawdown, VaR/CVaR |
| `risk/covariance` | Σ: sample, Ledoit-Wolf, LW constant-correlation, EWMA, factor-implied |
| `risk/decomposition` | marginal and percent risk contribution per asset |
| `risk/tail` | Cornish-Fisher VaR, EVT peaks-over-threshold (GPD) |
| `factors` | FF5 betas with HAC t-stats, systematic vs specific, return attribution |
| `pca` | latent factors, eigen-portfolios, **hidden concentration** |
| `correlation` | rolling/EWMA correlation, hierarchical clustering, MST topology |
| `regime` | Markov-switching HMM, regime-conditional beta and correlation |
| `stress` | historical replay through factor betas, parametric shocks |
| `optimization` | minimum variance, efficient frontier, risk parity, HRP |

### `backtest/` — 1,709 lines
Three contracts and one loop. See [`BACKTESTING.md`](BACKTESTING.md).

### `viz/` — 1,018 lines, 22 chart builders
Every function returns a Plotly figure and never calls `.show()`. That is why
the notebook, the dashboard and the tearsheet render identical charts.

### `export/` — 2,002 lines
Two producers feeding one renderer: `build.py` for the portfolio dashboard,
`tearsheet.py` for a backtest run. Both emit the same payload schema; the SPA
in `web/` knows that schema and nothing about finance.

---

## 4. The two pipelines

**Dashboard** — describes a portfolio as it stands.

```
portfolio.json → ingestion → analytics (full sample) → analysis.json → SPA
```

**Backtest** — asks what a rule would have done.

```
portfolio.json → ingestion → engine loop → BacktestResult → tearsheet → same SPA
                                 ↑
                          analytics on truncated slices
```

The same analytics serve both. The difference is entirely in *what window they
see*, which is the point of keeping them pure.

---

## 5. Every assumption, in one place

This is the section to read before trusting a number.

### Data

| assumption | where | consequence if wrong |
|---|---|---|
| Yahoo adjusted closes are correct | `ingestion/yahoo` | survivorship bias, unadjusted splits; `data_quality` flags outliers but cannot fix them |
| `adj_close == close` for crypto | `ingestion/crypto` | none — spot pairs have no dividends or splits; the column exists for schema uniformity |
| BTC-USD is quoted in USDT | `PRICE_REGISTRY` | USDT broke its peg to ~$0.92 in Oct 2018; a USDT pair is not a perfect USD series |
| French factors lag ~45 days | `FRENCH_PUBLICATION_LAG_DAYS` | conservative constant, not measured per-observation; too short would readmit look-ahead |
| daily closes are knowable at the close | `PriceSource` lag = 0 | distinct from execution lag — knowing a close is not being able to trade at it |

### Modelling

| assumption | where | where it stops being valid |
|---|---|---|
| returns are i.i.d. within an estimation window | covariance, PCA | volatility clusters; EWMA and shrinkage mitigate, do not solve |
| Σ is stable over the lookback | all four allocators | breaks in regime transitions, exactly when it matters |
| factor betas are constant | `stress/historical` | betas rise in crises, so replays **understate** losses — the rolling-beta chart shows how much |
| FF5 factors apply to the book | `factors` | meaningless for crypto; `align_factors` raises rather than returning a number |
| 2 regimes are enough | `regime` | a statistical convenience, not ground truth; more states overfit fast on daily data |
| GPD fits the tail beyond the threshold | `risk/tail` | sensitive to the threshold quantile; that is the bias/variance trade |
| equal Sharpe and equal correlation | `risk_parity` | the condition under which ERC is optimal; rarely argued, usually just implied |
| the correlation tree is stable | `hrp` | robust to *inverting* a noisy Σ, not to Σ being wrong |
| historical means forecast returns | `efficient_frontier` only | they do not; μ is shown for construction and labelled as illustrative |

### Backtest

| assumption | where | note |
|---|---|---|
| weight-space accounting | `engine` | no share-level granularity, no odd lots |
| linear transaction costs | `execution` | ignores market impact, which grows super-linearly; honest for liquid ETFs, wrong for size |
| decide at close, earn from tomorrow | `engine` | the bar ordering; a test pins it by orders of magnitude |
| derived quantities refresh on the rebalance schedule | `Context` | stale but causal, recorded in `derived_as_of` |
| filtered regime probabilities as signal | `_fit_regime_at` | smoothed would leak; the two disagree on 8.6% of days |
| no leverage by default | `gross_target` = 1.0 | shorting is supported; a levered book must declare its gross explicitly |
| borrow cost is not charged | `CostModel` | costs are per-trade, not per-day-held; a short-heavy result is optimistic by the financing charge |

### Structural

- **Everything on the dashboard is in-sample.** Full-sample fits describing a
  history.
- **Free data is patchy.** FRED times out on some networks, Yahoo rate-limits,
  the French library publishes late. Ingestion is incremental and degrades
  rather than failing.
- **No survivorship-bias correction anywhere.** The universe is whatever is in
  `portfolio.json` today, which is a universe selected with hindsight.

---

## 6. Where to look

| question | file |
|---|---|
| what does this number mean | `docs/METHODOLOGY.md` |
| how do I write a strategy | `docs/BACKTESTING.md` |
| how do I add a report page | `docs/DASHBOARD.md` |
| how do I call the risk engine | `docs/USAGE.md` |
| what are the rules for changing things | `CLAUDE.md` |
| how do the pieces connect | this file |

### Commands

```bash
python -m src.export --serve                 # build and open the dashboard
python -m src.backtest                       # compare strategies
python -m src.export.tearsheet runs/x --serve # render a saved run
python examples/custom_strategies.py         # four worked strategies
pytest -q                                    # 353 tests
ruff check src/ tests/ examples/
```

---

## 7. What is deliberately absent

- **Order management and execution simulation.** Weights and fills, not orders.
- **Intraday anything.** Daily bars throughout.
- **Leverage.** `Constraints.budget` is 1.0 and nothing borrows.
- **A database.** Parquet files and a stateless query engine.
- **A build step in the frontend.** No Node, no bundler; the SPA is three files
  and a vendored Plotly.
- **Point-in-time estimation on the dashboard.** By design: it describes.
