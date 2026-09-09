# Portfolio Backtesting & Risk Platform

[![CI](https://github.com/mb139619/portfolio-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/mb139619/portfolio-intelligence/actions/workflows/ci.yml)

A strategy backtesting and research platform built entirely on **free, public data**, whose distinguishing claim is *rigour rather than returns*. It answers whether a strategy's result is real: was the data available at the time, does the edge survive costs, does it hold out of sample, and where does its risk actually come from.

**Backtest rigour is the product; the strategies are the demonstration vehicle.** A well-instrumented equal-weight baseline is worth more here than an elaborate signal with a sloppy harness.

Underneath sits a full risk engine — factor exposures, latent structure, hidden concentration, correlation topology, tail behaviour, regimes. That is what makes a result here worth more than a for-loop over prices: every strategy arrives with its factor attribution, its regime-conditional behaviour and its tail risk attached.

Conceptually inspired by systems such as Aladdin, Barra, Axioma and Bloomberg PORT — using only open data sources.

---

## Why this exists

Most retail backtesters answer *"what would this have returned?"* — and most of those answers are wrong, because the harness quietly saw the future. This platform is built around the questions that decide whether a result means anything:

- Was the data **actually available at the time**, including publication lag?
- Does the edge survive **transaction costs, slippage and turnover**?
- Does it hold **out of sample**, or only on the window it was tuned on?
- Where does the **risk** come from, asset by asset and factor by factor?
- How does it behave in a **stress regime** rather than on average?
- What **hidden concentration** is it carrying — is a "diversified" book secretly a single bet?
- How badly does it lose in the **tail** the normal distribution does not model?

---

## Design principles

1. **Point-in-time by construction** — look-ahead is prevented structurally, not by convention. The engine hands a strategy a context whose data is already truncated at `t`, with publication lag applied; a strategy never holds a reference to the full dataset, so there is no discipline to remember.
2. **Separation of concerns** — data, analytics, backtesting and presentation are fully decoupled.
3. **Parquet-first storage** — data lives in Parquet files; DuckDB is a stateless query engine over them, not a persistent database.
4. **Pure analytics** — every analytic is a pure function (arrays in, results out): no state, no side effects, trivially testable.
5. **Unidirectional dependencies** — `report → backtest → optimization → analytics → store ← ingestion`, all resting on `domain`. Nothing below ever imports from above.
6. **Presentation renders, it does not compute** — a missing metric is added to the result object, never calculated in a template.
7. **Explainable & decomposable** — every number can be traced to its drivers.

---

## Architecture

```
portfolio-intelligence/
├── data/                       # local data lake (gitignored)
│   ├── raw/{prices,macro,factors}
│   ├── processed/
│   └── features/
├── src/
│   ├── config.py               # settings (single source of truth)
│   ├── domain/                 # Portfolio, Position, Asset, ReturnSeries — pure
│   │   └── calendar.py         #   TRADING_DAYS vs CONTINUOUS; annualisation
│   ├── store/parquet_store.py  # Parquet I/O + DuckDB query engine
│   ├── data_quality/          # post-ingestion checks (gaps, outliers, stale, gms)
│   ├── ingestion/              # data source connectors behind one interface
│   │   ├── base.py             #   abstract ingester
│   │   ├── prices.py           #   price registry: ticker -> backend + calendar
│   │   ├── yahoo.py            #   equities / ETFs OHLCV
│   │   ├── crypto.py           #   crypto OHLCV via CCXT (Binance spot)
│   │   ├── rates.py            #   rates: USD→FRED, EUR→ECB (unified interface)
│   │   ├── french.py           #   Fama-French factors
│   │   ├── http.py             #   resilient fetch (retry + backoff)
│   │   └── pipeline.py         #   orchestrator (incremental updates)
│   ├── analytics/
│   │   ├── performance.py      # Sharpe, Sortino, Calmar, drawdown, VaR, CVaR
│   │   ├── calendar_policy.py  # native / intersection / unsupported per analytic
│   │   ├── risk/decomposition.py        # MCR, risk contribution, %RC
│   │   ├── factors/            # factor engine, exposures, PCA-free attribution
│   │   ├── pca/                # PCA risk model + Hidden Concentration Detector
│   │   ├── correlation/        # rolling/EWMA corr, clustering, MST topology
│   │   └── stress/             # historical replay + parametric shocks
│   ├── backtest/               # (next) engine, strategies, execution, walk-forward
│   ├── viz/plots.py            # reusable Plotly figures
│   └── export/                 # portfolio.json -> analytics -> analysis.json
│       ├── spec.py             #   input contract + validation
│       ├── encode.py           #   JSON-safe encoding + payload schema
│       ├── build.py            #   the eight dashboard sections
│       └── __main__.py         #   CLI
├── portfolio.json              # the portfolio: universe, weights, parameters
├── web/                        # static dashboard (no build step, no backend)
│   ├── index.html, app.js, styles.css
│   ├── vendor/plotly.min.js
│   └── data/analysis.json      # the built payload - what a static host serves
├── notebooks/01_end_to_end.py  # full walkthrough (jupytext py:percent)
├── tests/                      # 204 unit tests
├── docs/                       # USAGE.md, METHODOLOGY.md, DASHBOARD.md
├── pyproject.toml
└── requirements.lock           # pinned, reproducible environment
```

---

## Quick start

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# Install the project (editable) + dev tools
pip install -e ".[dev]"

# Run the test suite
pytest -q

# Build the dashboard and open it in a browser
python -m src.export --serve

# Or open the notebook (VS Code / JupyterLab opens the .py as a notebook)
jupyter lab notebooks/01_end_to_end.py
```

The first run downloads data from Yahoo / FRED / ECB / French and stores it as Parquet. Subsequent runs reuse the local files (incremental updates).

---

## Dashboard

Edit `portfolio.json`, run `python -m src.export --serve`, and the whole analysis
is rendered as a navigable static report. The eight pages read as one argument:
what the book is (overview), what environment it lives in (factor model, market
regimes), what its risk properties are (risk decomposition, tail risk, latent
structure), what breaks it (stress testing), and only then what to do about it
(portfolio construction).

The pipeline is deliberately offline — `portfolio.json → analytics → analysis.json → static SPA`.
Nothing computes at view time, so the frontend is a plain static directory that
any host serves for free, and the universe stays unbounded: ingestion runs
locally, where Yahoo is reachable and memory is not rationed.

`--standalone` bundles everything into a single self-contained HTML file that
opens with a double click and needs no server at all.

See [`docs/DASHBOARD.md`](docs/DASHBOARD.md) for the payload schema, the full
option list and how to add a section.

> **`uv` users:** replace the install steps with `uv venv && uv pip install -e ".[dev]"`.

---

## Data sources

| Logical data    | Source                | Notes                                   |
|-----------------|-----------------------|-----------------------------------------|
| Prices / ETFs   | Yahoo Finance (`yfinance`) | OHLCV + adjusted close              |
| Crypto          | Binance spot via `ccxt` | Daily OHLCV, 24/7 calendar             |
| USD rates       | FRED (CSV endpoint)   | No API key required                     |
| EUR rates       | ECB Data Portal (SDMX)| No API key required                     |
| Factors         | Kenneth French Library| FF5 + Momentum, daily                   |

Both rates and prices are exposed through a **single interface**: you request a logical series (`USD_FEDFUNDS`, `EUR_DFR`) or a logical ticker (`SPY`, `BTC-USD`) and the registry routes it to the right backend, carrying the trading calendar with it. Anything unregistered falls through to Yahoo, so the universe stays open. Adding a currency, a venue or an asset class is a registry entry, not new plumbing.

---

## Feature overview

| Module          | What it answers                                                        |
|-----------------|------------------------------------------------------------------------|
| Performance     | Return, volatility, risk-adjusted ratios, drawdowns, VaR/CVaR          |
| Risk decomposition | Marginal & percent risk contribution per asset                      |
| Factor engine   | FF5 betas (HAC t-stats), systematic vs idiosyncratic risk, attribution |
| PCA risk model  | Latent factors, explained variance, eigen-portfolios                   |
| Hidden Concentration Detector | Effective number of bets vs naive diversification        |
| Correlation     | Rolling/EWMA correlation, clustering, Minimum Spanning Tree            |
| Stress testing  | Historical replay (any era) + parametric factor/macro shocks          |
| Tail risk       | Cornish-Fisher (modified) VaR + EVT peaks-over-threshold (GPD)         |
| Regime detection| Gaussian HMM + vol-states baseline; regime-conditional beta/correlation |
| Data quality    | Post-ingestion checks: gaps, dated outliers (per asset class), stale  |
| Calendars       | Trading-day vs continuous; intersection, never forward-fill           |
| Backtesting     | *(next)* point-in-time engine, costs & slippage, walk-forward folds   |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the whole system fits together and every assumption it rests on, [`docs/BACKTESTING.md`](docs/BACKTESTING.md) for the engine and how to write a strategy, [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the models and assumptions, [`docs/USAGE.md`](docs/USAGE.md) for a guided tour of the risk engine, and [`docs/DASHBOARD.md`](docs/DASHBOARD.md) for the web report.

---

## Status

- **Phase 1 (done)** — ingestion, store, domain model, performance & risk decomposition
- **Phase 2 (done)** — factor engine, PCA + hidden concentration, correlation analytics, stress testing, viz module
- **Phase 3 (done)** — regime detection (HMM + vol-states + regime-conditional analytics), network analytics, risk topology map
- **Phase 4 (in progress)** — static dashboard done; crypto asset class and calendar-aware analytics done; portfolio optimization: minimum variance and the efficient frontier done, risk parity and HRP still to come
- **Phase 5 (next)** — backtesting engine: point-in-time `Context`, execution modelling with explicit costs, walk-forward harness, tearsheet rendered through the existing export pipeline

---

## Disclaimer

This software is for research and educational purposes only. It is **not investment advice**. Free data sources contain errors, gaps and survivorship bias; verify before relying on any output. See `docs/METHODOLOGY.md` for known limitations.
