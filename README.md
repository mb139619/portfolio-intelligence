# Portfolio Intelligence Platform

[![CI](https://github.com/mb139619/portfolio-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/mb139619/portfolio-intelligence/actions/workflows/ci.yml)

A near-institutional-grade portfolio analytics and risk platform, built entirely on **free, public data**. The goal is not performance tracking but a deep understanding of where return and risk come from: factor exposures, latent structure, hidden concentration, correlation topology, and behaviour under stress.

Conceptually inspired by systems such as Aladdin, Barra, Axioma and Bloomberg PORT — using only open data sources.

---

## Why this exists

Most retail portfolio tools answer *"how much did I make?"*. This platform answers harder questions:

- Where does the **return** come from?
- Where does the **risk** come from, asset by asset?
- Which **factors** drive the portfolio (market, value, size, quality, ...)?
- How does the portfolio react to **shocks** (2008, COVID, +200bp rates)?
- What **hidden concentrations** exist — is a "diversified" portfolio secretly a single bet?
- How does risk **evolve over time**?

---

## Design principles

1. **Separation of concerns** — data, analytics and visualization are fully decoupled.
2. **Risk engine before dashboards** — the quantitative core comes first.
3. **Parquet-first storage** — data lives in Parquet files; DuckDB is a stateless query engine over them, not a persistent database.
4. **Pure analytics** — every analytic is a pure function (arrays in, results out): no state, no side effects, trivially testable.
5. **Unidirectional dependencies** — `domain ← analytics`, `store ← ingestion`, `viz` depends only on analytics output.
6. **Explainable & decomposable** — every number can be traced to its drivers.

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
| Regime detection| Markov-switching HMM + regime-conditional risk                        |
| Data quality    | Post-ingestion checks: gaps, dated outliers (per asset class), stale  |
| Regime detection| Gaussian HMM + vol-states baseline; regime-conditional beta/correlation |

See [`docs/USAGE.md`](docs/USAGE.md) for a guided tour, [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the models and assumptions, and [`docs/DASHBOARD.md`](docs/DASHBOARD.md) for the web report.

---

## Status

- **Phase 1 (done)** — ingestion, store, domain model, performance & risk decomposition
- **Phase 2 (done)** — factor engine, PCA + hidden concentration, correlation analytics, stress testing, viz module
- **Phase 3 (done)** — regime detection (HMM + vol-states + regime-conditional analytics), network analytics, risk topology map
- **Phase 4 (in progress)** — static dashboard done; portfolio optimization: minimum variance and the efficient frontier done, risk parity and HRP still to come

---

## Disclaimer

This software is for research and educational purposes only. It is **not investment advice**. Free data sources contain errors, gaps and survivorship bias; verify before relying on any output. See `docs/METHODOLOGY.md` for known limitations.
