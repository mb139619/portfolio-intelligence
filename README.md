# Portfolio Intelligence Platform

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
│   ├── store/parquet_store.py  # Parquet I/O + DuckDB query engine
│   ├── ingestion/              # data source connectors behind one interface
│   │   ├── base.py             #   abstract ingester
│   │   ├── yahoo.py            #   prices / OHLCV
│   │   ├── rates.py            #   rates: USD→FRED, EUR→ECB (unified interface)
│   │   ├── french.py           #   Fama-French factors
│   │   ├── http.py             #   resilient fetch (retry + backoff)
│   │   └── pipeline.py         #   orchestrator (incremental updates)
│   ├── analytics/
│   │   ├── performance.py      # Sharpe, Sortino, Calmar, drawdown, VaR, CVaR
│   │   ├── risk/decomposition.py        # MCR, risk contribution, %RC
│   │   ├── factors/            # factor engine, exposures, PCA-free attribution
│   │   ├── pca/                # PCA risk model + Hidden Concentration Detector
│   │   ├── correlation/        # rolling/EWMA corr, clustering, MST topology
│   │   └── stress/             # historical replay + parametric shocks
│   └── viz/plots.py            # reusable Plotly figures
├── notebooks/01_end_to_end.py  # full walkthrough (jupytext py:percent)
├── tests/                      # 87 unit tests
├── docs/                       # USAGE.md, METHODOLOGY.md
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

# Open the notebook (VS Code / JupyterLab opens the .py as a notebook)
jupyter lab notebooks/01_end_to_end.py
```

The first notebook run downloads data from Yahoo / FRED / ECB / French and stores it as Parquet. Subsequent runs reuse the local files (incremental updates).

> **`uv` users:** replace the install steps with `uv venv && uv pip install -e ".[dev]"`.

---

## Data sources

| Logical data    | Source                | Notes                                   |
|-----------------|-----------------------|-----------------------------------------|
| Prices / ETFs   | Yahoo Finance (`yfinance`) | OHLCV + adjusted close              |
| USD rates       | FRED (CSV endpoint)   | No API key required                     |
| EUR rates       | ECB Data Portal (SDMX)| No API key required                     |
| Factors         | Kenneth French Library| FF5 + Momentum, daily                   |

Rates are exposed through a **single interface**: you request a logical series (`USD_FEDFUNDS`, `EUR_DFR`) and the registry routes it to the right backend. Adding a new currency or source is a registry entry, not new plumbing.

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
| Regime detection| Gaussian HMM + vol-states baseline; regime-conditional beta/correlation |

See [`docs/USAGE.md`](docs/USAGE.md) for a guided tour and [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the models and assumptions.

---

## Status

- **Phase 1 (done)** — ingestion, store, domain model, performance & risk decomposition
- **Phase 2 (done)** — factor engine, PCA + hidden concentration, correlation analytics, stress testing, viz module
- **Phase 3 (in progress)** — regime detection (HMM + vol-states + regime-conditional analytics) done; network analytics, risk topology map
- **Phase 4 (planned)** — portfolio optimization (mean-variance, risk parity, HRP), API & dashboard

---

## Disclaimer

This software is for research and educational purposes only. It is **not investment advice**. Free data sources contain errors, gaps and survivorship bias; verify before relying on any output. See `docs/METHODOLOGY.md` for known limitations.
