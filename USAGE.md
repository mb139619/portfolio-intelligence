# Usage Guide

A practical, task-oriented tour of the platform. Every example assumes you have
installed the project (`pip install -e ".[dev]"`) and are running from the
project root or inside the notebook.

---

## 1. Setup

```python
from src.config import settings
from src.store.parquet_store import ParquetStore
from src.ingestion.pipeline import IngestionPipeline

settings.ensure_dirs()
store = ParquetStore(settings.data_dir)
pipeline = IngestionPipeline(store)
```

`settings` is the single configuration object. Override any value with a `.env`
file (prefix `PI_`), e.g. `PI_RISK_FREE_RATE=0.045`.

---

## 2. Ingesting data

Ingestion is **incremental**: re-running only fetches new observations.

```python
pipeline.update_prices(["SPY", "TLT", "GLD", "QQQ", "EFA"])
pipeline.update_rates(["USD_FEDFUNDS", "USD_10Y", "EUR_DFR"])
pipeline.update_factors(["FF5", "MOM"])
```

Available rate series live in `src.ingestion.rates.RATE_REGISTRY`; list them with:

```python
from src.ingestion.rates import available_rates
available_rates(currency="EUR")
```

If a source is slow or blocked (corporate networks sometimes throttle FRED),
the fetch retries with exponential backoff and then fails gracefully without
breaking the rest of the run.

---

## 3. Querying the store

Two access patterns.

**Typed reads** (used by analytics):

```python
prices  = store.read_prices(["SPY", "TLT"], start="2018-01-01")   # wide DataFrame
returns = store.read_returns(["SPY", "TLT"], start="2018-01-01")  # ReturnSeries
factors = store.read_factors(["Mkt-RF", "SMB", "HML", "RF"])
```

**Ad-hoc SQL** (DuckDB over the Parquet files — great in notebooks):

```python
store.sql("SELECT ticker, count(*) FROM prices() GROUP BY ticker")
```

The `prices()`, `rates()` and `factors()` table macros are registered
automatically per query. There is no persistent database to manage.

---

## 4. Building a portfolio

```python
from src.domain.portfolio import Portfolio, Position, Asset, AssetClass

assets = [
    Asset("SPY", "S&P 500 ETF", AssetClass.EQUITY),
    Asset("TLT", "20Y Treasury ETF", AssetClass.FIXED_INCOME),
    Asset("GLD", "Gold ETF", AssetClass.COMMODITY),
]
portfolio = Portfolio("Demo", [
    Position(assets[0], 0.6),
    Position(assets[1], 0.3),
    Position(assets[2], 0.1),
])
# weights must sum to 1 (validated); use Portfolio.equal_weight(name, assets) for 1/N
```

---

## 5. Performance & risk

```python
from src.analytics.performance import compute_metrics
from src.analytics.risk.decomposition import decompose_risk

rs = store.read_returns(portfolio.tickers, start="2018-01-01")
port_returns = rs.portfolio_returns(portfolio.weights).to_numpy()

print(compute_metrics(port_returns, rfr=settings.risk_free_rate).summary())
print(decompose_risk(portfolio.weights, rs, cov_method="ledoit_wolf").summary())
```

Covariance estimators: `"sample"`, `"ledoit_wolf"` (shrinkage, robust),
`"ewma"` (RiskMetrics).

---

## 6. Factor engine

```python
from src.analytics.factors.prepare import align_factors, FF5_FACTORS
from src.analytics.factors.engine import estimate_factor_model
from src.analytics.factors.attribution import decompose_factor_risk, attribute_returns

factor_wide = store.read_factors(FF5_FACTORS + ["RF"])
aligned = align_factors(rs.portfolio_returns(portfolio.weights), rs.dates,
                        factor_wide, FF5_FACTORS)

model = estimate_factor_model(aligned, hac_lags=5)
print(model.summary())                          # betas + HAC t-stats, R², alpha
print(decompose_factor_risk(model, aligned).summary())  # systematic vs specific
print(attribute_returns(model, aligned).summary())      # return by factor
```

`aligned.coverage` reports the exact window the model is fit on. If factor data
covers fewer dates than your prices (the French library publishes with a lag),
a warning is logged automatically.

---

## 7. PCA & hidden concentration

```python
from src.analytics.pca.model import fit_pca
from src.analytics.pca.concentration import detect_hidden_concentration

pca = fit_pca(rs, method="covariance")
print(pca.summary())                                    # explained variance per PC
print(detect_hidden_concentration(portfolio.weights, rs, model=pca).summary())
```

The detector compares the **effective number of bets** implied by risk to the
naive number implied by weights. A large gap = hidden concentration.

---

## 8. Correlation & topology

```python
from src.analytics.correlation.clustering import cluster_correlations
from src.analytics.correlation.network import build_mst
from src.analytics.correlation.rolling import average_correlation

print(cluster_correlations(rs, n_clusters=3).summary())  # correlation clusters
print(build_mst(rs).summary())                           # hubs & diversifiers
avg = average_correlation(rs, window=63)                 # systemic correlation series
```

---

## 9. Stress testing

```python
from src.analytics.stress.scenarios import HISTORICAL_SCENARIOS, FACTOR_SHOCKS
from src.analytics.stress.historical import run_historical_factor
from src.analytics.stress.parametric import run_factor_shock

factors_all = store.read_factors(FF5_FACTORS)

# Factor-based historical replay (works even for pre-ETF crises like 2008)
print(run_historical_factor(model, factors_all, HISTORICAL_SCENARIOS["GFC_2008"]).summary())

# Parametric shock
print(run_factor_shock(model, {"Mkt-RF": -0.20}, name="EQUITY_CRASH").summary())
```

For macro shocks (rates / oil / USD / VIX), estimate sensitivities first with
`estimate_macro_sensitivities(...)` then apply `run_macro_shock(...)`.

---

## 10. Visualization

All plotting functions return Plotly figures (they never call `.show()`):

```python
from src.viz.plots import plot_performance, plot_factor_exposures, plot_mst_network

plot_performance(port_returns, rs.dates.to_list()).show()
plot_factor_exposures(model).show()
plot_mst_network(build_mst(rs)).show()
```

Because they return figures, the exact same code works in a notebook, a script,
or a future web dashboard.

---

## 11. Running the full walkthrough

`notebooks/01_end_to_end.py` ties everything together. Open it in VS Code or
JupyterLab (it is a jupytext `py:percent` file, so it renders as a notebook)
and run the cells top to bottom.
