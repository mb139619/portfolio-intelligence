# %% [markdown]
# # 01 — End-to-end analysis
#
# This notebook walks through the full pipeline:
# **ingestion → store (Parquet) → ReturnSeries → performance, risk, factors, PCA, correlation, stress.**
#
# Written in `jupytext` `py:percent` format: it opens as a notebook in
# JupyterLab / VS Code but is version-controlled as a clean `.py` file.
#
# Run the cells in order. The first run downloads data; later runs
# reuse the local Parquet files (incremental updates).

# %%
from src.config import settings
from src.store.parquet_store import ParquetStore
from src.ingestion.pipeline import IngestionPipeline
from src.domain.portfolio import Portfolio, Position, Asset, AssetClass
from src.analytics.performance import compute_metrics
from src.analytics.risk.decomposition import decompose_risk, rolling_risk_contribution

settings.ensure_dirs()
store = ParquetStore(settings.data_dir)
pipeline = IngestionPipeline(store)

# %% [markdown]
# ## 1. Ingestion
# Minimal universe: equity, bonds, gold. Incremental, so re-running is cheap.

# %%
UNIVERSE = ["SPY", "TLT", "GLD", "QQQ", "EFA"]

results = pipeline.update_prices(UNIVERSE)
for r in results:
    print(r)

# Risk-free + a couple of rates for stress / context (USD and EUR via the same interface)
pipeline.update_rates(["USD_FEDFUNDS", "USD_10Y", "EUR_DFR"])

# Fama-French 5 + momentum (for the factor engine)
pipeline.update_factors(["FF5", "MOM"])

# %% [markdown]
# ## 2. Ad-hoc SQL over the Parquet files (DuckDB as a query engine)
# No persistent database: DuckDB reads the files on demand.

# %%
print(store.sql("""
    SELECT ticker,
           count(*)        AS days,
           min(date)       AS first,
           max(date)       AS last,
           round(avg(volume)) AS avg_volume
    FROM prices()
    GROUP BY ticker
    ORDER BY ticker
"""))

# %% [markdown]
# ## 3. Build the portfolio + ReturnSeries

# %%
assets = [
    Asset("SPY", "S&P 500 ETF", AssetClass.EQUITY),
    Asset("TLT", "20Y Treasury ETF", AssetClass.FIXED_INCOME),
    Asset("GLD", "Gold ETF", AssetClass.COMMODITY),
    Asset("QQQ", "Nasdaq 100 ETF", AssetClass.EQUITY),
    Asset("EFA", "MSCI EAFE ETF", AssetClass.EQUITY),
]
portfolio = Portfolio("Demo 60/40-ish", [
    Position(assets[0], 0.35),
    Position(assets[1], 0.30),
    Position(assets[2], 0.10),
    Position(assets[3], 0.15),
    Position(assets[4], 0.10),
])
print(portfolio)

rs = store.read_returns(portfolio.tickers, start="2018-01-01")
print(rs)

# %% [markdown]
# ## 4. Portfolio performance

# %%
import numpy as np
port_returns = rs.portfolio_returns(portfolio.weights).to_numpy()

# Use the REAL daily risk-free series (French 'RF'), not a constant.
# We align it to the portfolio returns by date; Sharpe/Sortino are then computed
# on genuine excess returns — important when rates move across the sample.
from src.analytics.performance import align_risk_free, compute_metrics
rf_wide = store.read_factors(["RF"])
r_aligned, rf_daily, _ = align_risk_free(
    rs.portfolio_returns(portfolio.weights), rs.dates, rf_wide
)
metrics = compute_metrics(r_aligned, rf=rf_daily)
print(metrics.summary())

# Equity curve + drawdown (from the reusable viz module)
from src.viz.plots import plot_performance
plot_performance(port_returns, rs.dates.to_list(), title="Demo 60/40-ish — performance").show()

# %% [markdown]
# ## 5. Risk decomposition — who actually generates the risk?
# The `<--` flag marks assets whose risk contribution far exceeds their weight:
# this is the seed of the **Hidden Concentration Detector**.

# %%
decomp = decompose_risk(portfolio.weights, rs, cov_method="ledoit_wolf")
print(decomp.summary())
print()
print(decomp.to_dataframe())

# %% [markdown]
# ## 6. Risk evolution over time (rolling %RC)

# %%
rolling = rolling_risk_contribution(portfolio.weights, rs, window=63, cov_method="ewma")
print(rolling.tail(10))

# Quick plot with Plotly
import plotly.express as px
fig = px.area(
    rolling.to_pandas(), x="date", y="pct_rc", color="ticker",
    title="Rolling % Risk Contribution (63d, EWMA)",
)
fig.show()

# %% [markdown]
# ## 7. Factor Engine — exposures, systematic risk, attribution
#
# We regress the portfolio's excess returns on the Fama-French 5 factors.
# Three key outputs:
# - **exposures** (betas) with HAC t-stats
# - **risk decomposition**: how much is market / value / ... / idiosyncratic
# - **return attribution**: where the return comes from

# %%
from src.analytics.factors.prepare import align_factors, FF5_FACTORS
from src.analytics.factors.engine import estimate_factor_model, rolling_betas
from src.analytics.factors.attribution import decompose_factor_risk, attribute_returns

# FF5 factors from the store (already downloaded in step 1)
factor_wide = store.read_factors(FF5_FACTORS + ["RF"])

# Portfolio returns, aligned to the factors and converted to excess of RF
port_ret_series = rs.portfolio_returns(portfolio.weights)
aligned = align_factors(port_ret_series, rs.dates, factor_wide, FF5_FACTORS)

model = estimate_factor_model(aligned, hac_lags=5)
print(f"Factor model fit on: {aligned.coverage}")  # actual window (see also the log warning)
print(model.summary())

# Factor exposures chart (from the reusable viz module)
from src.viz.plots import plot_factor_exposures
plot_factor_exposures(model).show()

# %%
print(decompose_factor_risk(model, aligned).summary())
print()
print(attribute_returns(model, aligned).summary())

# %% [markdown]
# ## 8. Dynamic Factor Evolution — betas that change over time

# %%
import plotly.graph_objects as go
roll = rolling_betas(aligned, window=252)
fig = go.Figure()
for f in FF5_FACTORS:
    fig.add_trace(go.Scatter(x=roll["dates"], y=roll["betas"][f], name=f, mode="lines"))
fig.update_layout(title="Rolling 252d factor betas", xaxis_title="date", yaxis_title="beta")
fig.show()

# %% [markdown]
# ## 9. PCA Risk Model + Hidden Concentration Detector
#
# How many latent factors really drive the portfolio? And is the portfolio
# as diversified as its weights suggest?

# %%
from src.analytics.pca.model import fit_pca
from src.analytics.pca.concentration import detect_hidden_concentration

pca_model = fit_pca(rs, method="covariance")
print(pca_model.summary())

# Scree plot
scree = pca_model.scree_data().to_pandas()
fig = px.bar(scree, x="pc", y="explained", title="Scree plot — explained variance per PC")
fig.add_scatter(x=scree["pc"], y=scree["cumulative"], name="cumulative", mode="lines+markers")
fig.show()

# %% [markdown]
# ### Hidden Concentration Detector
# Contrast between "apparent" diversification (weights) and "true" (risk).

# %%
report = detect_hidden_concentration(portfolio.weights, rs, model=pca_model)
print(report.summary())

# %% [markdown]
# ### Loadings heatmap — which assets load on which components

# %%
load_df = pca_model.loadings_dataframe(n_components=pca_model.n_factors_for(0.95)).to_pandas()
fig = px.imshow(
    load_df.set_index("ticker"),
    color_continuous_scale="RdBu", color_continuous_midpoint=0,
    title="PCA factor loadings", aspect="auto",
)
fig.show()

# %% [markdown]
# ## 10. Correlation Analytics
#
# Static structure (quasi-diagonal heatmap), dynamic (average correlation over
# time) and topology (MST — Mantegna's asset tree).

# %%
from src.analytics.correlation.matrices import correlation_matrix, correlation_distance
from src.analytics.correlation.rolling import average_correlation, rolling_pairwise_correlation
from src.analytics.correlation.clustering import cluster_correlations, reorder_correlation
from src.analytics.correlation.network import build_mst

# Heatmap reordered by cluster (block-diagonal)
clust = cluster_correlations(rs, n_clusters=3)
print(clust.summary())
corr = correlation_matrix(rs)
ordered = reorder_correlation(corr, clust.quasi_diagonal_order)
ordered_tickers = clust.ordered_tickers()

import plotly.express as px
fig = px.imshow(ordered, x=ordered_tickers, y=ordered_tickers,
                color_continuous_scale="RdBu", color_continuous_midpoint=0,
                title="Correlation matrix (quasi-diagonal order)")
fig.show()

# %% [markdown]
# ### Average correlation (systemic indicator)

# %%
avg = average_correlation(rs, window=63)
fig = px.line(avg.to_pandas(), x="date", y="avg_correlation",
              title="Average pairwise correlation (63d) — systemic correlation")
fig.show()

# %% [markdown]
# ### Risk topology — Minimum Spanning Tree

# %%
mst = build_mst(rs)
print(mst.summary())
print()
print(mst.edges_dataframe())

# %% [markdown]
# ### Risk topology — MST network graph

# %%
from src.viz.plots import plot_mst_network
plot_mst_network(mst).show()

# %% [markdown]
# ## 11. Stress Testing
#
# Two methodologies: **factor-based historical replay** (current betas ×
# factor returns over a real crisis — works for any era, even pre-ETF)
# and **parametric shocks** (hypothetical moves propagated through the betas).

# %%
from src.analytics.stress.scenarios import HISTORICAL_SCENARIOS, FACTOR_SHOCKS
from src.analytics.stress.historical import run_historical_factor
from src.analytics.stress.parametric import run_factor_shock

# Factor-based historical replays: use the already-downloaded FF5 factors (deep history)
factor_wide_all = store.read_factors(FF5_FACTORS)
for key in ["COVID_2020", "INFLATION_2022", "GFC_2008"]:
    scen = HISTORICAL_SCENARIOS[key]
    try:
        res = run_historical_factor(model, factor_wide_all, scen)
        print(res.summary())
        print()
    except ValueError as e:
        print(f"{key}: {e}\n")

# %% [markdown]
# ### Parametric factor shocks

# %%
for name, sh in FACTOR_SHOCKS.items():
    print(run_factor_shock(model, sh.moves, name=name).summary())
    print()

# %% [markdown]
# ### Stress summary chart
# PnL of each scenario, sorted from worst.

# %%
import plotly.express as px
scenarios_pnl = []
for key, scen in HISTORICAL_SCENARIOS.items():
    try:
        r = run_historical_factor(model, factor_wide_all, scen)
        scenarios_pnl.append({"scenario": key, "pnl": r.total_pnl, "type": "historical"})
    except ValueError:
        pass
for name, sh in FACTOR_SHOCKS.items():
    r = run_factor_shock(model, sh.moves, name=name)
    scenarios_pnl.append({"scenario": name, "pnl": r.total_pnl, "type": "parametric"})

import polars as pl
sdf = pl.DataFrame(scenarios_pnl).sort("pnl").to_pandas()
fig = px.bar(sdf, x="pnl", y="scenario", color="type", orientation="h",
             title="Stress test — PnL per scenario", text_auto=".1%")
fig.update_xaxes(tickformat=".0%")
fig.show()
