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
# ## 2b. Data quality gate
#
# Free data fails silently (unadjusted splits, stale prices, gaps). Run a
# quality report after ingestion and BEFORE trusting anything downstream. It
# detects and reports — it does not auto-correct; a human decides.

# %%
from src.data_quality.report import run_quality_report
quality = run_quality_report(store)
print(quality.summary())
# Inspect machine-readable findings:
# print(quality.to_dataframe())

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
from src.viz.plots import plot_performance, plot_rolling_volatility
plot_performance(port_returns, rs.dates.to_list(), title="Demo 60/40-ish — performance").show()

# Rolling volatility — the vol clustering a single headline sigma averages away
plot_rolling_volatility(port_returns, rs.dates.to_list(), window=63,
                        title="Demo 60/40-ish — rolling volatility").show()

# %% [markdown]
# ## 5. Risk decomposition
# Per-asset marginal and percent contribution to portfolio risk. The `<--` flag
# marks assets whose risk contribution exceeds their portfolio weight.

# %%
decomp = decompose_risk(portfolio.weights, rs, cov_method="ledoit_wolf")
print(decomp.summary())
print()
print(decomp.to_dataframe())

# Visual: capital weight vs % risk contribution. Where %RC overshoots the
# weight, that asset carries more risk than its size — the concentration tell.
from src.viz.plots import plot_risk_contribution
plot_risk_contribution(decomp, title="Demo — risk contribution vs weight").show()

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
# ## 7. Tail risk — beyond the Gaussian
#
# Historical VaR/CVaR only see the tail that occurred. Two extensions: the
# Cornish-Fisher (modified) VaR corrects the Gaussian quantile for skewness and
# kurtosis; Extreme Value Theory (peaks-over-threshold) fits a Generalised
# Pareto Distribution to the exceedances to model losses beyond the sample.
# Tail risk is a summary risk statistic, so it lives here next to performance
# and risk decomposition rather than at the end of the notebook.

# %%
from src.analytics.risk.tail import tail_risk_comparison, fit_evt_pot
from src.viz.plots import plot_var_distribution, plot_evt_tail

comp = tail_risk_comparison(port_returns, confidence=0.99)
print("Tail risk comparison (99%, 1-day):")
for k, v in comp.items():
    if v is None:
        print(f"  {k:20} n/a")
    elif "shape" in k:
        print(f"  {k:20} {v:>8.3f}")
    else:
        print(f"  {k:20} {v:>8.2%}")

# The whole VaR family on top of the actual return distribution
plot_var_distribution(port_returns, confidence=0.99).show()

# %% [markdown]
# ### EVT peaks-over-threshold detail
# The GPD shape parameter xi summarises tail heaviness (xi > 0 heavy-tailed).
# The diagnostic plot overlays the fitted GPD survival on the empirical
# exceedances — a good fit tracks the points into the tail.

# %%
print(fit_evt_pot(port_returns, confidence=0.99, threshold_quantile=0.95).summary())
plot_evt_tail(port_returns, confidence=0.99).show()

# %% [markdown]
# ## 8. Factor Engine — exposures, systematic risk, attribution
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
# ## 9. Dynamic Factor Evolution — betas that change over time

# %%
import plotly.graph_objects as go
roll = rolling_betas(aligned, window=252)
fig = go.Figure()
for f in FF5_FACTORS:
    fig.add_trace(go.Scatter(x=roll["dates"], y=roll["betas"][f], name=f, mode="lines"))
fig.update_layout(title="Rolling 252d factor betas", xaxis_title="date", yaxis_title="beta")
fig.show()

# %% [markdown]
# ## 10. PCA Risk Model + Hidden Concentration Detector
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
# ## 11. Correlation Analytics
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
# ## 12. Stress Testing
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

# %% [markdown]
# ## 13. Regime Detection
#
# Two approaches: a Gaussian hidden Markov model (Markov-switching, primary) and
# a volatility-state baseline. The regime is estimated on the broad market
# (Mkt-RF); the portfolio's analytics are then recomputed within each regime.

# %%
from src.analytics.regime.hmm import fit_regimes
from src.analytics.regime.volatility_states import volatility_states
from src.analytics.regime.conditional import (
    regime_conditional_stats, regime_conditional_beta,
    regime_conditional_avg_correlation, align_states_to_returns,
)
from src.domain.returns import ReturnSeries

# Estimate regimes on the market factor
mkt = store.read_factors(["Mkt-RF"]).drop_nulls().sort("date")
mkt_ret = mkt["Mkt-RF"].to_numpy()
mkt_dates = mkt["date"].to_list()

regime = fit_regimes(mkt_ret, mkt_dates, n_states=2, search_reps=20)
print(regime.summary())
print()
print("Transition matrix (row-stochastic):")
print(regime.transition_matrix.round(3))

# %% [markdown]
# The summary reports, per state, the annualised mean and volatility, the
# average duration, and the unconditional frequency. The transition matrix
# diagonal gives the persistence of each regime.

# %% [markdown]
# ### Regime timeline — market shaded by detected state

# %%
from src.viz.plots import plot_regime_timeline
plot_regime_timeline(mkt_ret, mkt_dates, regime.states, regime.labels,
                     title="Market (Mkt-RF) with detected stress regimes").show()

# %% [markdown]
# ### Portfolio analytics conditioned on regime
# Map the market regimes onto the portfolio calendar (inner-join on date), then
# recompute the portfolio's statistics within each regime.

# %%
port_states = align_states_to_returns(mkt_dates, regime.states, rs.dates)
valid = port_states >= 0

port_r = rs.portfolio_returns(portfolio.weights).to_numpy()[valid]
states_v = port_states[valid]

print("Per-regime portfolio stats (annualised return, volatility):")
print(regime_conditional_stats(port_r, states_v, regime.labels))

# Regime-conditional market beta, on the common portfolio/market window
mkt_df = mkt.rename({"Mkt-RF": "mkt"})
port_df = pl.DataFrame({"date": rs.dates, "port": rs.portfolio_returns(portfolio.weights)})
joined = port_df.join(mkt_df.select(["date", "mkt"]), on="date", how="inner").drop_nulls()
js = align_states_to_returns(mkt_dates, regime.states, joined["date"])
jv = js >= 0
print()
print("Per-regime market beta:")
print(regime_conditional_beta(joined["port"].to_numpy()[jv], joined["mkt"].to_numpy()[jv],
                              js[jv], regime.labels))

# Regime-conditional average correlation across the universe
rs_masked = ReturnSeries(rs.data.filter(pl.Series(valid)), rs.tickers)
print()
print("Per-regime average pairwise correlation:")
print(regime_conditional_avg_correlation(rs_masked, states_v, regime.labels))

# %% [markdown]
# The tables report the portfolio's volatility, market beta and average pairwise
# correlation computed separately within the calm and stress states. Volatility
# is markedly higher in the stress state. Beta and average correlation differ by
# regime as well; the size and direction of those differences depend on the
# specific portfolio and universe.

# %% [markdown]
# ### Volatility-state baseline
# A transparent quantile classifier on rolling volatility, for comparison with
# the Markov model above.

# %%
import plotly.graph_objects as go
stress_idx = regime.labels.index("stress")
fig = go.Figure(go.Scatter(
    x=regime.dates, y=regime.smoothed_probs[:, stress_idx],
    fill="tozeroy", line=dict(color="#C44E52"),
))
fig.update_layout(title="P(stress regime) over time", yaxis_title="probability",
                  yaxis_range=[0, 1])
fig.show()

# %% [markdown]
# ### Regime-conditional risk — the payoff
# Map the market regimes onto the portfolio calendar (inner-join on date, so the
# regime frequencies here match the overlapping window, not the full market
# history), then recompute risk WITHIN each regime.

# %%
# Align market-estimated states onto the portfolio's return dates
states_aligned = align_states_to_returns(regime.dates, regime.states, rs.dates)
valid = states_aligned >= 0

port_r = rs.portfolio_returns(portfolio.weights).to_numpy()[valid]
states_v = states_aligned[valid]
rs_valid = ReturnSeries(rs.data.filter(pl.Series(valid)), rs.tickers)

# Build market excess on the SAME aligned window for the conditional beta
mkt_r = mkt.join(rs.data.select("date"), on="date", how="inner").sort("date")
mkt_aligned = mkt_r["Mkt-RF"].to_numpy()

print("Per-regime portfolio stats:")
print(regime_conditional_stats(port_r, states_v, regime.labels))
print()
print("Per-regime market beta:")
print(regime_conditional_beta(port_r, mkt_aligned, states_v, regime.labels))
print()
print("Per-regime average correlation:")
print(regime_conditional_avg_correlation(rs_valid, states_v, regime.labels))

# %% [markdown]
# The robust, unambiguous effect is **volatility**: it roughly doubles from the calm 
# to the stress regime, and the
# return/vol ratio flips from strongly positive to negative — the portfolio is a
# different animal in stress. **Beta and correlation, however, barely move here**
# (beta even ticks *down* slightly). That is not a bug, it is a real and
# instructive result:
#
# - Beta is $\mathrm{cov}(p,m)/\mathrm{var}(m)$. In stress *both* the covariance
#   and the market variance blow up and largely cancel, so beta can stay flat or
#   fall even as absolute risk rises. The thing that genuinely explodes is
#   volatility, not necessarily beta.
# - Average correlation barely rises because this 5-asset book is already highly
#   US-equity-driven, so correlation starts high (~0.28) in calm and has little
#   room to climb. The "diversification breaks down in crises" effect is much
#   sharper on broad, genuinely cross-asset universes.
#


# %% [markdown]
# ### Baseline cross-check — volatility states
# A transparent quantile classifier; if the HMM doesn't add value over this,
# the extra machinery isn't earning its keep.

# %%
vs = volatility_states(mkt["Mkt-RF"].to_numpy(), mkt["date"].to_list(), window=21, n_states=2)
print(vs.summary())

