"""
Dashboard builder — runs the analytics and assembles the JSON payload.

This module is the only place that knows both the analytics API and the payload
schema. It computes nothing itself: every number comes from `src.analytics` and
every chart from `src.viz.plots`, unchanged. That constraint is deliberate — if
a figure needs new logic, the logic belongs in the analytics or viz layer where
it is unit-tested, not in a serialisation script.

Two structural choices worth knowing about:

  * Shared expensive objects (returns, covariance, factor model, PCA, regimes)
    hang off a lazily-evaluated `Context`, so the factor model is estimated once
    and reused by the factors, stress and structure sections.

  * Every section is wrapped so that a failure degrades to a visible error panel
    instead of aborting the build. Free data is patchy and the factor library
    publishes with a lag; a book with 8 months of history genuinely cannot
    support a 252-day rolling beta, and the honest response is to say so on that
    page rather than lose the seven pages that do work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import polars as pl
from loguru import logger

from src.analytics.correlation.clustering import (
    cluster_correlations,
    reorder_correlation,
)
from src.analytics.correlation.matrices import correlation_matrix
from src.analytics.correlation.network import build_mst
from src.analytics.correlation.rolling import average_correlation
from src.analytics.factors.attribution import attribute_returns, decompose_factor_risk
from src.analytics.factors.engine import estimate_factor_model, rolling_betas
from src.analytics.factors.prepare import FF5_FACTORS, align_factors
from src.analytics.optimization import (
    LONG_SHORT,
    efficient_frontier,
    hrp,
    min_variance,
    risk_parity,
)
from src.analytics.pca.concentration import detect_hidden_concentration
from src.analytics.pca.model import fit_pca
from src.analytics.performance import align_risk_free, compute_metrics
from src.analytics.regime.conditional import (
    align_states_to_returns,
    regime_conditional_avg_correlation,
    regime_conditional_beta,
    regime_conditional_stats,
)
from src.analytics.regime.hmm import fit_regimes
from src.analytics.risk.covariance import estimate_covariance
from src.analytics.risk.decomposition import decompose_risk, rolling_risk_contribution
from src.analytics.risk.tail import fit_evt_pot, tail_risk_comparison
from src.analytics.stress.historical import run_historical_factor
from src.analytics.stress.parametric import run_factor_shock
from src.analytics.stress.scenarios import FACTOR_SHOCKS, HISTORICAL_SCENARIOS
from src.domain.returns import ReturnSeries
from src.export import encode
from src.export.spec import PortfolioSpec
from src.store.parquet_store import ParquetStore
from src.viz import plots

# ──────────────────────────────────────────────────────────────────────────
# Shared context
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class Context:
    """Everything the sections share. Expensive members are computed on first use."""

    spec: PortfolioSpec
    store: ParquetStore
    rs: ReturnSeries
    warnings: list[str] = field(default_factory=list)

    _cache: dict = field(default_factory=dict, repr=False)

    def _memo(self, key: str, compute: Callable):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    # --- portfolio-level series ---

    @property
    def weights(self) -> dict[str, float]:
        return self.spec.weights

    @property
    def port_returns(self) -> np.ndarray:
        return self._memo(
            "port_returns",
            lambda: self.rs.portfolio_returns(self.weights).to_numpy(),
        )

    @property
    def dates(self) -> list:
        return self.rs.dates.to_list()

    # --- estimated models ---

    @property
    def covariance(self):
        return self._memo("cov", lambda: estimate_covariance(
            self.rs, method=self.spec.cov_method
        ))

    @property
    def pca(self):
        return self._memo("pca", lambda: fit_pca(
            self.rs, method=self.spec.settings.pca_method
        ))

    @property
    def factors_wide(self) -> pl.DataFrame:
        return self._memo(
            "factors_wide",
            lambda: self.store.read_factors(FF5_FACTORS + ["RF"]),
        )

    @property
    def aligned(self):
        """Portfolio excess returns aligned to the FF5 factors."""
        def compute():
            fw = self.factors_wide
            if fw.is_empty():
                raise ValueError(
                    "No factor data in the store. Run the build without "
                    "--skip-ingest, or check the Kenneth French download."
                )
            # Passing the calendar is what arms the UNSUPPORTED policy: on a
            # crypto book this raises instead of returning betas on factors
            # built from US equity portfolios. The section wrapper turns that
            # into a visible error panel, which is the intended outcome.
            a = align_factors(
                self.rs.portfolio_returns(self.weights),
                self.rs.dates, fw, FF5_FACTORS,
                calendar=self.rs.calendar,
            )
            dropped = self.rs.n_obs - a.n_obs
            if dropped > 0:
                self.warnings.append(
                    f"Factor model covers {a.coverage}; the most recent "
                    f"{dropped} return days are excluded because the Fama-French "
                    f"library publishes with a lag."
                )
            return a
        return self._memo("aligned", compute)

    @property
    def factor_model(self):
        return self._memo("factor_model", lambda: estimate_factor_model(
            self.aligned, hac_lags=self.spec.settings.hac_lags
        ))

    @property
    def market(self) -> tuple[np.ndarray, list]:
        """The broad-market series regimes are estimated on (Mkt-RF, full history)."""
        def compute():
            mkt = self.store.read_factors(["Mkt-RF"]).drop_nulls().sort("date")
            if mkt.is_empty():
                raise ValueError(
                    "No Mkt-RF factor data available for regime detection."
                )
            return mkt["Mkt-RF"].to_numpy(), mkt["date"].to_list()
        return self._memo("market", compute)

    @property
    def regimes(self):
        def compute():
            mkt_ret, mkt_dates = self.market
            logger.info(
                f"Fitting {self.spec.settings.regime_states}-state HMM on "
                f"{len(mkt_ret)} market observations "
                f"({self.spec.settings.regime_search_reps} EM restarts)..."
            )
            return fit_regimes(
                mkt_ret, mkt_dates,
                n_states=self.spec.settings.regime_states,
                search_reps=self.spec.settings.regime_search_reps,
            )
        return self._memo("regimes", compute)


# ──────────────────────────────────────────────────────────────────────────
# Sections
# ──────────────────────────────────────────────────────────────────────────

def build_overview(ctx: Context) -> dict:
    r = ctx.port_returns
    fw = ctx.factors_wide

    # Descriptors that do not involve the risk-free rate — total return, CAGR,
    # volatility, drawdown, VaR, the moments — are computed on the FULL return
    # history, so they describe the same period the equity chart draws. The rf
    # argument is irrelevant to every field read out of this object.
    metrics = compute_metrics(r, rf=0.0)

    # Sharpe and Sortino are different: they need excess returns, so they can
    # only be computed where a risk-free observation exists. The French RF
    # series ends weeks before the price data, so these two are reported on a
    # shorter window and the hint says which. Using the real daily series rather
    # than a constant matters — across this sample the policy rate went from
    # ~2% to ~0 to >5%, and a fixed rate silently misprices both ratios.
    rf_note = ""
    if not fw.is_empty() and "RF" in fw.columns:
        r_aligned, rf_daily, aligned_dates = align_risk_free(
            ctx.rs.portfolio_returns(ctx.weights), ctx.rs.dates, fw
        )
        excess = compute_metrics(r_aligned, rf=rf_daily)
        rf_hint = (f"Real daily risk-free, {excess.n_observations:,} obs "
                   f"to {aligned_dates[-1]}")
        if excess.n_observations < metrics.n_observations:
            rf_note = (
                f"Sharpe and Sortino are computed on {excess.n_observations:,} "
                f"observations ending {aligned_dates[-1]}, because they need the "
                f"daily risk-free rate and the Fama-French series stops there. "
                f"Every other statistic on this page uses all "
                f"{metrics.n_observations:,} days."
            )
    else:
        from src.config import settings as _s
        excess = compute_metrics(r, rf=_s.risk_free_rate)
        rf_hint = f"Constant {_s.risk_free_rate:.1%} fallback"
        rf_note = (
            "No daily risk-free series available; Sharpe and Sortino use the "
            "constant fallback rate from config."
        )

    weights_fig = plots.plot_weights_comparison(
        {ctx.spec.name: ctx.weights, "equal weight": ctx.spec.equal_weights()},
        title="Allocation vs equal weight",
    )

    notes = [
        "Sharpe and Sortino use the arithmetic mean of daily excess returns, "
        "annualised — the textbook definition. The CAGR shown alongside is "
        "geometric and is a descriptor, not the Sharpe numerator.",
    ]
    if rf_note:
        notes.append(rf_note)

    return encode.section(
        "overview", "Overview",
        subtitle=ctx.spec.description or f"{len(ctx.spec.positions)} positions",
        stats=[
            encode.stat("Total return", metrics.total_return, "percent",
                        "Compounded over the whole window"),
            encode.stat("Ann. return (CAGR)", metrics.annualized_return, "percent",
                        "Geometric"),
            encode.stat("Ann. volatility", metrics.annualized_volatility, "percent"),
            encode.stat("Sharpe", excess.sharpe_ratio, "ratio", rf_hint),
            encode.stat("Sortino", excess.sortino_ratio, "ratio",
                        "Downside deviation in the denominator"),
            encode.stat("Calmar", metrics.calmar_ratio, "ratio",
                        "CAGR over max drawdown"),
            encode.stat("Max drawdown", metrics.max_drawdown, "percent", tone="bad"),
            encode.stat("Drawdown duration", metrics.max_drawdown_duration, "integer",
                        "Longest run of trading days below the previous peak"),
            encode.stat("VaR 95% (1d)", metrics.var_95, "percent", "Historical"),
            encode.stat("CVaR 95% (1d)", metrics.cvar_95, "percent",
                        "Mean loss beyond VaR"),
            encode.stat("Skewness", metrics.skewness, "number"),
            encode.stat("Excess kurtosis", metrics.excess_kurtosis, "number",
                        "0 = Gaussian tails"),
            encode.stat("Hit rate", metrics.hit_rate, "percent",
                        "Share of positive days"),
            encode.stat("Observations", metrics.n_observations, "integer"),
        ],
        figures=[
            encode.figure(
                plots.plot_performance(
                    r, ctx.dates, title=f"{ctx.spec.name} — performance"
                ),
                "performance", "Growth of 1 and drawdown",
            ),
            encode.figure(
                plots.plot_rolling_volatility(
                    r, ctx.dates, window=ctx.spec.settings.rolling_window,
                    title="Rolling volatility",
                ),
                "rolling_vol", "Rolling volatility",
                "Volatility clusters. A single headline sigma averages this away.",
            ),
            encode.figure(weights_fig, "weights", "Allocation"),
        ],
        notes=notes,
    )


def build_risk(ctx: Context) -> dict:
    decomp = decompose_risk(ctx.weights, ctx.rs, cov_method=ctx.spec.cov_method)
    cov = ctx.covariance

    df = decomp.to_dataframe().sort("pct_risk_contribution", descending=True)
    overweight = df.filter(
        pl.col("pct_risk_contribution") > 1.5 * pl.col("weight")
    )["ticker"].to_list()

    rolling = rolling_risk_contribution(
        ctx.weights, ctx.rs,
        window=ctx.spec.settings.rolling_window, cov_method="ewma",
    )

    figures = [
        encode.figure(
            plots.plot_risk_contribution(decomp, title="Risk contribution vs weight"),
            "risk_contribution", "Risk contribution vs weight",
            "Where the risk bar overshoots the weight bar, that asset carries "
            "more risk than its size implies.",
        ),
    ]
    if not rolling.is_empty():
        figures.append(encode.figure(
            plots.plot_rolling_risk_contribution(
                rolling,
                title=(f"Rolling % risk contribution "
                       f"({ctx.spec.settings.rolling_window}d, EWMA)"),
            ),
            "rolling_rc", "Risk mix through time",
            "Static weights, drifting risk. The mix is not what the allocation "
            "suggests.",
        ))

    notes = [
        f"Covariance estimator: {cov.method}"
        + (f" (shrinkage δ = {cov.shrinkage:.3f})" if cov.shrinkage is not None else "")
        + f", condition number {cov.condition_number:.1f}.",
    ]
    if overweight:
        notes.append(
            f"Risk-concentrated relative to weight: {', '.join(overweight)} — "
            f"each contributes more than 1.5× its capital share."
        )

    return encode.section(
        "risk", "Risk decomposition",
        subtitle="Where portfolio risk actually sits, asset by asset",
        stats=[
            encode.stat("Portfolio volatility", decomp.portfolio_volatility, "percent",
                        "Annualised"),
            encode.stat("Largest risk contributor",
                        float(df["pct_risk_contribution"][0]),
                        "percent", f"{df['ticker'][0]}"),
            encode.stat("Covariance condition", cov.condition_number, "number",
                        "κ(Σ) — large means unstable to invert"),
            encode.stat("Estimator", cov.method, "text"),
        ],
        figures=figures,
        tables=[encode.table(
            "decomposition", "Per-asset decomposition", df,
            formats={
                "weight": "percent", "mcr": "number",
                "risk_contribution": "percent", "pct_risk_contribution": "percent",
            },
            note="MCR is the marginal contribution to risk: the change in "
                 "portfolio volatility per unit increase in that weight.",
        )],
        notes=notes,
    )


def build_tail(ctx: Context) -> dict:
    r = ctx.port_returns
    conf = ctx.spec.settings.var_confidence
    comp = tail_risk_comparison(r, confidence=conf)

    stats = [
        encode.stat("Gaussian VaR", comp["gaussian_var"], "percent",
                    f"{conf:.0%}, 1-day — assumes normality"),
        encode.stat("Cornish-Fisher VaR", comp["cornish_fisher_var"], "percent",
                    "Corrected for skew and kurtosis"),
        encode.stat("Historical VaR", comp["historical_var"], "percent",
                    "Empirical quantile"),
        encode.stat("Historical CVaR", comp["historical_cvar"], "percent",
                    "Mean loss beyond VaR"),
    ]
    figures = [encode.figure(
        plots.plot_var_distribution(r, confidence=conf),
        "var_distribution", "Return distribution and VaR",
        "The gap between the Gaussian marker and the historical one is the cost "
        "of assuming normality.",
    )]
    notes = []

    try:
        evt = fit_evt_pot(
            r, confidence=conf,
            threshold_quantile=ctx.spec.settings.evt_threshold_quantile,
        )
        tail_desc = ("heavy-tailed" if evt.shape > 0.02
                     else "thin-tailed" if evt.shape < -0.02 else "exponential")
        stats += [
            encode.stat("EVT VaR", evt.var, "percent", "Peaks-over-threshold, GPD"),
            encode.stat("EVT CVaR", evt.cvar, "percent",
                        "Infinite when ξ ≥ 1" if not np.isfinite(evt.cvar) else ""),
            encode.stat("GPD shape ξ", evt.shape, "number", tail_desc),
            encode.stat("Exceedances", evt.n_exceedances, "integer",
                        f"of {evt.n_total} observations"),
        ]
        figures.append(encode.figure(
            plots.plot_evt_tail(
                r, confidence=conf,
                threshold_quantile=ctx.spec.settings.evt_threshold_quantile,
            ),
            "evt_tail", "EVT tail fit",
            "A good fit tracks the empirical points into the tail.",
        ))
        notes.append(
            f"EVT fits a Generalised Pareto Distribution to the {evt.n_exceedances} "
            f"losses beyond the {ctx.spec.settings.evt_threshold_quantile:.0%} "
            f"threshold, which is what lets it price losses larger than any "
            f"observed. It is sensitive to that threshold choice."
        )
    except ValueError as e:
        notes.append(f"EVT could not be fitted: {e}")

    return encode.section(
        "tail", "Tail risk",
        subtitle="What the Gaussian assumption misses",
        stats=stats, figures=figures, notes=notes,
    )


def build_factors(ctx: Context) -> dict:
    model = ctx.factor_model
    aligned = ctx.aligned
    risk = decompose_factor_risk(model, aligned)
    attrib = attribute_returns(model, aligned)

    betas_df = pl.DataFrame({
        "factor": model.factor_names,
        "beta": [model.betas[f] for f in model.factor_names],
        "t_stat": [model.beta_tstats[f] for f in model.factor_names],
        "p_value": [model.beta_pvalues[f] for f in model.factor_names],
        "pct_of_variance": [risk.factor_variance_contribution[f]
                            for f in model.factor_names],
        "return_contribution": [attrib.factor_contribution[f]
                                for f in model.factor_names],
    })

    figures = [encode.figure(
        plots.plot_factor_exposures(model),
        "exposures", "Factor exposures",
        "Solid bars are significant at p < 0.05 using HAC standard errors.",
    )]

    window = ctx.spec.settings.beta_window
    if aligned.n_obs > window:
        figures.append(encode.figure(
            plots.plot_rolling_betas(
                rolling_betas(aligned, window=window),
                title=f"Rolling factor betas ({window}d)",
            ),
            "rolling_betas", "Exposures through time",
            "These paths are why constant-beta stress tests understate crisis losses.",
        ))

    return encode.section(
        "factors", "Factor model",
        subtitle=f"Fama-French 5, fitted on {aligned.coverage}",
        stats=[
            encode.stat("R²", model.r_squared, "number",
                        "Share of variance explained by the five factors"),
            encode.stat("Adjusted R²", model.adj_r_squared, "number"),
            encode.stat("Alpha (ann.)", model.alpha_annualized, "percent",
                        f"t = {model.alpha_tstat:+.2f}, p = {model.alpha_pvalue:.3f}",
                        tone="good" if model.alpha_pvalue < 0.05
                        and model.alpha > 0 else "neutral"),
            encode.stat("Systematic risk", risk.pct_systematic, "percent",
                        "Share of variance from the factors"),
            encode.stat("Specific risk", risk.pct_specific, "percent",
                        "Idiosyncratic — what the factors do not explain"),
            encode.stat("Idiosyncratic vol", model.idiosyncratic_vol, "percent",
                        "Annualised residual volatility"),
            encode.stat("Total excess return", attrib.total_excess_annualized,
                        "percent", "Annualised, arithmetic"),
            encode.stat("Observations", model.n_obs, "integer",
                        f"HAC lags: {model.hac_lags}" if model.hac_lags else "OLS SE"),
        ],
        figures=figures,
        tables=[encode.table(
            "betas", "Exposures, risk and return by factor", betas_df,
            formats={
                "beta": "number", "t_stat": "number", "p_value": "number",
                "pct_of_variance": "percent", "return_contribution": "percent",
            },
            note="Return contributions reconcile exactly: alpha plus the factor "
                 f"contributions equals the total excess return to within "
                 f"{abs(attrib.reconciliation_error):.1e}.",
        )],
        notes=[
            "Betas are estimated by OLS with Newey-West (HAC) standard errors, "
            "which correct the autocorrelation and heteroskedasticity that plain "
            "OLS standard errors understate on daily data.",
        ],
    )


def build_structure(ctx: Context) -> dict:
    pca = ctx.pca
    report = detect_hidden_concentration(ctx.weights, ctx.rs, model=pca)
    n_assets = len(ctx.rs.tickers)

    corr = correlation_matrix(ctx.rs)
    figures = [
        encode.figure(plots.plot_scree(pca), "scree",
                      "Explained variance per component"),
        encode.figure(
            plots.plot_loadings_heatmap(pca, n_components=pca.n_factors_for(0.95)),
            "loadings", "Factor loadings",
            "Which assets load on which latent component, and with what sign.",
        ),
    ]

    # Clustering needs at least as many assets as clusters, and the quasi-diagonal
    # reordering is what turns a noisy heatmap into a readable block structure.
    n_clusters = min(ctx.spec.settings.n_clusters, max(n_assets - 1, 1))
    tables = []
    if n_assets >= 3:
        clust = cluster_correlations(ctx.rs, n_clusters=n_clusters)
        ordered = reorder_correlation(corr, clust.quasi_diagonal_order)
        figures.append(encode.figure(
            plots.plot_correlation_heatmap(
                ordered, clust.ordered_tickers(),
                title="Correlation matrix (cluster order)",
            ),
            "correlation", "Correlation structure",
            "Reordered so correlated assets sit adjacent — the blocks are the "
            "clusters.",
        ))
        tables.append(encode.table(
            "clusters", "Correlation clusters",
            [{"cluster": cid, "members": ", ".join(m)}
             for cid, m in sorted(clust.clusters().items())],
            note="Hierarchical clustering on the Mantegna distance "
                 "d = √(2(1−ρ)). This is also the ordering step used by "
                 "Hierarchical Risk Parity.",
        ))
    else:
        figures.append(encode.figure(
            plots.plot_correlation_heatmap(corr, ctx.rs.tickers),
            "correlation", "Correlation structure",
        ))

    avg = average_correlation(ctx.rs, window=ctx.spec.settings.corr_window)
    if not avg.is_empty():
        figures.append(encode.figure(
            plots.plot_average_correlation(
                avg,
                title=(f"Average pairwise correlation "
                       f"({ctx.spec.settings.corr_window}d)"),
            ),
            "avg_correlation", "Systemic correlation",
            "Spikes toward 1 mark the episodes where diversification stops working.",
        ))

    mst = build_mst(ctx.rs)
    figures.append(encode.figure(
        plots.plot_mst_network(mst), "mst", "Risk topology",
        "The minimum spanning tree keeps only the strongest links needed to "
        "connect every asset. Hubs are systemic; leaves are the real diversifiers.",
    ))
    tables.append(encode.table(
        "mst_edges", "Minimum spanning tree — links", mst.edges_dataframe(),
        formats={"distance": "number", "correlation": "number"},
    ))

    ratio = (report.effective_bets_risk / report.effective_bets_weights
             if report.effective_bets_weights else 1.0)

    return encode.section(
        "structure", "Latent structure",
        subtitle="Hidden concentration, correlation topology and principal components",
        stats=[
            encode.stat("Holdings", report.n_holdings, "integer"),
            encode.stat("Effective bets (weights)", report.effective_bets_weights,
                        "number", "Inverse Herfindahl — the naive count"),
            encode.stat("Effective bets (risk)", report.effective_bets_risk, "number",
                        "Entropy of the variance shares across components",
                        tone="bad" if ratio < 0.5 else "neutral"),
            encode.stat("Top component share", report.top_pc_share, "percent",
                        "Share of portfolio variance from PC1"),
            encode.stat("Components for 90%", pca.n_factors_for(0.90), "integer",
                        f"of {n_assets} assets"),
            encode.stat("Hubs", ", ".join(t for t, _ in mst.hubs(2)), "text",
                        "Most connected nodes in the tree"),
        ],
        figures=figures, tables=tables,
        notes=[
            report.verdict() + ". The effective number of bets compares the "
            "diversification the weights imply against the diversification the "
            "risk actually delivers; a large gap means the book is a more "
            "concentrated position than it looks.",
        ],
    )


def build_stress(ctx: Context) -> dict:
    model = ctx.factor_model
    factors_all = ctx.store.read_factors(FF5_FACTORS)

    rows, summary = [], []
    for key, scen in HISTORICAL_SCENARIOS.items():
        try:
            res = run_historical_factor(model, factors_all, scen)
        except ValueError as e:
            logger.warning(f"Scenario {key} skipped: {e}")
            continue
        summary.append({"scenario": key, "pnl": res.total_pnl, "type": "historical"})
        rows.append({
            "scenario": key, "type": "historical", "window": res.window,
            "pnl": res.total_pnl, "max_drawdown": res.max_drawdown,
            "description": scen.description,
        })

    for name, shock in FACTOR_SHOCKS.items():
        res = run_factor_shock(model, shock.moves, name=name)
        summary.append({"scenario": name, "pnl": res.total_pnl, "type": "parametric"})
        rows.append({
            "scenario": name, "type": "parametric", "window": "instantaneous",
            "pnl": res.total_pnl, "max_drawdown": None,
            "description": shock.description,
        })

    if not summary:
        raise ValueError(
            "No stress scenario could be evaluated — the factor history does "
            "not cover any of the defined windows."
        )

    worst = min(summary, key=lambda s: s["pnl"])

    return encode.section(
        "stress", "Stress testing",
        subtitle="Historical replays and parametric shocks through the factor betas",
        stats=[
            encode.stat("Worst scenario", worst["scenario"], "text"),
            encode.stat("Worst-case PnL", worst["pnl"], "percent", tone="bad"),
            encode.stat("Scenarios evaluated", len(summary), "integer"),
        ],
        figures=[encode.figure(
            plots.plot_stress_summary(summary), "stress_summary",
            "PnL by scenario",
        )],
        tables=[encode.table(
            "scenarios", "Scenario detail", rows,
            formats={"pnl": "percent", "max_drawdown": "percent"},
        )],
        notes=[
            "Historical scenarios are replayed by applying the portfolio's "
            "current factor betas to the factor returns of that window, so they "
            "work for crises that predate the ETFs themselves. They capture the "
            "systematic PnL only.",
            "These figures likely UNDERSTATE the true loss: the betas are "
            "full-sample and held constant, while in real crises betas and "
            "correlations typically rise. The rolling-beta chart on the factor "
            "page shows how much that assumption is violated.",
        ],
    )


def _slice_regimes(regime, start: str):
    """
    Restrict the regime series to the portfolio window for plotting.

    The model is fitted on the full market history — a century of data gives far
    better transition-probability estimates — but charting 26,000 days would add
    megabytes to the payload and thousands of shaded bands to the figure, for a
    period the portfolio did not exist in.
    """
    import datetime as dt

    cutoff = dt.date.fromisoformat(start)
    idx = [i for i, d in enumerate(regime.dates) if d >= cutoff]
    lo = idx[0] if idx else 0
    # Slice the model as a whole. Replacing a subset of its arrays would
    # leave the rest full-length and silently misaligned against the dates.
    sliced = replace(
        regime,
        dates=regime.dates[lo:],
        states=regime.states[lo:],
        smoothed_probs=regime.smoothed_probs[lo:],
        filtered_probs=regime.filtered_probs[lo:],
    )
    return sliced, lo


def build_regimes(ctx: Context) -> dict:
    regime = ctx.regimes
    mkt_ret, mkt_dates = ctx.market

    regime_w, lo = _slice_regimes(regime, ctx.spec.start)
    dates_w, states_w = regime_w.dates, regime_w.states
    mkt_w = mkt_ret[lo:]

    freqs = np.bincount(regime.states, minlength=regime.n_states) / len(regime.states)
    regime_df = pl.DataFrame({
        "regime": regime.labels,
        "ann_return": [float((1 + m) ** 252 - 1) for m in regime.means],
        "ann_volatility": [float(v * np.sqrt(252)) for v in regime.volatilities],
        "avg_duration_days": [float(d) for d in regime.expected_durations],
        "frequency": [float(f) for f in freqs],
        "persistence": [float(regime.transition_matrix[i, i])
                        for i in range(regime.n_states)],
    })

    # Map market regimes onto the portfolio calendar and recompute within each.
    states_aligned = align_states_to_returns(regime.dates, regime.states, ctx.rs.dates)
    valid = states_aligned >= 0
    tables = [encode.table(
        "regime_params", "Regime parameters (full market history)", regime_df,
        formats={
            "ann_return": "percent", "ann_volatility": "percent",
            "avg_duration_days": "number", "frequency": "percent",
            "persistence": "percent",
        },
        note="Persistence is the diagonal of the transition matrix: the "
             "probability of still being in that regime tomorrow.",
    )]

    conditional_note = ""
    if valid.sum() > 10:
        port_r = ctx.port_returns[valid]
        states_v = states_aligned[valid]
        rs_valid = ReturnSeries(
            data=ctx.rs.data.filter(pl.Series(valid)), tickers=ctx.rs.tickers
        )
        tables.append(encode.table(
            "regime_stats", "Portfolio behaviour by regime",
            regime_conditional_stats(port_r, states_v, regime.labels),
            formats={
                "frequency": "percent", "ann_return": "percent",
                "ann_volatility": "percent", "return_vol_ratio": "number",
                "n_days": "integer",
            },
        ))
        tables.append(encode.table(
            "regime_corr", "Average correlation by regime",
            regime_conditional_avg_correlation(rs_valid, states_v, regime.labels),
            formats={"avg_correlation": "number", "n_days": "integer"},
        ))

        # Beta needs the market on the SAME dates as the portfolio.
        mkt_df = pl.DataFrame({"date": mkt_dates, "mkt": mkt_ret})
        port_df = pl.DataFrame({
            "date": ctx.rs.dates,
            "port": ctx.rs.portfolio_returns(ctx.weights),
        })
        joined = port_df.join(mkt_df, on="date", how="inner").drop_nulls().sort("date")
        js = align_states_to_returns(regime.dates, regime.states, joined["date"])
        jv = js >= 0
        if jv.sum() > 10:
            tables.append(encode.table(
                "regime_beta", "Market beta by regime",
                regime_conditional_beta(
                    joined["port"].to_numpy()[jv], joined["mkt"].to_numpy()[jv],
                    js[jv], regime.labels,
                ),
                formats={"beta": "number", "n_days": "integer"},
            ))
            conditional_note = (
                "Beta is cov(p,m)/var(m). In stress both the covariance and the "
                "market variance inflate and can largely cancel, so beta may stay "
                "flat or even fall while absolute risk rises sharply. Volatility "
                "is the robust effect; beta and correlation are conditional."
            )

    probs = regime.current_probabilities()
    notes = [
        f"Regimes are estimated on the broad market (Mkt-RF) over "
        f"{len(mkt_ret):,} observations, not on the portfolio: a regime is a "
        f"property of the environment, which the portfolio then inherits. "
        f"States are canonicalised by ascending volatility, so labels stay "
        f"stable across refits.",
        f"Charts are restricted to the portfolio window from {ctx.spec.start}; "
        f"the parameter table above reflects the full fitted history.",
    ]
    if conditional_note:
        notes.append(conditional_note)

    return encode.section(
        "regimes", "Market regimes",
        subtitle=f"{regime.n_states}-state Markov-switching model",
        stats=[
            encode.stat("Current regime", regime.current_label, "text",
                        tone="bad" if regime.current_state == regime.n_states - 1
                        else "good"),
            encode.stat("Confidence", probs[regime.current_label], "percent",
                        "Smoothed probability on the last observation"),
            encode.stat("Stress frequency", float(freqs[-1]), "percent",
                        f"Share of history in the {regime.labels[-1]} state"),
            encode.stat("Stress persistence",
                        float(regime.transition_matrix[-1, -1]), "percent",
                        "Probability of remaining in stress tomorrow"),
            encode.stat("Log-likelihood", regime.log_likelihood, "number"),
        ],
        figures=[
            encode.figure(
                plots.plot_regime_timeline(
                    mkt_w, dates_w, states_w, regime.labels,
                    title="Market with detected stress regimes",
                ),
                "regime_timeline", "Regime timeline",
                "Shaded bands are the periods the model classifies as stress.",
            ),
            encode.figure(
                plots.plot_regime_probability(regime_w),
                "regime_prob", "Probability of stress",
                "Reading the probability rather than the hard classification "
                "shows where the model is genuinely uncertain.",
            ),
        ],
        tables=tables, notes=notes,
    )


def build_optimization(ctx: Context) -> dict:
    cov = ctx.covariance
    tickers = ctx.rs.tickers
    n = len(tickers)

    mv = min_variance(cov)
    mv_ls = min_variance(cov, LONG_SHORT)
    # Risk parity and HRP answer different questions from the same Σ: minimum
    # variance minimises risk and concentrates, risk parity equalises the
    # contributions, HRP allocates down a correlation tree without inverting
    # anything. Showing them side by side is the point of the page.
    rp = risk_parity(cov)
    hrp_res = hrp(cov)

    ew = np.full(n, 1.0 / n)
    ew_vol = float(np.sqrt(ew @ cov.matrix @ ew))
    w_current = np.array([ctx.weights[t] for t in tickers])
    cur_vol = float(np.sqrt(w_current @ cov.matrix @ w_current))

    # Sample means are a poor forecast; they are used here only to give the
    # frontier a return axis, and the note below says so.
    mu = ctx.rs.to_numpy().mean(axis=0) * 252
    ef = efficient_frontier(cov, mu, n_points=ctx.spec.settings.frontier_points)

    rng = np.random.default_rng(0)
    W = rng.random((ctx.spec.settings.cloud_points, n))
    W /= W.sum(axis=1, keepdims=True)

    frontier_fig = plots.plot_efficient_frontier(
        ef,
        assets=(tickers, cov.volatilities, mu),
        markers=[
            ("equal weight", ew_vol, float(ew @ mu)),
            (ctx.spec.name, cur_vol, float(w_current @ mu)),
        ],
        cloud=(np.sqrt(np.einsum("ij,jk,ik->i", W, cov.matrix, W)), W @ mu),
        title="Efficient frontier (μ = historical mean — illustrative)",
    )

    weights_df = pl.DataFrame({
        "ticker": tickers,
        "current": [ctx.weights[t] for t in tickers],
        "equal_weight": [1.0 / n] * n,
        "min_variance": [mv.weights_dict()[t] for t in tickers],
        "risk_parity": [rp.weights_dict()[t] for t in tickers],
        "hrp": [hrp_res.weights_dict()[t] for t in tickers],
        "min_var_long_short": [mv_ls.weights_dict()[t] for t in tickers],
    })

    return encode.section(
        "optimization", "Portfolio construction",
        subtitle="Minimum variance and the efficient frontier",
        stats=[
            encode.stat("Current volatility", cur_vol, "percent",
                        f"{ctx.spec.name}, annualised"),
            encode.stat("Equal-weight volatility", ew_vol, "percent"),
            encode.stat("Min-variance volatility", mv.expected_volatility, "percent",
                        tone="good"),
            encode.stat("Reduction vs current",
                        1 - mv.expected_volatility / cur_vol if cur_vol else 0.0,
                        "percent", "Expected volatility saved"),
            encode.stat("Effective positions", mv.effective_n_positions, "number",
                        f"of {n} — inverse Herfindahl of the min-variance weights"),
            encode.stat("Risk parity volatility", rp.expected_volatility,
                        "percent", "Every asset contributes equal risk"),
            encode.stat("HRP volatility", hrp_res.expected_volatility, "percent",
                        "No matrix inversion anywhere"),
            encode.stat("Long-short gross exposure", mv_ls.gross_exposure, "number",
                        "Σ|w| — leverage the unconstrained solution takes on"),
        ],
        figures=[
            encode.figure(
                frontier_fig, "frontier", "Efficient frontier",
                "Expected returns are historical means, used here only to give "
                "the frontier a vertical axis — see the caveat below.",
            ),
            encode.figure(
                plots.plot_weights_comparison({
                    ctx.spec.name: ctx.weights,
                    "equal weight": {t: 1.0 / n for t in tickers},
                    "min variance": mv.weights_dict(),
                    "risk parity": rp.weights_dict(),
                    "HRP": hrp_res.weights_dict(),
                }, title="Allocation by strategy"),
                "weights_comparison", "Allocations side by side",
            ),
        ],
        tables=[encode.table(
            "weights", "Weights by strategy", weights_df,
            formats={
                "current": "percent", "equal_weight": "percent",
                "min_variance": "percent", "risk_parity": "percent",
                "hrp": "percent", "min_var_long_short": "percent",
            },
        )],
        notes=[
            "Minimum variance needs no expected returns, which is exactly what "
            "makes it robust: μ is the noisiest input in mean-variance and the "
            "one that drives error maximisation. The long-only constraint acts "
            "as an implicit regulariser (Jagannathan & Ma 2003).",
            "The frontier does need μ, and here it is the historical mean — a "
            "poor forecast, shown for construction purposes only. In practice μ "
            "would come from Black-Litterman or explicit views.",
            "Risk parity holds every asset by construction and minimum variance "
            "will drop assets outright, so the two differ most where the "
            "covariance is least trustworthy. HRP never inverts Σ at all, which "
            "is why it degrades gracefully when the estimate is poorly "
            "conditioned — see METHODOLOGY §12.",
            f"The long-short solution reaches {mv_ls.expected_volatility:.2%} "
            f"volatility but with {mv_ls.gross_exposure:.2f}× gross exposure: "
            f"large offsetting legs fitted to the noise in Σ.",
        ],
    )


# ──────────────────────────────────────────────────────────────────────────
# Orchestration
# ──────────────────────────────────────────────────────────────────────────

# Reading order, not dependency order. The narrative runs: what the book is,
# what environment it lives in, what its risk properties are, what breaks it,
# and only then what to do about it.
#
# Portfolio construction is deliberately last. It is the one page that
# prescribes rather than diagnoses, and its headline number — volatility saved
# versus the current allocation — is meaningless until the reader has seen
# where the current risk actually sits.
#
# Factors must precede stress: the historical replays apply the factor betas,
# and the stress page's caveat points back at the rolling-beta chart.
SECTIONS: list[tuple[str, str, Callable[[Context], dict]]] = [
    ("overview", "Overview", build_overview),
    ("factors", "Factor model", build_factors),
    ("regimes", "Market regimes", build_regimes),
    ("risk", "Risk decomposition", build_risk),
    ("tail", "Tail risk", build_tail),
    ("structure", "Latent structure", build_structure),
    ("stress", "Stress testing", build_stress),
    ("optimization", "Portfolio construction", build_optimization),
]


def build_payload(
    spec: PortfolioSpec,
    store: ParquetStore,
    quality: object | None = None,
    only: list[str] | None = None,
) -> dict:
    """Run every section and assemble the full dashboard payload."""
    import datetime as dt
    import time

    rs = store.read_returns(spec.tickers, start=spec.start, end=spec.end)
    missing = [t for t in spec.tickers if t not in rs.tickers]
    if missing:
        raise ValueError(
            f"No price data for {missing}. Run the build without --skip-ingest "
            f"to download them."
        )

    ctx = Context(spec=spec, store=store, rs=rs.select(spec.tickers))
    sections = []

    for sid, title, fn in SECTIONS:
        if only and sid not in only:
            continue
        t0 = time.perf_counter()
        try:
            sections.append(fn(ctx))
            logger.info(f"  section {sid:<13} {time.perf_counter() - t0:6.2f}s")
        except Exception as e:                       # noqa: BLE001
            logger.opt(exception=True).error(f"  section {sid:<13} FAILED: {e}")
            sections.append(encode.section(
                sid, title, error=f"{type(e).__name__}: {e}"
            ))

    meta = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "portfolio": {
            "name": spec.name,
            "description": spec.description,
            "positions": [
                {"ticker": p.ticker, "name": p.name,
                 "asset_class": p.asset_class, "weight": p.weight}
                for p in spec.positions
            ],
            "cov_method": spec.cov_method,
        },
        "coverage": {
            "start": ctx.dates[0],
            "end": ctx.dates[-1],
            "observations": ctx.rs.n_obs,
            "requested_start": spec.start,
            "requested_end": spec.end,
        },
        "warnings": ctx.warnings,
    }

    if quality is not None:
        meta["quality"] = {
            "tickers_checked": len(quality.tickers),
            "findings": len(quality.findings),
            "critical": quality.n_critical,
            "warning": quality.n_warning,
            "detail": encode.safe(quality.to_dataframe()),
        }

    return encode.safe({"meta": meta, "sections": sections})
