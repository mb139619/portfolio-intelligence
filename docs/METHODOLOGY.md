# Methodology & Models

This document states the models, formulas and assumptions behind each analytic.
It is meant to be read by someone who will trust the numbers — so every method
is described together with its limitations.

Throughout: returns are daily simple returns unless noted; annualisation uses
`T = 252` trading days; the default risk-free rate is configurable
(`settings.risk_free_rate`).

---

## 1. Returns

Simple returns `r_t = P_t / P_{t-1} - 1` are computed from **adjusted** close
prices (dividends and splits incorporated). Log returns are available via
`ReturnSeries.from_log_prices`. The portfolio return series is
`r_p = R w` for weight vector `w` — i.e. it assumes **daily rebalancing back to
the target weights**. This is the standard convention for risk analytics; it
slightly differs from a buy-and-hold drifting-weight portfolio.

---

## 2. Performance metrics

| Metric | Definition |
|--------|------------|
| Annualised return | `(∏(1+r))^(252/T) − 1` (geometric) |
| Annualised volatility | `std(r, ddof=1) · √252` |
| Sharpe | `(annual return − rf) / annual vol` |
| Sortino | `(annual return − rf) / downside deviation` |
| Calmar | `annual return / |max drawdown|` |
| Max drawdown | minimum of `cum/peak − 1` |
| Historical VaR (95%) | negative of the 5th percentile of returns |
| Historical CVaR | mean loss beyond VaR |

**Assumptions / limits.** VaR and CVaR are *historical* (empirical quantiles),
so they inherit the sample's tail behaviour and say nothing about losses larger
than anything observed. Ratios assume returns are roughly comparable across time
(no regime adjustment). Downside deviation uses the risk-free rate as the
minimum acceptable return.

---

## 3. Risk decomposition

For weights `w` and annualised covariance `Σ`:

- Portfolio volatility: `σ_p = √(wᵀ Σ w)`
- Marginal contribution to risk: `MCR = Σw / σ_p`
- Risk contribution: `RC_i = w_i · MCR_i` (these sum exactly to `σ_p`)
- Percent contribution: `%RC_i = RC_i / σ_p` (sums to 1)

**Covariance estimators.** `sample` (unbiased), `ledoit_wolf` (shrinkage toward
a structured target — more stable when observations are scarce relative to the
number of assets), `ewma` (RiskMetrics, λ = 0.94 — emphasises recent data).

**Assumptions / limits.** Risk contributions are a *local* (first-order)
decomposition: they describe sensitivity at the current weights and are exact
for variance but assume the covariance is a faithful description of risk
(elliptical-ish returns). They are not stable through regime shifts.

---

## 4. Factor engine

Time-series regression of portfolio **excess** returns on the Fama-French
factors:

```
r_excess(t) = α + Σ_k β_k · f_k(t) + ε(t)
```

where `r_excess = r_portfolio − RF` and the factors (`Mkt-RF`, `SMB`, `HML`,
`RMW`, `CMA`) are themselves excess/long-short returns from the French library.

- **Estimation:** OLS with an intercept.
- **Standard errors:** Newey-West **HAC** by default (`hac_lags=5`). Daily
  returns are autocorrelated and heteroskedastic; plain OLS standard errors
  understate uncertainty. HAC changes the *standard errors and t-stats only* —
  the coefficients are identical to OLS.
- **Outputs:** `α` (annualised), per-factor `β` with t-stats and p-values,
  `R²` / adjusted `R²`, and annualised idiosyncratic (residual) volatility.

### 4.1 Factor risk decomposition

With factor covariance `Σ_f`:

```
Var(r) = βᵀ Σ_f β   +   Var(ε)
         └ systematic ┘   └ specific ┘
```

Per-factor variance contribution is `β_i · (Σ_f β)_i`, which sums exactly to the
systematic variance.

**Important caveat.** Because the factors are *correlated* (notably `HML`/`CMA`),
individual per-factor contributions can be **negative** even though they sum to
the systematic total. This is mathematically correct, not a bug: a factor can
have negative covariance-weighted contribution. Interpret per-factor shares as a
covariance-aware attribution, not as independent buckets.

### 4.2 Return attribution

`contribution_i = β_i · mean(f_i)` annualised; the residual ≈ `α`. This is a
first-order attribution and does not perfectly reconcile with compounded total
return over long windows.

### 4.3 Data alignment caveat

The French library publishes with a lag, so factor data typically **ends weeks
before** your price data. `align_factors` performs an inner join on date and
logs a warning reporting how many recent observations are dropped. The factor
model is therefore fit on a slightly shorter, slightly older window than the raw
performance statistics — the two are not directly comparable.

---

## 5. PCA risk model

Eigendecomposition of the (daily) covariance or correlation matrix
`M = V Λ Vᵀ`.

- Eigenvalues are reported in daily-variance units; explained-variance ratios
  are unit-free.
- Eigenvectors (loadings) are sign-fixed so each PC's largest-magnitude loading
  is positive — making PC1 typically all-positive (a market/level direction)
  and keeping signs stable across refits.
- Scores (PC time series) satisfy `Var(score_i) = λ_i` for covariance PCA.

**Covariance vs correlation.** Covariance PCA preserves actual risk magnitudes
(right for risk work). Correlation PCA is scale-free (useful when assets have
very different volatilities and you care about co-movement, not magnitude).

**Assumptions / limits.** PCA factors are statistical, not economic — they have
no inherent interpretation beyond "directions of common variation", and their
identity can rotate between samples when eigenvalues are close.

---

## 6. Hidden Concentration Detector

Based on Meucci's *effective number of bets*. Decompose portfolio variance along
the principal components: with PC exposures `e = Vᵀ w`,

```
σ²_p = Σ_i λ_i e_i²   →   p_i = λ_i e_i² / σ²_p   (shares, sum to 1)
ENB  = exp( − Σ_i p_i ln p_i )
```

`ENB` ranges from 1 (all risk in one component — a hidden single bet) to `N`
(risk spread equally across all independent components). The detector contrasts
this with the naive weight-based diversification `1 / Σ w_i²` (inverse
Herfindahl). A large gap signals hidden concentration.

**Assumptions / limits.** The measure depends on the chosen basis (PCA). Meucci's
*minimum-torsion* basis can give more stable bets; PCA is used here for
transparency. ENB is a point-in-time, covariance-based diagnostic.

---

## 7. Correlation analytics

- **Estimators:** Pearson, Spearman (rank, robust to outliers), and an EWMA
  correlation (from the EWMA covariance, λ = 0.94) for a current-state view.
- **Distance:** Mantegna's metric `d_ij = √(2(1 − ρ_ij))`, a proper distance
  used for both clustering and the MST.
- **Clustering:** hierarchical (default `ward`; `single` linkage matches the MST
  topology and is the HRP choice). The leaf order gives a quasi-diagonal
  reordering that makes block structure visible.
- **Minimum Spanning Tree:** Mantegna's asset tree — the `N−1` strongest links
  connecting the universe. Node degree identifies systemic **hubs** vs
  peripheral **diversifiers**.

**Assumptions / limits.** Correlation is a linear, second-moment measure; it
misses non-linear dependence and tail co-movement. The MST hub/leaf reading is
most meaningful for larger universes.

---

## 8. Stress testing

### 8.1 Factor-based historical replay
Apply the portfolio's **current** factor betas to the factor returns of a real
crisis window. Because Fama-French factors reach back to 1926, this covers
crises that predate your instruments (e.g. 2008). It captures the **systematic**
PnL only — by construction it excludes idiosyncratic moves and alpha, so it
slightly understates the realised loss of a concentrated portfolio.

### 8.2 Asset-based historical replay
Apply current weights to the actual asset returns over the window. Requires the
assets to have existed; missing assets are dropped and weights renormalised
(with a warning). Per-asset contributions are approximate due to compounding.

### 8.3 Parametric factor shocks
`PnL = Σ_i β_i · shock_i` — exact and additive across factors.

### 8.4 Macro shocks (rates / oil / USD / VIX)
For variables outside the FF factor set, the portfolio's sensitivity to daily
moves is estimated by linear regression, then a shock is applied in native units
(e.g. +100bp = +0.01 on the rate series).

**Assumptions / limits.** Macro shocks are **first-order (delta)**:
`PnL = sensitivity · shock`. They ignore convexity and second-order effects —
for a bond-heavy portfolio under a large rate shock (e.g. +200bp), duration
captures the linear move but not the convexity correction. Sensitivities are
also estimated on historical co-movement, which may not hold in a genuine shock.

---

## 9. Data quality & general caveats

- **Free data is imperfect.** Yahoo data can contain bad ticks, gaps, and
  adjustment quirks; the ingester drops null/non-positive adjusted closes but
  does not perform deep cleaning.
- **Survivorship bias.** The universe is defined statically; delisted
  instruments are not included.
- **No transaction costs, taxes, or liquidity modelling.**
- **Point-in-time vs revised data.** FRED/ECB series may be revised; the
  platform stores the latest values, not the data as known at the time.
- **Not investment advice.** All outputs are for research and education.
