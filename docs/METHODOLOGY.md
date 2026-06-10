# Methodology & Models

This document states the models, formulas and assumptions behind each analytic.
It is meant to be read by someone who will trust the numbers — so every method
is described together with its limitations.

Throughout: returns are daily simple returns unless noted; annualisation uses
$T = 252$ trading days; the default risk-free rate is configurable
(`settings.risk_free_rate`).

> Equations render via LaTeX/MathJax on GitHub and in most Markdown previewers.

Each section carries two extras: a **📖 How to read it** block with the
interpretive nuance that matters in practice, and a **📚 Key references** box
pointing to the primary literature.

---

## 1. Returns

Simple returns are computed from **adjusted** close prices (dividends and splits
incorporated):

$$r_t = \frac{P_t}{P_{t-1}} - 1$$

Log returns are available via `ReturnSeries.from_log_prices`. The portfolio
return series is $r_p = R\,w$ for weight vector $w$ — i.e. it assumes **daily
rebalancing back to the target weights**. This is the standard convention for
risk analytics; it slightly differs from a buy-and-hold drifting-weight
portfolio.

---

## 2. Performance metrics

| Metric | Definition |
|--------|------------|
| Annualised return | $\left(\prod_t (1+r_t)\right)^{252/T} - 1$ (geometric) |
| Annualised volatility | $\mathrm{std}(r)\cdot\sqrt{252}$ |
| Sharpe | $\dfrac{\overline{r-r_f}\,\cdot 252}{\mathrm{std}(r-r_f)\,\sqrt{252}} = \dfrac{\overline{r-r_f}}{\mathrm{std}(r-r_f)}\sqrt{252}$ (arithmetic excess) |
| Sortino | $\dfrac{\overline{r-r_f}\cdot 252}{\text{downside deviation}}$ |
| Calmar | $\dfrac{\text{annual return}}{\lvert \text{max drawdown}\rvert}$ |
| Max drawdown | $\min_t\left(\dfrac{C_t}{\max_{s\le t} C_s} - 1\right)$, with $C_t=\prod_{s\le t}(1+r_s)$ |
| Historical VaR (95%) | negative of the 5th percentile of returns |
| Historical CVaR | mean loss beyond VaR |

Downside deviation, with minimum acceptable return $r_f$ (daily):

$$\text{DD} = \sqrt{\frac{1}{T}\sum_t \min(r_t - r_f,\, 0)^2}\,\cdot\sqrt{252}$$

**Assumptions / limits.** VaR and CVaR are *historical* (empirical quantiles),
so they inherit the sample's tail behaviour and say nothing about losses larger
than anything observed. Ratios assume returns are roughly comparable across time
(no regime adjustment). Downside deviation uses the risk-free rate as the
minimum acceptable return.

> **📖 How to read it.** The Sharpe ratio is a *signal-to-noise* measure, not a
> return measure: it answers "how much excess return per unit of variability",
> and it is only comparable across strategies at the same frequency (a daily
> Sharpe annualised by $\sqrt{252}$ assumes i.i.d. returns — serial correlation
> inflates it, which is why hedge funds with smoothed returns can show
> implausibly high Sharpes). Sortino replaces total volatility with downside
> deviation, so it rewards strategies whose volatility is mostly upside; the two
> diverge most for skewed return streams. Calmar (return over max drawdown) is a
> path-dependent cousin that institutional allocators watch because investors
> redeem on drawdowns, not on variance.

> **📚 Key references**
> - Sharpe, W. (1966), *Mutual Fund Performance*, Journal of Business.
> - Sharpe, W. (1994), *The Sharpe Ratio*, Journal of Portfolio Management.
> - Sortino & Price (1994), *Performance Measurement in a Downside Risk Framework*, J. of Investing.
> - Lo, A. (2002), *The Statistics of Sharpe Ratios*, Financial Analysts Journal (the $\sqrt{T}$ / autocorrelation caveat).

**Risk-free convention.** Sharpe and Sortino use the *arithmetic* mean of daily **excess** returns $r_t - r_{f,t}$, annualised — the textbook definition. The risk-free can be a constant annual rate or, preferably, the **actual daily risk-free series** (the French `RF` factor, or FRED Fed Funds): over a sample where rates move from ~0% to ~5%, a constant rate materially distorts the ratio. The geometric annualised return (CAGR) is reported separately as a performance descriptor and is *not* used in the Sharpe numerator.

---

## 3. Risk decomposition

For weights $w$ and annualised covariance $\Sigma$:

$$\sigma_p = \sqrt{w^\top \Sigma\, w}
\qquad
\text{MCR} = \frac{\Sigma w}{\sigma_p}
\qquad
\text{RC}_i = w_i \cdot \text{MCR}_i
\qquad
\%\text{RC}_i = \frac{\text{RC}_i}{\sigma_p}$$

The risk contributions $\text{RC}_i$ sum exactly to $\sigma_p$, and the percent
contributions $\%\text{RC}_i$ sum to 1.

**Covariance estimators.** `sample` (unbiased), `ledoit_wolf` (shrinkage toward
a structured target — more stable when observations are scarce relative to the
number of assets), `ewma` (RiskMetrics, $\lambda = 0.94$ — emphasises recent
data, **zero-mean convention**: returns are not demeaned, consistent with the
RiskMetrics specification and free of any sequential look-ahead from a
full-sample mean).

**Assumptions / limits.** Risk contributions are a *local* (first-order)
decomposition: they describe sensitivity at the current weights and are exact
for variance but assume the covariance is a faithful description of risk
(elliptical-ish returns). They are not stable through regime shifts.

> **📖 How to read it.** Percent risk contribution answers a question weights
> cannot: *which positions actually drive portfolio volatility?* An asset can
> carry a small weight but a large risk share if it is volatile and correlated
> with the rest (and vice versa). The classic example is a 60/40 portfolio,
> where equities are ~40% of capital but routinely ~90% of risk — the seed of
> risk parity. Marginal contribution to risk (MCR) is the *sensitivity*: how
> portfolio volatility moves if you add a unit of that asset, and it is exactly
> what you set equal across assets to build an equal-risk-contribution (ERC)
> portfolio. Shrinkage (Ledoit-Wolf) matters here because risk contributions
> inherit the instability of the sample covariance: with $N$ assets you estimate
> $N(N+1)/2$ parameters, and the smallest eigenvalues — which dominate the
> inverse used in optimisation — are the noisiest.

> **📚 Key references**
> - Litterman, R. (1996), *Hot Spots and Hedges*, Goldman Sachs (marginal/contribution-to-risk framework).
> - Ledoit & Wolf (2004), *A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices*, J. of Multivariate Analysis.
> - Maillard, Roncalli & Teïletche (2010), *The Properties of Equally Weighted Risk Contribution Portfolios*, JPM.
> - RiskMetrics Technical Document (1996), J.P. Morgan (EWMA, $\lambda=0.94$).

---

## 4. Factor engine

Time-series regression of portfolio **excess** returns on the Fama-French
factors:

$$r_{\text{excess}}(t) = \alpha + \sum_{k} \beta_k\, f_k(t) + \varepsilon(t)$$

where $r_{\text{excess}} = r_{\text{portfolio}} - \text{RF}$ and the factors
($\text{Mkt-RF}$, $\text{SMB}$, $\text{HML}$, $\text{RMW}$, $\text{CMA}$) are
themselves excess / long-short returns from the French library.

- **Estimation:** OLS with an intercept.
- **Standard errors:** Newey-West **HAC** by default (`hac_lags=5`). Daily
  returns are autocorrelated and heteroskedastic; plain OLS standard errors
  understate uncertainty. HAC changes the *standard errors and t-stats only* —
  the coefficients are identical to OLS.
- **Outputs:** $\alpha$ (annualised), per-factor $\beta$ with t-stats and
  p-values, $R^2$ / adjusted $R^2$, and annualised idiosyncratic (residual)
  volatility. **Alpha is annualised arithmetically** ($\alpha \times 252$): it is a regression intercept, not a compounding return.

### 4.1 Factor risk decomposition

With factor covariance $\Sigma_f$, total variance splits cleanly:

$$\mathrm{Var}(r) = \underbrace{\beta^\top \Sigma_f\, \beta}_{\text{systematic}} + \underbrace{\mathrm{Var}(\varepsilon)}_{\text{specific}}$$

Per-factor variance contribution is $\beta_i \,(\Sigma_f\, \beta)_i$, which sums
exactly to the systematic variance.

**Important caveat.** Because the factors are *correlated* (notably
$\text{HML}$ / $\text{CMA}$), individual per-factor contributions can be
**negative** even though they sum to the systematic total. This is
mathematically correct, not a bug: a factor can have negative covariance-weighted
contribution. Interpret per-factor shares as a covariance-aware attribution, not
as independent buckets.

### 4.2 Return attribution

$$\text{contribution}_i = \beta_i \cdot \overline{f_i}\cdot 252,
\qquad
\overline{r_{\text{excess}}}\cdot 252 = \underbrace{\alpha\cdot 252}_{\text{alpha}} + \sum_k \beta_k\,\overline{f_k}\cdot 252$$

Because the OLS residual has zero mean, this **reconciles exactly**: alpha plus
the factor contributions equal the total annualised excess return to machine
precision. (Using arithmetic, not geometric, annualisation is what preserves the
identity.)

### 4.3 Data alignment caveat

The French library publishes with a lag, so factor data typically **ends weeks
before** your price data. `align_factors` performs an inner join on date and
logs a warning reporting how many recent observations are dropped. The factor
model is therefore fit on a slightly shorter, slightly older window than the raw
performance statistics — the two are not directly comparable.

> **📖 How to read it.** The betas are the portfolio's *style fingerprint*. A
> market beta near 1 with near-zero SMB/HML is a closet index fund; a positive
> HML is a value tilt, negative is growth; positive SMB is a small-cap tilt. The
> $R^2$ tells you how much of the return variation is *style* versus security
> selection — a diversified equity fund often sits at 0.90+, meaning almost
> everything is factor beta and very little is idiosyncratic skill. **Alpha is
> the residual that the known factors cannot explain**; its t-stat is the
> honest question — most apparent alpha is statistically indistinguishable from
> zero once you account for daily autocorrelation (hence HAC errors). On the
> risk side, the systematic/specific split is the single most decision-relevant
> number: specific risk is diversifiable (add more names), systematic risk is
> not (you must hedge the factor). On the **negative per-factor contributions**:
> they are a feature of a *correlated* factor set — a long-value, short-growth
> tilt can have HML and CMA contributions of opposite sign that net to the true
> systematic variance. Read them as covariance-aware, not as standalone buckets;
> if you need orthogonal buckets, rotate the factors first.

> **📚 Key references**
> - Fama & French (1993), *Common Risk Factors in the Returns on Stocks and Bonds*, JFE (3-factor).
> - Fama & French (2015), *A Five-Factor Asset Pricing Model*, JFE (adds RMW, CMA).
> - Carhart (1997), *On Persistence in Mutual Fund Performance*, J. of Finance (momentum).
> - Newey & West (1987), *A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix*, Econometrica.
> - Sharpe, W. (1992), *Asset Allocation: Management Style and Performance Measurement*, JPM (returns-based style analysis).

---

## 5. PCA risk model

Eigendecomposition of the (daily) covariance or correlation matrix:

$$M = V \Lambda V^\top$$

- Eigenvalues are reported in daily-variance units; explained-variance ratios
  are unit-free.
- Eigenvectors (loadings) are sign-fixed so each PC's largest-magnitude loading
  is positive — making PC1 typically all-positive (a market/level direction)
  and keeping signs stable across refits.
- Scores (PC time series) satisfy $\mathrm{Var}(\text{score}_i) = \lambda_i$
  for covariance PCA.

**Covariance vs correlation.** Covariance PCA preserves actual risk magnitudes
(right for risk work). Correlation PCA is scale-free (useful when assets have
very different volatilities and you care about co-movement, not magnitude).

**Assumptions / limits.** PCA factors are statistical, not economic — they have
no inherent interpretation beyond "directions of common variation", and their
identity can rotate between samples when eigenvalues are close.

> **📖 How to read it.** In almost every cross-asset universe the **first PC is
> the market/level factor** — it has all-positive loadings and explains the lion's
> share of variance (often 70–90% for equities). Subsequent PCs are *spreads*:
> PC2 frequently separates duration/defensives from cyclicals, PC3 a regional or
> sector axis, and so on. The **scree plot** answers "how many independent bets
> does this universe really contain?" — if three PCs explain 95%, a 30-name
> portfolio has roughly three degrees of freedom. **Eigen-portfolios** (the
> eigenvectors read as weights) are mutually uncorrelated by construction, which
> is why PCA underlies statistical-arbitrage and the latent-factor risk models of
> Barra/Axioma. The interpretive trap is *eigenvalue crowding*: when two
> eigenvalues are close their eigenvectors are nearly unidentified and will swap
> or rotate between samples, so never over-interpret a single mid-spectrum PC.

> **📚 Key references**
> - Connor & Korajczyk (1986), *Performance Measurement with the Arbitrage Pricing Theory*, JFE (PCA / approximate factor models).
> - Litterman & Scheinkman (1991), *Common Factors Affecting Bond Returns*, J. of Fixed Income (level/slope/curvature PCs).
> - Laloux, Cizeau, Bouchaud & Potters (1999), *Noise Dressing of Financial Correlation Matrices*, PRL (random-matrix view of which PCs are signal).
> - Avellaneda & Lee (2010), *Statistical Arbitrage in the US Equities Market*, Quantitative Finance (eigen-portfolios).

---

## 6. Hidden Concentration Detector

Based on Meucci's *effective number of bets*. Decompose portfolio variance along
the principal components: with PC exposures $e = V^\top w$,

$$\sigma_p^2 = \sum_i \lambda_i\, e_i^2
\qquad
p_i = \frac{\lambda_i\, e_i^2}{\sigma_p^2}
\quad\Big(\textstyle\sum_i p_i = 1\Big)$$

$$\text{ENB} = \exp\!\left(-\sum_i p_i \ln p_i\right)$$

$\text{ENB}$ ranges from 1 (all risk in one component — a hidden single bet) to
$N$ (risk spread equally across all independent components). The detector
contrasts this with the naive weight-based diversification (inverse Herfindahl):

$$\text{ENB}_{\text{weights}} = \frac{1}{\sum_i w_i^2}$$

A large gap between the two signals hidden concentration.

**Assumptions / limits.** The measure depends on the chosen basis (PCA).
Meucci's *minimum-torsion* basis can give more stable bets; PCA is used here for
transparency. ENB is a point-in-time, covariance-based diagnostic.

> **📖 How to read it.** The number of holdings and even the weight-based
> diversification (inverse Herfindahl) can lie: ten equally-weighted tech names
> look like ten bets but are effectively one. The **effective number of bets**
> measures diversification in *risk* space rather than *capital* space. Read the
> gap, not the level: $\mathrm{ENB}_{\text{weights}} = 10$ with
> $\mathrm{ENB}_{\text{risk}} = 1.5$ is the signature of hidden concentration —
> the portfolio is one macro bet wearing the costume of diversification. The
> entropy formulation rewards *spreading risk evenly across uncorrelated
> directions*, which is exactly what a genuinely diversified book does. The
> caveat worth stating aloud in an interview: ENB depends on the basis you
> decompose in (here PCA); Meucci's minimum-torsion basis stays closest to the
> original assets and tends to give more stable, more interpretable bets.

> **📚 Key references**
> - Meucci, A. (2009), *Managing Diversification*, Risk (the effective-number-of-bets / entropy framework).
> - Meucci, Santangelo & Deguest (2015), *Risk Budgeting and Diversification Based on Optimal Risk Factors* (minimum-torsion bets).
> - Choueifaty & Coignard (2008), *Toward Maximum Diversification*, JPM (the diversification ratio, a related lens).

---

## 7. Correlation analytics

- **Estimators:** Pearson, Spearman (rank, robust to outliers), and an EWMA
  correlation (from the EWMA covariance, $\lambda = 0.94$) for a current-state
  view.
- **Distance:** Mantegna's metric, a proper distance used for both clustering
  and the MST:

$$d_{ij} = \sqrt{2\,(1 - \rho_{ij})}$$

  with $\rho=1 \Rightarrow d=0$, $\rho=0 \Rightarrow d=\sqrt{2}$,
  $\rho=-1 \Rightarrow d=2$.
- **Clustering:** hierarchical (default `ward`; `single` linkage matches the MST
  topology and is the HRP choice). The leaf order gives a quasi-diagonal
  reordering that makes block structure visible.
- **Minimum Spanning Tree:** Mantegna's asset tree — the $N-1$ strongest links
  connecting the universe. Node degree identifies systemic **hubs** vs
  peripheral **diversifiers**.

**Assumptions / limits.** Correlation is a linear, second-moment measure; it
misses non-linear dependence and tail co-movement. The MST hub/leaf reading is
most meaningful for larger universes.

> **📖 How to read it.** Average pairwise correlation is a **systemic-stress
> gauge**: in calm regimes assets disperse (low average correlation, real
> diversification), but in crises everything moves together and the average
> spikes toward 1 — diversification evaporates exactly when you need it. The
> **clustered heatmap** turns a noisy matrix into visible blocks (equities,
> rates, commodities); the quasi-diagonal order is the same seriation that
> Hierarchical Risk Parity uses to avoid inverting an unstable covariance. The
> **Minimum Spanning Tree** is the cleanest one-picture summary of the risk
> network: it keeps only the strongest link per node, so *hubs* (high degree)
> are the systemic assets the rest of the book hangs off — shock the hub and the
> whole tree moves — while *leaves* are the genuine diversifiers. Empirically the
> tree contracts (shorter total length, more star-like around a single hub)
> during crises, which is itself a regime signal.

> **📚 Key references**
> - Mantegna, R. (1999), *Hierarchical Structure in Financial Markets*, European Physical Journal B (the asset tree / MST).
> - Onnela, Chakraborti, Kaski, Kertész & Kanto (2003), *Dynamic Asset Trees and Black Monday*, Physica A (tree length as a crisis indicator).
> - Tumminello, Aste, Di Matteo & Mantegna (2005), *A Tool for Filtering Information in Complex Systems*, PNAS (PMFG, the MST's richer cousin).
> - López de Prado (2016), *Building Diversified Portfolios that Outperform Out of Sample*, JPM (clustering / quasi-diagonalisation → HRP).

---

## 8. Stress testing

### 8.1 Factor-based historical replay
Apply the portfolio's **current** factor betas to the factor returns of a real
crisis window. Because Fama-French factors reach back to 1926, this covers
crises that predate your instruments (e.g. 2008). It captures the **systematic**
PnL only — by construction it excludes idiosyncratic moves and alpha. Moreover
the betas are **full-sample and assumed constant**, whereas in real crises betas
and correlations typically rise (correlation breakdown). Both effects mean the
figure most likely **understates** the true loss; treat it as a lower bound.

### 8.2 Asset-based historical replay
Apply current weights to the actual asset returns over the window. Requires the
assets to have existed; missing assets are dropped and weights renormalised
(with a warning). Per-asset contributions are approximate due to compounding.

### 8.3 Parametric factor shocks
Exact and additive across factors:

$$\text{PnL} = \sum_i \beta_i \cdot \text{shock}_i$$

### 8.4 Macro shocks (rates / oil / USD / VIX)
For variables outside the FF factor set, the portfolio's sensitivity to daily
moves is estimated by linear regression, then a shock is applied in native units
(e.g. +100bp $= +0.01$ on the rate series).

**Assumptions / limits.** Macro shocks are **first-order (delta)**:

$$\text{PnL} = \sum_v \text{sensitivity}_v \cdot \text{shock}_v$$

They ignore convexity and second-order effects — for a bond-heavy portfolio
under a large rate shock (e.g. +200bp), duration captures the linear move but
not the convexity correction. Sensitivities are also estimated on historical
co-movement, which may not hold in a genuine shock.

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

---

## 10. General references & further reading

Textbooks that cover the whole pipeline and are the standard desk references:

- **Grinold & Kahn (2000), *Active Portfolio Management*** — the canonical text on factor models, the fundamental law of active management, and risk attribution.
- **Meucci, A. (2005), *Risk and Asset Allocation*, Springer** — rigorous treatment of estimation, risk decomposition, and diversification.
- **McNeil, Frey & Embrechts (2015), *Quantitative Risk Management*, Princeton** — VaR/ES, EVT, copulas, coherent risk measures.
- **Ang, A. (2014), *Asset Management: A Systematic Approach to Factor Investing*, Oxford** — the modern factor-investing perspective.
- **López de Prado (2018), *Advances in Financial Machine Learning*, Wiley** — HRP, backtesting pitfalls, and the look-ahead / in-sample discipline.

> **A note on scope.** Everything in this platform is **descriptive and
> in-sample**: it characterises the risk of a portfolio given a history. None of
> it is a forecasting model or a backtested strategy. Turning any of these
> diagnostics into a tradeable signal requires re-deriving them point-in-time
> (expanding/rolling windows that use only past data) to avoid look-ahead — a
> deliberate boundary, not an oversight.
