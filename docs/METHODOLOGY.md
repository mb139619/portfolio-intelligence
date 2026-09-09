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

## 4. Tail risk

Historical VaR and CVaR (Section 2) only see losses that already happened and say
nothing about anything larger. Three complementary estimators model the tail more
honestly, each a different bet on what the tail looks like:

| Estimator | Definition |
|-----------|------------|
| Gaussian VaR | $-(\mu + \sigma\, z_\alpha)$, with $z_\alpha$ the standard-normal quantile |
| Cornish-Fisher (modified) VaR | $-(\mu + \sigma\, z_{CF})$, with $z_{CF}$ the moment-adjusted quantile |
| Historical VaR / CVaR | empirical quantile and mean loss beyond it (Section 2) |
| EVT (peaks-over-threshold) VaR / CVaR | a GPD fitted to losses beyond a high threshold |

**Cornish-Fisher.** Expand the Gaussian quantile $z$ to correct for the sample's
skewness $S$ and excess kurtosis $K$:

$$z_{CF} = z + \tfrac{1}{6}(z^2-1)\,S + \tfrac{1}{24}(z^3-3z)\,K - \tfrac{1}{36}(2z^3-5z)\,S^2$$

The modified VaR is then $-(\mu + \sigma\, z_{CF})$, which collapses to the Gaussian
VaR when $S=K=0$. A cheap, distribution-free correction that uses only the first
four moments.

**Extreme Value Theory — peaks over threshold.** Fix a high threshold $u$ (here the
95th percentile of losses) and model the *exceedances* $L-u \mid L>u$ with a
Generalised Pareto Distribution. The Pickands–Balkema–de Haan theorem makes the GPD
the limiting law of exceedances for a broad class of loss distributions — which is
what licenses extrapolating *beyond* the worst observed loss. With $n$ total
observations, $N_u$ exceedances, shape $\xi$ and scale $\beta$:

$$\mathrm{VaR}_p = u + \frac{\beta}{\xi}\left[\left(\frac{n}{N_u}(1-p)\right)^{-\xi} - 1\right], \qquad
\mathrm{CVaR}_p = \frac{\mathrm{VaR}_p + \beta - \xi u}{1-\xi}\quad(\xi<1)$$

The shape $\xi$ is the tail index: $\xi>0$ is heavy-tailed (power-law decay),
$\xi=0$ the exponential (light-tailed) limit, $\xi<0$ a finite upper bound.
Expected shortfall is finite only for $\xi<1$.

**Assumptions / limits.** Gaussian VaR ignores skew and fat tails and
systematically *understates* downside risk for real return series — it is kept only
as the baseline the others improve on. Cornish-Fisher is a fourth-moment expansion:
reliable for mild non-normality, but the adjusted quantile can turn non-monotone at
extreme confidence levels and genuine far-tail behaviour exceeds what four moments
capture. EVT is the principled tail model but is sensitive to the threshold choice
(a bias/variance trade-off: too low and the non-tail bulk contaminates the fit, too
high and too few exceedances inflate variance — the fit needs a minimum number of
exceedances), and the standard POT estimator assumes i.i.d. exceedances, which
volatility clustering violates (the textbook refinement pre-filters returns through
a volatility model first; not done here). All three are single-period (1-day) and
computed in-sample.

> **📖 How to read it.** Lining the four up at the same confidence level makes the
> cost of the normal assumption explicit: for equity-like returns the historical
> and EVT VaR sit well below (more negative than) the Gaussian one, and that gap is
> the fat-tail premium a Gaussian model would have you ignore. Prefer **CVaR
> (expected shortfall)** as the headline number: unlike VaR it is *coherent*
> (sub-additive, so it never penalises diversification), which is why bank
> regulation (FRTB) moved from VaR to ES. The GPD shape $\xi$ is the single most
> portable summary of tail heaviness — independent of sample size — and for daily
> equity losses it typically lands around $0.2$–$0.3$ (clearly heavy-tailed).

> **📚 Key references**
> - McNeil, Frey & Embrechts (2015), *Quantitative Risk Management*, Princeton — the standard treatment of POT/EVT and coherent risk measures.
> - Embrechts, Klüppelberg & Mikosch (1997), *Modelling Extremal Events*, Springer — the EVT reference.
> - Artzner, Delbaen, Eber & Heath (1999), *Coherent Measures of Risk*, Mathematical Finance — why ES is preferred to VaR.
> - Fisher & Cornish (1960), *The Percentile Points of Distributions Having Known Cumulants*, Technometrics — the quantile expansion.

---

## 5. Factor engine

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

### 5.1 Factor risk decomposition

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

### 5.2 Return attribution

$$\text{contribution}_i = \beta_i \cdot \overline{f_i}\cdot 252,
\qquad
\overline{r_{\text{excess}}}\cdot 252 = \underbrace{\alpha\cdot 252}_{\text{alpha}} + \sum_k \beta_k\,\overline{f_k}\cdot 252$$

Because the OLS residual has zero mean, this **reconciles exactly**: alpha plus
the factor contributions equal the total annualised excess return to machine
precision. (Using arithmetic, not geometric, annualisation is what preserves the
identity.)

### 5.3 Data alignment caveat

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

## 6. PCA risk model

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

## 7. Hidden Concentration Detector

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

## 8. Correlation analytics

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

## 9. Stress testing

### 9.1 Factor-based historical replay
Apply the portfolio's **current** factor betas to the factor returns of a real
crisis window. Because Fama-French factors reach back to 1926, this covers
crises that predate your instruments (e.g. 2008). It captures the **systematic**
PnL only — by construction it excludes idiosyncratic moves and alpha. Moreover
the betas are **full-sample and assumed constant**, whereas in real crises betas
and correlations typically rise (correlation breakdown). Both effects mean the
figure most likely **understates** the true loss; treat it as a lower bound.

### 9.2 Asset-based historical replay
Apply current weights to the actual asset returns over the window. Requires the
assets to have existed; missing assets are dropped and weights renormalised
(with a warning). Per-asset contributions are approximate due to compounding.

### 9.3 Parametric factor shocks
Exact and additive across factors:

$$\text{PnL} = \sum_i \beta_i \cdot \text{shock}_i$$

### 9.4 Macro shocks (rates / oil / USD / VIX)
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

## 10. Regime detection

Markets alternate between hidden states (calm vs stress) with different mean and
variance. Two detectors:

**Primary — Gaussian hidden Markov / Markov-switching model.** Each state has its
own mean and variance, and the state follows a hidden Markov chain:

$$r_t = \mu_{S_t} + e_t, \qquad e_t \sim \mathcal{N}(0, \sigma^2_{S_t}), \qquad
S_t \in \{1,\dots,K\}\ \text{Markov}$$

Fit by EM (Hamilton filter + Kim smoother) with multiple random restarts to
avoid local optima. Outputs: smoothed state probabilities, per-state mean and
volatility, the **row-stochastic** transition matrix
$P_{ij} = \Pr(S_{t+1}=j \mid S_t=i)$, and expected state durations
$1/(1-P_{ii})$. States are canonicalised by ascending volatility so state 0 is
always "calm" and the last state "stress".

**Baseline — volatility states.** Classify each day by trailing realised
volatility into quantile buckets. No latent model; fast and transparent. It is
the benchmark the HMM must beat to justify its complexity.

**Where it runs.** Regimes are estimated on the **broad market** (the Mkt-RF
factor), not on the portfolio: a regime is a property of the market environment.
The portfolio then *inherits* the labels, and analytics are recomputed per
regime (`conditional.py`): per-regime return/volatility, market beta, and average
correlation.

> **📖 How to read it.** The decisive output is the *contrast across regimes*. A
> well-behaved result shows the stress regime with much higher volatility,
> negative drift, short expected duration (crises are brief but violent), and low
> unconditional frequency. The **regime-conditional statistics are the punchline**:
> volatility roughly doubles in the stress state and the return/vol ratio flips
> negative — robustly, across portfolios. Market beta and average correlation
> *often* rise too (diversification weakening when it is most needed, the
> empirical measurement behind the stress-test caveat), but this is conditional,
> not automatic: beta is a ratio whose numerator and denominator both inflate in
> stress, and an already-concentrated book starts highly correlated. Reading
> whether the effect is present — rather than assuming it — is the skill.
> The transition matrix's diagonal tells you regime *persistence* (markets stay
> calm a long time, then cluster volatility); the expected duration is
> $1/(1-P_{ii})$. Caveat: HMM states are a statistical convenience, not ground
> truth — two states is the robust default, three can be informative, but more
> states overfit fast with daily data, and the model is fit in-sample (the
> smoothed probabilities use the whole series; for a point-in-time signal use the
> *filtered* probabilities instead).

### 10.1 Smoothed vs filtered probabilities

The model reports both, and the choice is the difference between a description
and a look-ahead bug.

$$
\text{smoothed}_s = P(S_s \mid r_1,\dots,r_T), \qquad
\text{filtered}_s = P(S_s \mid r_1,\dots,r_s)
$$

The Kim smoother runs backwards, so a smoothed probability at time $s$
incorporates every observation after $s$. **For describing history that is the
correct estimate** — asked when the crisis was, one should use all the evidence
— and it is what the dashboard, `regime_conditional_*` and the reported state
frequencies use.

As a trading signal it is invalid: the model has already seen the crash it is
supposed to anticipate. Anything under `backtest/` must use `filtered_probs` /
`filtered_states`, and `RegimeModel.probabilities_at()` therefore defaults to
the filtered estimate — the safe default is the one that cannot leak.

**Measured on the Mkt-RF series (9,212 daily observations, 2-state fit):**

| | |
|---|---|
| Maximum divergence between the two probabilities | 0.780 |
| Days whose hard classification differs | 793 (8.6%) |
| Stress frequency, smoothed | 29.3% |
| Stress frequency, filtered | 28.8% |

The last two rows are the reason this is worth stating explicitly. The
*aggregate* is nearly identical — half a percentage point apart — so every
summary statistic looks unchanged. The disagreement is concentrated at the
regime transitions, which is exactly where a regime-conditional strategy would
act. A backtest built on smoothed states would appear sound in every headline
number while being wrong at every turning point.

By construction the two agree exactly on the final observation, where there is
no future left to smooth over. That identity is used as a test.

### 10.2 Publication lag

Truncating data at the decision date is necessary and not sufficient. A value
*dated* before $t$ is not necessarily *knowable* at $t$.

The Kenneth French library updates in batches roughly monthly. Measured against
this repository's own store the factor data has run **35 days behind the price
data and 40 days behind the current date**. A backtest estimating factor betas
at $t$ from data dated $t$ would therefore be using numbers nobody had.

Each source declares a lag: 0 days for daily closes (a close is published at the
close), 1 day for FRED and ECB daily series, and a deliberately conservative 45
days for the French factors. The true factor lag cannot be recovered from a
single snapshot — a snapshot shows current staleness, not when each observation
first appeared — so it is a stated assumption, recorded in `RunMeta` and
surfaced in the tearsheet rather than buried in code. Erring long only costs
history; erring short silently readmits look-ahead.

This is distinct from **execution lag**, which belongs to the execution layer:
knowing today's close does not imply one could have traded at it. Conflating the
two is a common way to smuggle in a bar of hindsight.

> **📚 Key references**
> - Hamilton, J. (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*, Econometrica (the regime-switching model).
> - Kim, C-J. (1994), *Dynamic Linear Models with Markov-Switching*, J. of Econometrics (the smoother).
> - Ang & Bekaert (2002), *International Asset Allocation with Regime Shifts*, Review of Financial Studies.
> - Guidolin & Timmermann (2007), *Asset Allocation under Multivariate Regime Switching*, J. of Economic Dynamics & Control.

---

## 11. Data quality & general caveats

- **Free data is imperfect.** Yahoo data can contain bad ticks, gaps, and
  adjustment quirks. Beyond the ingester dropping null/non-positive adjusted
  closes, a dedicated **data-quality layer** (`src/data_quality`) runs after
  ingestion and flags — without auto-correcting — calendar gaps, return
  outliers (an unadjusted split shows up as a ~50% daily move), stale price
  runs (suspended/illiquid names that deflate volatility), short histories,
  and null/non-positive prices, each with a severity. Outlier severity uses
  **per-asset-class bands** (a 10% day is routine for crypto, a red flag for a
  government-bond ETF) combined with a robust median±k·MAD band, and each
  finding lists the **dates** of the flagged moves so they can be investigated. The philosophy is to
  surface where to look and let a human decide, exactly as a risk desk would.
- **Survivorship bias.** The universe is defined statically; delisted
  instruments are not included.
- **No transaction costs, taxes, or liquidity modelling.**
- **Point-in-time vs revised data.** FRED/ECB series may be revised; the
  platform stores the latest values, not the data as known at the time.
- **Not investment advice.** All outputs are for research and education.

---

## 12. Calendars & multi-asset-class data

Equities trade roughly 252 days a year; a crypto pair trades all 365. Once both
are in the same store, the calendar can no longer be an assumption baked into
code — it becomes an attribute of the data, carried on the domain object and
recorded alongside the prices. Every annualisation factor derives from it.

### 12.1 Never forward-fill one calendar onto another

The tempting way to align a 24/7 series with an exchange-traded one is to carry
Friday's equity close through the weekend. This platform never does, and the
reason is worth stating precisely, because the resulting output looks perfectly
plausible.

Filling weekends manufactures two artificial zero returns per week — about 28%
of the sample. Those zeros:

- **deflate volatility by roughly 15%.** They enter the variance denominator
  while contributing nothing to the numerator.
- **compress correlations toward zero.** The fabricated zeros are shared across
  every equity, diluting genuine co-movement.
- **corrupt regime detection.** A Markov-switching model fitted on such a series
  learns a spurious low-volatility state that is simply "the weekend", and
  reports it as a market regime with an expected duration of two days.

The alternative is intersection: keep only dates on which every asset genuinely
traded. `ParquetStore.read_returns` does this by construction (an outer join
followed by `drop_nulls`), and logs how many observations it dropped. Reading
BTC alongside SPY and TLT over 2018-2026 intersects 3,170 crypto days down to
2,151 common dates — the 1,019 discarded rows are weekends and market holidays,
and none of them is replaced by a fabricated price.

The combined series is then annualised on the **most restrictive** calendar
present. Annualising an intersected series at 365 would overstate volatility by
√(365/252) ≈ 20%.

### 12.2 Each analytic declares a cross-calendar policy

| Policy | Meaning | Used by |
|---|---|---|
| **Native** | Runs on the asset's own frequency | tail risk, regime detection, realised volatility |
| **Intersection** | Only dates where every asset traded | correlation, MST, covariance, factor betas |
| **Unsupported** | Refuses with an explicit error | the Fama-French factor engine on crypto |

The distinction between Native and Intersection is not cosmetic. BTC's realised
volatility computed on the intersection with equities is not BTC's volatility —
it is the volatility of BTC *observed on weekdays*, a different and less useful
quantity that discards 28% of the sample.

**Unsupported** is a deliberate refusal rather than a warning. The Fama-French
factors are constructed from US equity portfolios sorted on size, value,
profitability and investment. A beta of BTC on HML has no economic referent, and
a clear exception is better than a coefficient someone later pastes into a
report. `align_factors` raises `UnsupportedCalendarError` on a continuous
calendar.

### 12.3 Bar close misalignment

Binance daily candles close at **00:00 UTC**; US equities close at 16:00 ET
(21:00 UTC in winter, 20:00 in summer). A daily crypto bar dated *t* therefore
ends three to four hours **after** the equity bar dated *t*, and contains news
the equity bar could not.

The consequence is spurious lead-lag structure: same-day crypto/equity
correlations are contaminated by crypto having "seen" more of the day, and
lagged correlations are biased in the opposite direction.

**The chosen convention is exchange-native dating.** Nothing is re-based. This
is a documented bias rather than a solved problem, and the honesty matters more
than the fix: re-basing crypto to the equity close would introduce an
undocumented interpolation in place of a stated offset. Cross-asset correlations
involving crypto should be read with this in mind, and the effect is largest at
daily frequency — it washes out at weekly.

### 12.4 Sample period asymmetry

Crypto history is short and covers roughly one full cycle. The Fama-French
factors reach back to 1963; BTC/USDT on Binance begins in 2017. An HMM fitted on
9,000 daily market observations spanning three decades and several distinct
crises is estimating transition probabilities from a genuinely diverse sample;
the same model on 3,169 BTC observations covering one boom and one bust is not.

**Regime results across these two are not comparable**, and the difference is
not one of statistical precision but of coverage: the crypto sample contains no
observation of, say, a sustained high-inflation regime. Regime frequencies and
expected durations estimated on it describe that single cycle, not a stationary
process.

A further wrinkle: the pairs are quoted in **USDT, not USD**. Tether's peg has
broken before — to roughly $0.92 in October 2018, inside the sample — so a USDT
pair is not a clean USD series. It is the deepest free history available, which
is the trade being made explicitly rather than silently.

### 12.5 Fat tails — what the data actually says

The expectation going in was that crypto would show materially heavier tails,
and therefore materially larger GPD shape parameters ξ in the peaks-over-
threshold model. **Measured on 2018-2026 daily returns, it does not.**

| Asset | Ann. vol | Skew | Excess kurtosis | GPD ξ | EVT VaR (99%) |
|---|---|---|---|---|---|
| SPY | 19.1% | −0.29 | 13.1 | 0.197 | 3.50% |
| TLT | 15.4% | +0.17 | 5.1 | 0.175 | 2.42% |
| GLD | 16.9% | −0.63 | 7.1 | 0.230 | 2.95% |
| BTC-USD | 64.4% | −0.37 | 9.3 | 0.197 | 9.21% |
| ETH-USD | 84.9% | −0.16 | 6.8 | 0.187 | 12.00% |

BTC and SPY have essentially the same tail *shape* (ξ = 0.197 for both), and
SPY's excess kurtosis is in fact the **highest** in the table. What separates
crypto is not the shape of the tail but its **scale**: BTC's 99% one-day VaR is
2.6× SPY's, and that ratio is almost exactly the ratio of their volatilities.

Two things follow. First, the intuition that "crypto has fatter tails" conflates
tail shape with volatility; on this sample the extra risk is overwhelmingly a
scale effect, and a model that gets the volatility right needs no special tail
treatment. Second, ξ ≈ 0.2 for every asset here means the same qualitative
regime — heavy-tailed, finite variance, infinite fifth moment — so the same EVT
machinery applies unchanged.

This result is reported rather than smoothed away, and it is a within-sample
finding on one cycle: see §12.4 before generalising it.

---

## 13. General references & further reading

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
