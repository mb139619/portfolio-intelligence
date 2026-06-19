"""
Reusable visualization layer.

Every function takes analytics output (or a ReturnSeries) and returns a Plotly
Figure — it never calls .show() and never touches the network or the store.
This keeps plotting pure and composable: the notebook decides when to render,
a future Dash/Streamlit app can reuse the exact same figures.

A single muted, consistent palette is defined here so every chart in the
platform looks like it belongs to the same product.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.analytics.performance import drawdown_series

# --- Shared palette -------------------------------------------------------
PALETTE = [
    "#2E5EAA", "#C44E52", "#55A868", "#8172B3",
    "#CCB974", "#64B5CD", "#E08E45", "#937860",
]
POSITIVE = "#55A868"
NEGATIVE = "#C44E52"
GRID = "rgba(0,0,0,0.08)"


def _base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=height,
        margin=dict(l=60, r=30, t=60, b=50),
        font=dict(size=12),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 1. Performance — equity curve + drawdown (two stacked panels)
# ─────────────────────────────────────────────────────────────────────────

def plot_performance(
    returns: np.ndarray,
    dates: list,
    title: str = "Portfolio performance",
) -> go.Figure:
    """
    Cumulative growth of 1 unit on top, drawdown underwater plot below.
    Shares the x-axis so peaks and troughs line up visually.
    """
    from plotly.subplots import make_subplots

    equity = np.cumprod(1 + returns)
    dd = drawdown_series(returns)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=("Growth of 1", "Drawdown"),
    )

    fig.add_trace(
        go.Scatter(x=dates, y=equity, mode="lines", name="Equity",
                   line=dict(color=PALETTE[0], width=1.8)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=dates, y=dd, mode="lines", name="Drawdown",
                   fill="tozeroy", line=dict(color=NEGATIVE, width=1),
                   fillcolor="rgba(196,78,82,0.25)"),
        row=2, col=1,
    )

    fig = _base_layout(fig, title, height=560)
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 2. Factor exposures — beta bars with significance
# ─────────────────────────────────────────────────────────────────────────

def plot_factor_exposures(model, title: str = "Factor exposures (beta)") -> go.Figure:
    """
    Horizontal bar chart of factor betas. Significant factors (p < 0.05) are
    drawn solid; non-significant ones are faded — so the eye goes to what
    actually matters. Expects a FactorModel.
    """
    factors = model.factor_names
    betas = [model.betas[f] for f in factors]
    sig = [model.beta_pvalues[f] < 0.05 for f in factors]

    colors = [
        (PALETTE[0] if b >= 0 else NEGATIVE) if s
        else ("rgba(46,94,170,0.35)" if b >= 0 else "rgba(196,78,82,0.35)")
        for b, s in zip(betas, sig)
    ]

    fig = go.Figure(go.Bar(
        x=betas, y=factors, orientation="h",
        marker_color=colors,
        text=[f"{b:+.2f}{'*' if s else ''}" for b, s in zip(betas, sig)],
        textposition="outside",
    ))
    fig = _base_layout(fig, title, height=80 + 50 * len(factors))
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.3)", line_width=1)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title="beta (HAC t-stat significant = solid, * = p<0.05)")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 3. MST network graph — the risk topology
# ─────────────────────────────────────────────────────────────────────────

def plot_mst_network(
    mst,
    title: str = "Risk topology — Minimum Spanning Tree",
    layout: str = "auto",
    small_threshold: int = 8,
) -> go.Figure:
    """
    Network graph of the asset tree. Node size scales with degree (hubs are
    bigger), edge width scales with correlation strength. No networkx dependency.

    layout:
      "auto"     — circular for small graphs (≤ small_threshold nodes) or pure
                   chains, spring otherwise. Force-directed layouts collapse a
                   5-node chain onto a straight diagonal, which hides the
                   structure; a circle keeps every node and edge readable.
      "circular" — force a ring layout.
      "spring"   — force the Fruchterman-Reingold layout.
    """
    tickers = mst.tickers
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    if layout == "auto":
        max_degree = max(mst.degree.values()) if mst.degree else 0
        is_chain = max_degree <= 2          # a path graph has no branching
        use_circular = n <= small_threshold or is_chain
    else:
        use_circular = (layout == "circular")

    if use_circular:
        pos = _circular_layout(tickers, mst.edges, idx)
    else:
        pos = _spring_layout(tickers, mst.edges, idx)

    edge_traces = []
    max_corr = max((abs(e.correlation) for e in mst.edges), default=1.0)
    for e in mst.edges:
        x0, y0 = pos[e.source]
        x1, y1 = pos[e.target]
        width = 1 + 5 * (abs(e.correlation) / max_corr)
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line=dict(width=width, color="rgba(0,0,0,0.25)"),
            hoverinfo="text",
            text=f"{e.source}–{e.target}: ρ={e.correlation:.2f}",
            showlegend=False,
        ))

    degrees = [mst.degree[t] for t in tickers]
    node_trace = go.Scatter(
        x=[pos[t][0] for t in tickers],
        y=[pos[t][1] for t in tickers],
        mode="markers+text",
        text=tickers, textposition="top center",
        marker=dict(
            size=[14 + 8 * d for d in degrees],
            color=degrees, colorscale="Blues", showscale=True,
            colorbar=dict(title="degree"),
            line=dict(width=1.5, color="white"),
        ),
        hovertext=[f"{t} (degree {mst.degree[t]})" for t in tickers],
        hoverinfo="text", showlegend=False,
    )

    fig = go.Figure(edge_traces + [node_trace])
    fig = _base_layout(fig, title, height=520)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def _circular_layout(nodes, edges, idx) -> dict:
    """
    Ring layout. Nodes are ordered around the circle by a depth-first traversal
    of the tree starting from a leaf, so that tree-adjacent nodes are also
    circle-adjacent — this keeps edges short and avoids the diagonal collapse
    that a force-directed layout produces on small/chain graphs.
    """
    import numpy as np

    # Build adjacency
    adj: dict = {t: [] for t in nodes}
    for e in edges:
        adj[e.source].append(e.target)
        adj[e.target].append(e.source)

    # Start from a leaf (degree 1) if one exists, else the first node
    start = next((t for t in nodes if len(adj[t]) == 1), nodes[0])

    # DFS ordering
    order, seen = [], set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        for nb in sorted(adj[node], key=lambda x: len(adj[x])):
            if nb not in seen:
                stack.append(nb)
    # Append any disconnected nodes (shouldn't happen for an MST)
    for t in nodes:
        if t not in seen:
            order.append(t)

    n = len(order)
    pos = {}
    for k, t in enumerate(order):
        angle = 2 * np.pi * k / n - np.pi / 2   # start at top
        pos[t] = (float(np.cos(angle)), float(np.sin(angle)))
    return pos


def _spring_layout(nodes, edges, idx, iterations: int = 200, seed: int = 42) -> dict:
    """Minimal force-directed layout; returns {ticker: (x, y)}."""
    rng = np.random.default_rng(seed)
    n = len(nodes)
    p = rng.normal(0, 1, size=(n, 2))
    k = 1.0 / np.sqrt(n)

    adj = np.zeros((n, n))
    for e in edges:
        i, j = idx[e.source], idx[e.target]
        adj[i, j] = adj[j, i] = 1

    for it in range(iterations):
        disp = np.zeros((n, 2))
        for i in range(n):
            delta = p[i] - p                      # (n,2)
            dist = np.linalg.norm(delta, axis=1)
            dist[i] = 1.0
            rep = (k * k / dist**2)[:, None] * delta
            disp[i] += rep.sum(axis=0)
            attr = adj[i][:, None] * (delta * dist[:, None] / k)
            disp[i] -= attr.sum(axis=0)
        length = np.linalg.norm(disp, axis=1)
        length[length == 0] = 1.0
        cool = 0.1 * (1 - it / iterations)
        p += (disp / length[:, None]) * np.minimum(length, cool)[:, None]

    return {nodes[i]: (float(p[i, 0]), float(p[i, 1])) for i in range(n)}


# ─────────────────────────────────────────────────────────────────────────
# 4. Regime timeline — price/equity shaded by detected regime
# ─────────────────────────────────────────────────────────────────────────

def plot_regime_timeline(
    returns: np.ndarray,
    dates: list,
    states: np.ndarray,
    labels: list,
    title: str = "Regime timeline",
) -> go.Figure:
    """
    Cumulative growth line with the background shaded by detected regime.
    The highest-index regime (stress) is shaded most prominently, making the
    crisis periods visually obvious.
    """
    equity = np.cumprod(1 + returns)
    n_states = len(labels)
    # Shade bands per contiguous run of the same state
    shade_colors = {
        i: f"rgba(196,78,82,{0.06 + 0.16 * i / max(n_states - 1, 1)})"
        for i in range(n_states)
    }

    fig = go.Figure()
    # background regime bands
    start = 0
    for i in range(1, len(states) + 1):
        if i == len(states) or states[i] != states[start]:
            s = int(states[start])
            if s > 0:  # don't shade the calm regime
                fig.add_vrect(
                    x0=dates[start], x1=dates[i - 1],
                    fillcolor=shade_colors[s], line_width=0, layer="below",
                )
            start = i

    fig.add_trace(go.Scatter(
        x=dates, y=equity, mode="lines",
        line=dict(color=PALETTE[0], width=1.6), name="Growth of 1",
    ))
    fig = _base_layout(fig, title, height=440)
    fig.update_layout(showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 5. Rolling volatility — vol clustering a headline number hides
# ─────────────────────────────────────────────────────────────────────────

def plot_rolling_volatility(
    returns: np.ndarray,
    dates: list,
    window: int = 63,
    ppy: int = 252,
    title: str = "Rolling volatility (annualised)",
) -> go.Figure:
    """
    Annualised trailing-window volatility of a single return stream, with the
    full-sample volatility drawn as a reference. Surfaces vol clustering — calm
    stretches vs stress — that one headline sigma number averages away.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < window:
        raise ValueError(f"need at least window={window} points, got {n}")

    roll = np.full(n, np.nan)
    for i in range(window, n + 1):
        roll[i - 1] = r[i - window:i].std(ddof=1) * np.sqrt(ppy)
    full = float(r.std(ddof=1) * np.sqrt(ppy))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=roll, mode="lines", name=f"{window}d rolling",
        line=dict(color=PALETTE[0], width=1.6),
    ))
    fig.add_hline(
        y=full, line=dict(color=NEGATIVE, width=1, dash="dash"),
        annotation_text=f"full-sample {full:.1%}", annotation_position="top left",
    )
    _base_layout(fig, title)
    fig.update_yaxes(tickformat=".0%", title="annualised σ", rangemode="tozero")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 6. Risk contribution vs weight — the 'concentrated in risk' tell
# ─────────────────────────────────────────────────────────────────────────

def plot_risk_contribution(
    decomp,
    title: str = "Risk contribution vs weight",
) -> go.Figure:
    """
    Per-asset capital weight against percent risk contribution, sorted by %RC.
    Where the %RC bar overshoots the weight bar, that asset carries more risk
    than its size implies — the 'diversified on paper, one bet in risk' signal.
    Takes a RiskDecomposition (uses its public to_dataframe()).
    """
    df = decomp.to_dataframe().sort("pct_risk_contribution", descending=True)
    tickers = df["ticker"].to_list()

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=tickers, x=df["weight"].to_list(), orientation="h",
        name="weight", marker_color=PALETTE[5],
    ))
    fig.add_trace(go.Bar(
        y=tickers, x=df["pct_risk_contribution"].to_list(), orientation="h",
        name="% risk contribution", marker_color=PALETTE[1],
    ))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08))
    _base_layout(fig, title)
    fig.update_xaxes(tickformat=".0%", title="share of portfolio")
    fig.update_yaxes(autorange="reversed")  # largest %RC on top
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 7. Return distribution with VaR / CVaR markers
# ─────────────────────────────────────────────────────────────────────────

def plot_var_distribution(
    returns: np.ndarray,
    confidence: float = 0.99,
    title: str | None = None,
) -> go.Figure:
    """
    Histogram of daily returns with VaR/CVaR markers. Gaussian, Cornish-Fisher
    and historical VaR sit side by side so the gap between a normal assumption
    and the real (skewed, fat-tailed) loss tail is visible at a glance.
    """
    from src.analytics.risk.tail import tail_risk_comparison

    r = np.asarray(returns, dtype=float)
    comp = tail_risk_comparison(r, confidence=confidence)
    if title is None:
        title = f"Return distribution & VaR ({confidence:.0%}, 1-day)"

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=80, marker_color=PALETTE[0], opacity=0.55, name="daily returns",
    ))
    # VaR/CVaR are positive losses → drawn at -value on the return axis.
    markers = [
        ("Gaussian VaR", comp.get("gaussian_var"), PALETTE[4], "dot"),
        ("Cornish-Fisher VaR", comp.get("cornish_fisher_var"), PALETTE[3], "dash"),
        ("Historical VaR", comp.get("historical_var"), NEGATIVE, "solid"),
        ("Historical CVaR", comp.get("historical_cvar"), "#7A1F22", "solid"),
    ]
    for i, (name, val, color, dash) in enumerate(markers):
        if val is None:
            continue
        fig.add_vline(
            x=-val, line=dict(color=color, width=1.4, dash=dash),
            annotation_text=name,
            annotation_position="top" if i % 2 == 0 else "bottom",
        )
    _base_layout(fig, title)
    fig.update_xaxes(tickformat=".1%", title="daily return")
    fig.update_yaxes(title="frequency")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 8. EVT peaks-over-threshold tail fit (diagnostic)
# ─────────────────────────────────────────────────────────────────────────

def plot_evt_tail(
    returns: np.ndarray,
    confidence: float = 0.99,
    threshold_quantile: float = 0.95,
    title: str = "EVT tail fit (peaks-over-threshold)",
) -> go.Figure:
    """
    Diagnostic for the POT/GPD fit. The empirical exceedance-survival curve
    (losses beyond the threshold u, conditional on exceeding u) is plotted on a
    log y-axis against the fitted GPD survival; a good fit tracks the points into
    the tail. EVT VaR/CVaR at `confidence` are marked on the loss axis.
    """
    from src.analytics.risk.tail import fit_evt_pot

    r = np.asarray(returns, dtype=float)
    evt = fit_evt_pot(r, confidence=confidence, threshold_quantile=threshold_quantile)
    xi, beta, u = evt.shape, evt.scale, evt.threshold

    # Reconstruct exceedances exactly as fit_evt_pot does (losses = -returns).
    losses = -r
    exceed = np.sort(losses[losses > u] - u)
    n_u = len(exceed)

    # Empirical conditional survival S(y) = P(L-u > y | L > u), Weibull positions.
    emp_surv = 1.0 - (np.arange(1, n_u + 1) - 0.5) / n_u

    # Fitted GPD survival over a smooth grid.
    grid = np.linspace(0.0, exceed.max() * 1.05, 200)
    if abs(xi) < 1e-8:
        gpd_surv = np.exp(-grid / beta)
    else:
        gpd_surv = np.power(1.0 + xi * grid / beta, -1.0 / xi)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=u + exceed, y=emp_surv, mode="markers", name="empirical",
        marker=dict(color=PALETTE[0], size=5, opacity=0.7),
    ))
    fig.add_trace(go.Scatter(
        x=u + grid, y=gpd_surv, mode="lines", name=f"GPD fit (ξ={xi:.2f})",
        line=dict(color=NEGATIVE, width=1.8),
    ))
    evt_markers = [("EVT VaR", evt.var, "dash"), ("EVT CVaR", evt.cvar, "dot")]
    for name, val, dash in evt_markers:
        if val is None or not np.isfinite(val):
            continue
        fig.add_vline(
            x=val, line=dict(color="#7A1F22", width=1.2, dash=dash),
            annotation_text=f"{name} {val:.1%}", annotation_position="top",
        )
    _base_layout(fig, title)
    fig.update_xaxes(tickformat=".1%", title="loss (−return)")
    fig.update_yaxes(type="log", title="exceedance prob. (cond. L > u)")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 9. Efficient frontier — the signature portfolio-construction chart
# ─────────────────────────────────────────────────────────────────────────

def plot_efficient_frontier(
    frontier,
    assets: tuple[list[str], np.ndarray, np.ndarray] | None = None,
    markers: list[tuple[str, float, float]] | None = None,
    cloud: tuple[np.ndarray, np.ndarray] | None = None,
    title: str = "Efficient frontier",
) -> go.Figure:
    """
    Risk/return frontier (annualised). Optionally overlays:
      assets  : (names, vols, returns) — the individual holdings,
      markers : [(name, vol, return), ...] — e.g. min-variance, equal-weight,
      cloud   : (vols, returns) — a scatter of random feasible portfolios.

    `frontier` is an EfficientFrontier (uses .volatilities and .returns).
    """
    fig = go.Figure()

    if cloud is not None:
        cv, cr = cloud
        fig.add_trace(go.Scatter(
            x=cv, y=cr, mode="markers", name="random portfolios",
            marker=dict(color="rgba(0,0,0,0.12)", size=4),
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=frontier.volatilities, y=frontier.returns, mode="lines",
        name="efficient frontier", line=dict(color=PALETTE[0], width=2.4),
    ))
    # leftmost frontier point = minimum-variance portfolio
    if len(frontier.volatilities):
        i = int(np.argmin(frontier.volatilities))
        fig.add_trace(go.Scatter(
            x=[frontier.volatilities[i]], y=[frontier.returns[i]],
            mode="markers+text", name="min-variance",
            marker=dict(color=PALETTE[0], size=11, symbol="star"),
            text=["min-var"], textposition="middle right",
        ))

    if assets is not None:
        names, avols, arets = assets
        fig.add_trace(go.Scatter(
            x=avols, y=arets, mode="markers+text", name="assets",
            marker=dict(color=PALETTE[1], size=8, symbol="circle-open"),
            text=names, textposition="top center", textfont=dict(size=10),
        ))

    if markers:
        for j, (name, v, r) in enumerate(markers):
            fig.add_trace(go.Scatter(
                x=[v], y=[r], mode="markers+text", name=name,
                marker=dict(color=PALETTE[(j + 2) % len(PALETTE)], size=11,
                            symbol="diamond"),
                text=[name], textposition="bottom center",
            ))

    _base_layout(fig, title, height=460)
    fig.update_xaxes(tickformat=".0%", title="annualised volatility")
    fig.update_yaxes(tickformat=".0%", title="annualised return")
    return fig


# ─────────────────────────────────────────────────────────────────────────
# 10. Allocation comparison across strategies
# ─────────────────────────────────────────────────────────────────────────

def plot_weights_comparison(
    weights_by_strategy: dict[str, dict[str, float]],
    title: str = "Allocation by strategy",
) -> go.Figure:
    """
    Grouped horizontal bars of portfolio weights, one colour per strategy.
    Input: {strategy_name: {ticker: weight}}. Handles negative (short) weights.
    Ticker order follows the first strategy, then any extras.
    """
    strategies = list(weights_by_strategy.keys())
    ordered: list[str] = []
    for wd in weights_by_strategy.values():
        for t in wd:
            if t not in ordered:
                ordered.append(t)

    fig = go.Figure()
    for j, strat in enumerate(strategies):
        wd = weights_by_strategy[strat]
        fig.add_trace(go.Bar(
            y=ordered, x=[wd.get(t, 0.0) for t in ordered], orientation="h",
            name=strat, marker_color=PALETTE[j % len(PALETTE)],
        ))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.08))
    _base_layout(fig, title, height=max(360, 26 * len(ordered) * len(strategies) // 2))
    fig.update_xaxes(tickformat=".0%", title="weight")
    fig.update_yaxes(autorange="reversed")
    return fig
