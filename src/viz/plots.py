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
import polars as pl

from src.domain.returns import ReturnSeries
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
