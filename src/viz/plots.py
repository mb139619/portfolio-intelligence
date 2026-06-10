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

def plot_mst_network(mst, title: str = "Risk topology — Minimum Spanning Tree") -> go.Figure:
    """
    Network graph of the asset tree. Node size scales with degree (hubs are
    bigger), edge width scales with correlation strength. A simple spring
    layout is computed locally (no networkx dependency).
    """
    tickers = mst.tickers
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}

    # --- spring layout (Fruchterman-Reingold, lightweight) ---
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
