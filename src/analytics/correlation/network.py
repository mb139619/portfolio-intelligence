"""
Minimum Spanning Tree (Mantegna's asset tree) + network centrality.

The MST keeps only the N-1 strongest links needed to connect every asset,
stripping correlation noise down to its backbone. The result is a tree whose
shape reveals the risk topology:
  - hubs (high-degree nodes) are assets the rest of the universe hangs off;
    they are systemic and dominate diversification.
  - peripheral leaves are the genuine diversifiers.

This is the analytical core of the Risk Topology Map (Phase 3 visualisation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries
from src.analytics.correlation.matrices import (
    correlation_matrix, correlation_distance,
)


@dataclass
class MSTEdge:
    source: str
    target: str
    distance: float
    correlation: float


@dataclass
class MinimumSpanningTree:
    tickers: list[str]
    edges: list[MSTEdge]
    degree: dict[str, int]            # number of links per node

    def total_distance(self) -> float:
        return sum(e.distance for e in self.edges)

    def hubs(self, top: int = 3) -> list[tuple[str, int]]:
        """Most connected nodes — the systemic centres of the network."""
        return sorted(self.degree.items(), key=lambda kv: kv[1], reverse=True)[:top]

    def leaves(self) -> list[str]:
        """Degree-1 nodes — the peripheral genuine diversifiers."""
        return [t for t, d in self.degree.items() if d == 1]

    def edges_dataframe(self) -> pl.DataFrame:
        return pl.DataFrame({
            "source": [e.source for e in self.edges],
            "target": [e.target for e in self.edges],
            "distance": [e.distance for e in self.edges],
            "correlation": [e.correlation for e in self.edges],
        })

    def summary(self) -> str:
        lines = [
            f"-- Minimum Spanning Tree ({len(self.tickers)} nodes, "
            f"{len(self.edges)} edges) --",
            "  Hubs (most connected):",
        ]
        for t, d in self.hubs():
            lines.append(f"    {t}: degree {d}")
        leaves = self.leaves()
        lines.append(f"  Leaves (diversifiers): {', '.join(leaves)}")
        return "\n".join(lines)


def build_mst(
    rs: ReturnSeries,
    corr_method: str = "pearson",
) -> MinimumSpanningTree:
    """Build the Mantegna asset tree from the correlation distance matrix."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    corr = correlation_matrix(rs, method=corr_method)
    dist = correlation_distance(corr)

    mst = minimum_spanning_tree(dist).toarray()   # upper-triangular tree
    tickers = rs.tickers
    degree = {t: 0 for t in tickers}
    edges: list[MSTEdge] = []

    rows, cols = np.nonzero(mst)
    for i, j in zip(rows, cols):
        edges.append(MSTEdge(
            source=tickers[i],
            target=tickers[j],
            distance=float(mst[i, j]),
            correlation=float(corr[i, j]),
        ))
        degree[tickers[i]] += 1
        degree[tickers[j]] += 1

    return MinimumSpanningTree(tickers=tickers, edges=edges, degree=degree)
