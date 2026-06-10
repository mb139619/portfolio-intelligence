"""
Correlation clustering via hierarchical clustering on the Mantegna distance.

Produces:
  - a linkage matrix (the dendrogram structure)
  - cluster assignments at a chosen number of clusters
  - the quasi-diagonal ordering (leaf order of the dendrogram)

The quasi-diagonal order reorders the correlation matrix so that highly
correlated assets sit next to each other — turning a noisy heatmap into a
block-diagonal one where the cluster structure is visible. It is also the
exact ordering step used by Hierarchical Risk Parity (Phase 4), so building
it here pays off twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries
from src.analytics.correlation.matrices import (
    correlation_matrix, correlation_distance,
)


@dataclass
class ClusteringResult:
    tickers: list[str]
    linkage: np.ndarray = field(repr=False)        # scipy linkage matrix
    quasi_diagonal_order: list[int]                 # leaf order (indices)
    cluster_labels: dict[str, int]                  # ticker -> cluster id
    n_clusters: int

    def ordered_tickers(self) -> list[str]:
        return [self.tickers[i] for i in self.quasi_diagonal_order]

    def clusters(self) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        for t, c in self.cluster_labels.items():
            out.setdefault(c, []).append(t)
        return out

    def summary(self) -> str:
        lines = [f"-- Correlation Clusters ({self.n_clusters}) --"]
        for cid, members in sorted(self.clusters().items()):
            lines.append(f"  Cluster {cid}: {', '.join(members)}")
        return "\n".join(lines)


def cluster_correlations(
    rs: ReturnSeries,
    n_clusters: int = 3,
    corr_method: str = "pearson",
    linkage_method: str = "ward",
) -> ClusteringResult:
    """
    Hierarchically cluster assets by correlation.

    linkage_method:
      "ward"    — compact, balanced clusters (good default for interpretation)
      "single"  — nearest-neighbour; matches the MST topology and is the
                  classic choice for HRP
      "average" — UPGMA
    """
    from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
    from scipy.spatial.distance import squareform

    corr = correlation_matrix(rs, method=corr_method)
    dist = correlation_distance(corr)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=linkage_method)
    order = leaves_list(Z).tolist()
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    label_map = {rs.tickers[i]: int(labels[i]) for i in range(len(rs.tickers))}

    return ClusteringResult(
        tickers=rs.tickers,
        linkage=Z,
        quasi_diagonal_order=order,
        cluster_labels=label_map,
        n_clusters=n_clusters,
    )


def reorder_correlation(corr: np.ndarray, order: list[int]) -> np.ndarray:
    """Apply a quasi-diagonal ordering to a correlation matrix."""
    idx = np.array(order)
    return corr[np.ix_(idx, idx)]
