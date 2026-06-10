"""
PCA Risk Model.

Eigendecomposition of the asset return covariance (or correlation) matrix.
This exposes the *latent* structure of risk: a handful of principal components
usually explain most of the joint variation, and the first PC is typically a
broad "level/market" direction.

Conventions:
  - Eigendecomposition is done on the DAILY covariance (or correlation) matrix.
    Eigenvalues are therefore in daily-variance units; explained-variance
    ratios are unit-free and always interpretable. Annualised figures are
    provided where meaningful.
  - Eigenvectors (loadings) are sign-fixed so the largest-magnitude loading of
    each PC is positive — this makes PC1 typically all-positive (market-like)
    and keeps signs stable across refits.

Outputs feed: scree plot, factor loadings heatmap, eigen-portfolios, and the
Hidden Concentration Detector (see concentration.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from src.domain.returns import ReturnSeries


@dataclass
class PCARiskModel:
    tickers: list[str]
    method: str                            # "covariance" or "correlation"
    eigenvalues: np.ndarray                # daily, descending
    explained_variance_ratio: np.ndarray   # sums to 1
    cumulative_variance: np.ndarray
    loadings: np.ndarray = field(repr=False)   # (N, N): columns are PCs (eigenvectors)
    scores: np.ndarray = field(repr=False)     # (T, N): PC time series
    ppy: int = 252

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def pc_volatility_annualized(self) -> np.ndarray:
        """Annualised volatility of each principal component."""
        return np.sqrt(np.maximum(self.eigenvalues, 0) * self.ppy)

    def n_factors_for(self, threshold: float = 0.90) -> int:
        """Number of PCs needed to explain `threshold` of total variance."""
        return int(np.searchsorted(self.cumulative_variance, threshold) + 1)

    def scree_data(self) -> pl.DataFrame:
        n = self.n_assets
        return pl.DataFrame({
            "pc": [f"PC{i+1}" for i in range(n)],
            "eigenvalue": self.eigenvalues.tolist(),
            "explained": self.explained_variance_ratio.tolist(),
            "cumulative": self.cumulative_variance.tolist(),
        })

    def loadings_dataframe(self, n_components: int | None = None) -> pl.DataFrame:
        """Loadings as a DataFrame: ticker | PC1 | PC2 | ... (rows = assets)."""
        k = n_components or self.n_assets
        data = {"ticker": self.tickers}
        for i in range(k):
            data[f"PC{i+1}"] = self.loadings[:, i].tolist()
        return pl.DataFrame(data)

    def eigen_portfolio(self, i: int, normalize: str = "sum") -> dict[str, float]:
        """
        The i-th eigen-portfolio (0-indexed) as ticker -> weight.

        normalize:
          "sum"  : weights sum to 1 (long/short portfolio interpretation)
          "unit" : raw unit-norm eigenvector
        """
        v = self.loadings[:, i].copy()
        if normalize == "sum":
            s = v.sum()
            if abs(s) > 1e-12:
                v = v / s
        return dict(zip(self.tickers, v.tolist()))

    def summary(self) -> str:
        lines = [
            f"-- PCA Risk Model ({self.method}, {self.n_assets} assets) --",
            f"  PCs for 90% variance: {self.n_factors_for(0.90)}",
            f"  {'PC':<6} {'explained':>10} {'cumulative':>11} {'ann.vol':>9}",
        ]
        for i in range(self.n_assets):
            lines.append(
                f"  PC{i+1:<4} {self.explained_variance_ratio[i]:>10.1%} "
                f"{self.cumulative_variance[i]:>11.1%} "
                f"{self.pc_volatility_annualized[i]:>9.2%}"
            )
        return "\n".join(lines)


def _fix_signs(eigvecs: np.ndarray) -> np.ndarray:
    """Flip each eigenvector so its largest-magnitude component is positive."""
    out = eigvecs.copy()
    for j in range(out.shape[1]):
        idx = np.argmax(np.abs(out[:, j]))
        if out[idx, j] < 0:
            out[:, j] *= -1
    return out


def fit_pca(
    rs: ReturnSeries,
    method: str = "covariance",
    ppy: int = 252,
) -> PCARiskModel:
    """
    Fit a PCA risk model on a ReturnSeries.

    method:
      "covariance"  — PCA on the (daily) covariance matrix. Preserves the
                      actual risk magnitudes; right choice for risk work.
      "correlation" — PCA on the correlation matrix. Scale-free; useful when
                      assets have very different volatilities and you want to
                      see co-movement structure rather than magnitude.
    """
    R = rs.to_numpy()                      # (T, N)
    T, N = R.shape
    Rd = R - R.mean(axis=0)                # demean

    if method == "covariance":
        M = np.cov(Rd, rowvar=False, ddof=1)
        proj = Rd
    elif method == "correlation":
        std = Rd.std(axis=0, ddof=1)
        std = np.where(std == 0, 1.0, std)
        Rstd = Rd / std
        M = np.corrcoef(Rstd, rowvar=False)
        proj = Rstd
    else:
        raise ValueError(f"Unknown method: {method}")

    # Symmetric eigendecomposition (ascending) → reverse to descending
    eigvals, eigvecs = np.linalg.eigh(M)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.0)   # clip tiny negatives
    eigvecs = _fix_signs(eigvecs[:, order])

    total = eigvals.sum()
    explained = eigvals / total if total > 0 else np.zeros_like(eigvals)
    cumulative = np.cumsum(explained)

    scores = proj @ eigvecs                # (T, N)

    return PCARiskModel(
        tickers=rs.tickers,
        method=method,
        eigenvalues=eigvals,
        explained_variance_ratio=explained,
        cumulative_variance=cumulative,
        loadings=eigvecs,
        scores=scores,
        ppy=ppy,
    )
