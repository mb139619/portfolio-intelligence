"""
Dashboard export — turn a portfolio specification into a static payload.

The pipeline is deliberately one-directional and offline:

    portfolio.json → ingestion → analytics → analysis.json → static SPA

Nothing computes at view time. The analysis runs once, locally, where Yahoo is
reachable and memory is free, and the frontend is a static file that renders a
result someone else already computed. That is what keeps hosting free and makes
the universe unbounded: any ticker you can download is a ticker you can chart.

    python -m src.export --serve
"""

from src.export.spec import BuildSettings, PortfolioSpec, PositionSpec

__all__ = ["PortfolioSpec", "PositionSpec", "BuildSettings"]
