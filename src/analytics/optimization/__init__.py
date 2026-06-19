"""Portfolio optimisation — objectives that consume a CovarianceResult."""
from src.analytics.optimization.frontier import EfficientFrontier, efficient_frontier
from src.analytics.optimization.minimum_variance import min_variance
from src.analytics.optimization.result import OptimizationResult

__all__ = [
    "OptimizationResult",
    "min_variance",
    "EfficientFrontier",
    "efficient_frontier",
]
