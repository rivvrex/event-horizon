"""Module 4 public surface."""

from factor_lab.metrics.analytics import (
    BenchmarkComparison,
    DrawdownInfo,
    PerformanceAnalyzer,
    PerformanceMetrics,
)
from factor_lab.metrics.benchmark import SP500_PROXY, BenchmarkLoader
from factor_lab.metrics.periodicity import Periodicity, resample_returns

__all__ = [
    "SP500_PROXY",
    "BenchmarkComparison",
    "BenchmarkLoader",
    "DrawdownInfo",
    "PerformanceAnalyzer",
    "PerformanceMetrics",
    "Periodicity",
    "resample_returns",
]
