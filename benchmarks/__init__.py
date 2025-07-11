"""
CBFtorch Benchmarking Suite

This module provides comprehensive benchmarking and testing infrastructure
for measuring performance improvements and ensuring correctness of optimizations.
"""

__version__ = "0.1.0"

from .utils.timing import Timer, benchmark_function
from .utils.profiling import MemoryProfiler, ProfileResults
from .utils.comparison import BenchmarkComparator

__all__ = [
    "Timer",
    "benchmark_function", 
    "MemoryProfiler",
    "ProfileResults",
    "BenchmarkComparator",
]