"""
Configuration files for benchmark tests.
"""

from .benchmark_configs import *

__all__ = [
    "BenchmarkConfig",
    "get_default_config",
    "get_quick_test_config",
    "get_comprehensive_config",
]