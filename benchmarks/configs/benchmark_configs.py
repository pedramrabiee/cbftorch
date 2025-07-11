"""
Configuration settings for benchmark tests.
"""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    
    # General settings
    batch_sizes: List[int]
    num_runs: int
    warmup_runs: int
    max_rel_deg: int
    max_num_barriers: int
    
    # Tolerance settings
    absolute_tolerance: float
    relative_tolerance: float
    
    # Output settings
    save_results: bool
    output_dir: str
    plot_results: bool
    
    # Test selection
    run_correctness_tests: bool
    run_performance_tests: bool
    run_stability_tests: bool
    run_regression_tests: bool
    
    # Advanced settings
    enable_cuda_sync: bool
    profile_memory: bool
    
    
def get_default_config() -> BenchmarkConfig:
    """Get default benchmark configuration."""
    return BenchmarkConfig(
        batch_sizes=[1, 10, 50, 100],
        num_runs=50,
        warmup_runs=10,
        max_rel_deg=5,
        max_num_barriers=20,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-8,
        save_results=True,
        output_dir="benchmarks/results",
        plot_results=True,
        run_correctness_tests=True,
        run_performance_tests=True,
        run_stability_tests=True,
        run_regression_tests=False,
        enable_cuda_sync=True,
        profile_memory=True
    )


def get_quick_test_config() -> BenchmarkConfig:
    """Get configuration for quick testing during development."""
    return BenchmarkConfig(
        batch_sizes=[1, 10],
        num_runs=10,
        warmup_runs=2,
        max_rel_deg=3,
        max_num_barriers=5,
        absolute_tolerance=1e-8,
        relative_tolerance=1e-6,
        save_results=False,
        output_dir="benchmarks/results",
        plot_results=False,
        run_correctness_tests=True,
        run_performance_tests=True,
        run_stability_tests=False,
        run_regression_tests=False,
        enable_cuda_sync=False,
        profile_memory=False
    )


def get_comprehensive_config() -> BenchmarkConfig:
    """Get configuration for comprehensive benchmarking."""
    return BenchmarkConfig(
        batch_sizes=[1, 5, 10, 25, 50, 100, 200, 500],
        num_runs=100,
        warmup_runs=20,
        max_rel_deg=8,
        max_num_barriers=50,
        absolute_tolerance=1e-12,
        relative_tolerance=1e-10,
        save_results=True,
        output_dir="benchmarks/results",
        plot_results=True,
        run_correctness_tests=True,
        run_performance_tests=True,
        run_stability_tests=True,
        run_regression_tests=True,
        enable_cuda_sync=True,
        profile_memory=True
    )


def get_optimization_config() -> BenchmarkConfig:
    """Get configuration specifically for optimization development."""
    return BenchmarkConfig(
        batch_sizes=[1, 10, 50, 100],
        num_runs=30,
        warmup_runs=5,
        max_rel_deg=5,
        max_num_barriers=20,
        absolute_tolerance=1e-10,
        relative_tolerance=1e-8,
        save_results=True,
        output_dir="benchmarks/results",
        plot_results=True,
        run_correctness_tests=True,
        run_performance_tests=True,
        run_stability_tests=True,
        run_regression_tests=True,
        enable_cuda_sync=True,
        profile_memory=True
    )