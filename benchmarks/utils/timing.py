"""
Precision timing utilities for benchmarking CBFtorch operations.
"""

import time
import torch
import numpy as np
from typing import Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class TimingResult:
    """Container for timing measurement results."""
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    num_runs: int
    total_time: float
    
    def __str__(self):
        return (f"TimingResult(mean={self.mean_time:.6f}s, "
                f"std={self.std_time:.6f}s, runs={self.num_runs})")


class Timer:
    """High-precision timer for benchmarking operations."""
    
    def __init__(self, cuda_sync: bool = True):
        """
        Initialize timer.
        
        Args:
            cuda_sync: Whether to synchronize CUDA operations for accurate timing
        """
        self.cuda_sync = cuda_sync and torch.cuda.is_available()
        
    @contextmanager
    def time_context(self):
        """Context manager for timing code blocks."""
        if self.cuda_sync:
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            if self.cuda_sync:
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            self.last_elapsed = end_time - start_time
    
    def time_function(self, func: Callable, *args, **kwargs) -> float:
        """Time a single function call."""
        with self.time_context():
            result = func(*args, **kwargs)
        return self.last_elapsed, result


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 100,
    warmup_runs: int = 10,
    cuda_sync: bool = True
) -> TimingResult:
    """
    Benchmark a function with multiple runs and statistical analysis.
    
    Args:
        func: Function to benchmark
        args: Arguments to pass to function
        kwargs: Keyword arguments to pass to function
        num_runs: Number of timing runs
        warmup_runs: Number of warmup runs (not timed)
        cuda_sync: Whether to sync CUDA operations
        
    Returns:
        TimingResult with statistical timing information
    """
    if kwargs is None:
        kwargs = {}
    
    timer = Timer(cuda_sync=cuda_sync)
    
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(num_runs):
        elapsed, _ = timer.time_function(func, *args, **kwargs)
        times.append(elapsed)
    
    times = np.array(times)
    
    return TimingResult(
        mean_time=float(np.mean(times)),
        std_time=float(np.std(times)),
        min_time=float(np.min(times)),
        max_time=float(np.max(times)),
        num_runs=num_runs,
        total_time=float(np.sum(times))
    )


class BatchTimer:
    """Timer for measuring batch processing performance."""
    
    def __init__(self, cuda_sync: bool = True):
        self.timer = Timer(cuda_sync)
        self.batch_times = []
        
    def time_batch(self, func: Callable, batch_data: List[Any]) -> List[float]:
        """Time function on each item in batch."""
        batch_times = []
        for item in batch_data:
            elapsed, _ = self.timer.time_function(func, item)
            batch_times.append(elapsed)
        
        self.batch_times.extend(batch_times)
        return batch_times
    
    def get_batch_stats(self) -> Dict[str, float]:
        """Get statistical summary of batch timing."""
        if not self.batch_times:
            return {}
            
        times = np.array(self.batch_times)
        return {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
            'min': float(np.min(times)),
            'max': float(np.max(times)),
            'total': float(np.sum(times)),
            'count': len(times)
        }


def compare_functions(
    func1: Callable,
    func2: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, TimingResult]:
    """
    Compare timing of two functions.
    
    Returns:
        Dictionary with timing results for each function
    """
    if kwargs is None:
        kwargs = {}
    
    result1 = benchmark_function(func1, args, kwargs, num_runs, warmup_runs)
    result2 = benchmark_function(func2, args, kwargs, num_runs, warmup_runs)
    
    return {
        'function1': result1,
        'function2': result2,
        'speedup': result1.mean_time / result2.mean_time
    }