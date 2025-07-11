"""
Memory and CPU profiling utilities for CBFtorch benchmarking.
"""

import psutil
import torch
import gc
import tracemalloc
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a point in time."""
    rss_mb: float  # Resident Set Size in MB
    vms_mb: float  # Virtual Memory Size in MB
    cpu_percent: float
    gpu_allocated_mb: Optional[float] = None
    gpu_cached_mb: Optional[float] = None
    
    def __str__(self):
        gpu_str = ""
        if self.gpu_allocated_mb is not None:
            gpu_str = f", GPU: {self.gpu_allocated_mb:.1f}MB"
        return f"Memory(RSS: {self.rss_mb:.1f}MB, CPU: {self.cpu_percent:.1f}%{gpu_str})"


@dataclass 
class ProfileResults:
    """Results from profiling a function execution."""
    execution_time: float
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    peak_memory: MemorySnapshot
    memory_delta: Dict[str, float]
    
    def __str__(self):
        return (f"ProfileResults(\n"
                f"  Time: {self.execution_time:.6f}s\n"
                f"  Memory Delta: RSS={self.memory_delta['rss_mb']:.1f}MB, "
                f"GPU={self.memory_delta.get('gpu_allocated_mb', 'N/A')}\n"
                f"  Peak: {self.peak_memory}\n"
                f")")


class MemoryProfiler:
    """Memory and performance profiler for CBFtorch operations."""
    
    def __init__(self, track_gpu: bool = None):
        """
        Initialize profiler.
        
        Args:
            track_gpu: Whether to track GPU memory. Auto-detected if None.
        """
        self.track_gpu = track_gpu if track_gpu is not None else torch.cuda.is_available()
        self.process = psutil.Process()
        
    def get_memory_snapshot(self) -> MemorySnapshot:
        """Get current memory usage snapshot."""
        memory_info = self.process.memory_info()
        cpu_percent = self.process.cpu_percent()
        
        snapshot = MemorySnapshot(
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            cpu_percent=cpu_percent
        )
        
        if self.track_gpu and torch.cuda.is_available():
            snapshot.gpu_allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            snapshot.gpu_cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
            
        return snapshot
    
    @contextmanager
    def profile_context(self):
        """Context manager for profiling code blocks."""
        # Clear caches and run garbage collection
        gc.collect()
        if self.track_gpu:
            torch.cuda.empty_cache()
        
        # Start profiling
        import time
        start_time = time.perf_counter()
        memory_before = self.get_memory_snapshot()
        peak_memory = memory_before
        
        try:
            yield self
        finally:
            end_time = time.perf_counter()
            memory_after = self.get_memory_snapshot()
            
            # Calculate deltas
            memory_delta = {
                'rss_mb': memory_after.rss_mb - memory_before.rss_mb,
                'vms_mb': memory_after.vms_mb - memory_before.vms_mb,
            }
            
            if self.track_gpu and memory_after.gpu_allocated_mb is not None:
                memory_delta['gpu_allocated_mb'] = (
                    memory_after.gpu_allocated_mb - memory_before.gpu_allocated_mb
                )
                memory_delta['gpu_cached_mb'] = (
                    memory_after.gpu_cached_mb - memory_before.gpu_cached_mb
                )
            
            self.last_results = ProfileResults(
                execution_time=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                peak_memory=peak_memory,  # TODO: Track actual peak during execution
                memory_delta=memory_delta
            )
    
    def profile_function(self, func: Callable, *args, **kwargs) -> ProfileResults:
        """Profile a single function call."""
        with self.profile_context():
            result = func(*args, **kwargs)
        return self.last_results, result


class DetailedMemoryProfiler:
    """Detailed memory profiler using tracemalloc for Python memory tracking."""
    
    def __init__(self):
        self.is_tracing = False
        
    @contextmanager
    def trace_context(self):
        """Context manager for detailed memory tracing."""
        if not self.is_tracing:
            tracemalloc.start()
            self.is_tracing = True
            
        snapshot_before = tracemalloc.take_snapshot()
        
        try:
            yield
        finally:
            snapshot_after = tracemalloc.take_snapshot()
            
            # Analyze top differences
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            self.last_top_stats = top_stats[:10]  # Top 10 memory allocations
            
    def get_top_allocations(self, limit: int = 10) -> list:
        """Get top memory allocations from last trace."""
        if hasattr(self, 'last_top_stats'):
            return self.last_top_stats[:limit]
        return []


def profile_memory_leak(func: Callable, iterations: int = 100) -> Dict[str, Any]:
    """
    Profile potential memory leaks by running function multiple times.
    
    Args:
        func: Function to test for memory leaks
        iterations: Number of iterations to run
        
    Returns:
        Dictionary with memory usage over iterations
    """
    profiler = MemoryProfiler()
    memory_usage = []
    
    for i in range(iterations):
        with profiler.profile_context():
            func()
        
        snapshot = profiler.get_memory_snapshot()
        memory_usage.append({
            'iteration': i,
            'rss_mb': snapshot.rss_mb,
            'gpu_allocated_mb': snapshot.gpu_allocated_mb
        })
        
        # Force garbage collection every 10 iterations
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return {
        'memory_usage': memory_usage,
        'initial_memory': memory_usage[0]['rss_mb'],
        'final_memory': memory_usage[-1]['rss_mb'],
        'memory_growth': memory_usage[-1]['rss_mb'] - memory_usage[0]['rss_mb']
    }