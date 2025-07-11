"""
Comparison utilities for before/after optimization analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from .timing import TimingResult, benchmark_function
from .profiling import ProfileResults, MemoryProfiler


@dataclass
class ComparisonResult:
    """Results from comparing two implementations."""
    timing_comparison: Dict[str, TimingResult]
    memory_comparison: Dict[str, ProfileResults]
    speedup_factor: float
    memory_reduction: float
    correctness_check: bool
    max_error: Optional[float] = None
    
    def __str__(self):
        return (f"ComparisonResult(\n"
                f"  Speedup: {self.speedup_factor:.2f}x\n"
                f"  Memory reduction: {self.memory_reduction:.1f}MB\n"
                f"  Correctness: {'✓' if self.correctness_check else '✗'}\n"
                f"  Max error: {self.max_error or 'N/A'}\n"
                f")")


class BenchmarkComparator:
    """Utility for comparing original vs optimized implementations."""
    
    def __init__(self, tolerance: float = 1e-6):
        """
        Initialize comparator.
        
        Args:
            tolerance: Numerical tolerance for correctness checking
        """
        self.tolerance = tolerance
        self.profiler = MemoryProfiler()
        
    def compare_implementations(
        self,
        original_func: Callable,
        optimized_func: Callable,
        test_inputs: List[Any],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> ComparisonResult:
        """
        Comprehensive comparison of two implementations.
        
        Args:
            original_func: Original implementation
            optimized_func: Optimized implementation  
            test_inputs: List of test inputs to run on both functions
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            
        Returns:
            ComparisonResult with detailed analysis
        """
        # Timing comparison
        timing_results = {}
        memory_results = {}
        correctness_results = []
        max_errors = []
        
        for i, inputs in enumerate(test_inputs):
            # Benchmark timing
            if isinstance(inputs, tuple):
                args, kwargs = inputs, {}
            else:
                args, kwargs = (inputs,), {}
                
            timing_orig = benchmark_function(
                original_func, args, kwargs, num_runs, warmup_runs
            )
            timing_opt = benchmark_function(
                optimized_func, args, kwargs, num_runs, warmup_runs
            )
            
            timing_results[f'test_{i}'] = {
                'original': timing_orig,
                'optimized': timing_opt,
                'speedup': timing_orig.mean_time / timing_opt.mean_time
            }
            
            # Memory comparison
            with self.profiler.profile_context():
                result_orig = original_func(*args, **kwargs)
            memory_orig = self.profiler.last_results
            
            with self.profiler.profile_context():
                result_opt = optimized_func(*args, **kwargs)
            memory_opt = self.profiler.last_results
            
            memory_results[f'test_{i}'] = {
                'original': memory_orig,
                'optimized': memory_opt
            }
            
            # Correctness check
            is_correct, max_error = self._check_correctness(result_orig, result_opt)
            correctness_results.append(is_correct)
            if max_error is not None:
                max_errors.append(max_error)
        
        # Aggregate results
        avg_speedup = np.mean([
            result['speedup'] for result in timing_results.values()
        ])
        
        avg_memory_reduction = np.mean([
            result['original'].memory_delta['rss_mb'] - 
            result['optimized'].memory_delta['rss_mb']
            for result in memory_results.values()
        ])
        
        overall_correctness = all(correctness_results)
        overall_max_error = max(max_errors) if max_errors else None
        
        return ComparisonResult(
            timing_comparison=timing_results,
            memory_comparison=memory_results,
            speedup_factor=avg_speedup,
            memory_reduction=avg_memory_reduction,
            correctness_check=overall_correctness,
            max_error=overall_max_error
        )
    
    def _check_correctness(
        self, 
        result1: Any, 
        result2: Any
    ) -> Tuple[bool, Optional[float]]:
        """Check if two results are numerically equivalent."""
        try:
            if isinstance(result1, torch.Tensor) and isinstance(result2, torch.Tensor):
                diff = torch.abs(result1 - result2)
                max_error = torch.max(diff).item()
                is_correct = torch.allclose(result1, result2, atol=self.tolerance)
                return is_correct, max_error
            
            elif isinstance(result1, (list, tuple)) and isinstance(result2, (list, tuple)):
                if len(result1) != len(result2):
                    return False, None
                
                max_errors = []
                all_correct = True
                
                for r1, r2 in zip(result1, result2):
                    is_correct, max_error = self._check_correctness(r1, r2)
                    all_correct &= is_correct
                    if max_error is not None:
                        max_errors.append(max_error)
                
                return all_correct, max(max_errors) if max_errors else None
            
            elif isinstance(result1, dict) and isinstance(result2, dict):
                if set(result1.keys()) != set(result2.keys()):
                    return False, None
                
                max_errors = []
                all_correct = True
                
                for key in result1.keys():
                    is_correct, max_error = self._check_correctness(
                        result1[key], result2[key]
                    )
                    all_correct &= is_correct
                    if max_error is not None:
                        max_errors.append(max_error)
                
                return all_correct, max(max_errors) if max_errors else None
            
            else:
                # For non-tensor types, use numpy if possible
                try:
                    arr1 = np.array(result1)
                    arr2 = np.array(result2)
                    max_error = np.max(np.abs(arr1 - arr2))
                    is_correct = np.allclose(arr1, arr2, atol=self.tolerance)
                    return is_correct, float(max_error)
                except:
                    # Fallback to direct comparison
                    return result1 == result2, None
                    
        except Exception as e:
            print(f"Warning: Correctness check failed with error: {e}")
            return False, None
    
    def plot_comparison(
        self, 
        comparison_result: ComparisonResult,
        save_path: Optional[str] = None
    ):
        """Plot comparison results."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Speedup plot
        test_names = list(comparison_result.timing_comparison.keys())
        speedups = [
            result['speedup'] 
            for result in comparison_result.timing_comparison.values()
        ]
        
        axes[0, 0].bar(test_names, speedups)
        axes[0, 0].set_title('Speedup Factor')
        axes[0, 0].set_ylabel('Speedup (x)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Timing comparison
        orig_times = [
            result['original'].mean_time 
            for result in comparison_result.timing_comparison.values()
        ]
        opt_times = [
            result['optimized'].mean_time 
            for result in comparison_result.timing_comparison.values()
        ]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        axes[0, 1].bar(x - width/2, orig_times, width, label='Original')
        axes[0, 1].bar(x + width/2, opt_times, width, label='Optimized')
        axes[0, 1].set_title('Execution Time Comparison')
        axes[0, 1].set_ylabel('Time (s)')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(test_names, rotation=45)
        axes[0, 1].legend()
        
        # Memory usage comparison
        orig_memory = [
            result['original'].memory_delta['rss_mb']
            for result in comparison_result.memory_comparison.values()
        ]
        opt_memory = [
            result['optimized'].memory_delta['rss_mb']
            for result in comparison_result.memory_comparison.values()
        ]
        
        axes[1, 0].bar(x - width/2, orig_memory, width, label='Original')
        axes[1, 0].bar(x + width/2, opt_memory, width, label='Optimized')
        axes[1, 0].set_title('Memory Usage Comparison')
        axes[1, 0].set_ylabel('Memory (MB)')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(test_names, rotation=45)
        axes[1, 0].legend()
        
        # Summary text
        summary_text = (
            f"Overall Speedup: {comparison_result.speedup_factor:.2f}x\n"
            f"Memory Reduction: {comparison_result.memory_reduction:.1f}MB\n"
            f"Correctness: {'✓' if comparison_result.correctness_check else '✗'}\n"
            f"Max Error: {comparison_result.max_error or 'N/A'}"
        )
        
        axes[1, 1].text(0.1, 0.5, summary_text, transform=axes[1, 1].transAxes,
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig