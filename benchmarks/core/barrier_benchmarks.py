"""
Benchmarks for basic barrier function operations.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Callable
from ..utils.timing import benchmark_function, TimingResult
from ..utils.profiling import MemoryProfiler
from cbftorch.barriers.barrier import Barrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional, make_norm_rectangular_barrier_functional


class BarrierBenchmarks:
    """Benchmarks for barrier function evaluations."""
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        self.dynamics = UnicycleDynamics()
        
    def setup_test_barriers(self) -> Dict[str, Barrier]:
        """Setup various barrier functions for testing."""
        barriers = {}
        
        # Circle barrier
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        circle_barrier = Barrier().assign(circle_func, rel_deg=1)
        circle_barrier.assign_dynamics(self.dynamics)
        barriers['circle'] = circle_barrier
        
        # Rectangle barrier
        rect_func = make_norm_rectangular_barrier_functional(
            center=torch.tensor([0.0, 0.0]),
            size=torch.tensor([2.0, 2.0])
        )
        rect_barrier = Barrier().assign(rect_func, rel_deg=1)
        rect_barrier.assign_dynamics(self.dynamics)
        barriers['rectangle'] = rect_barrier
        
        # Higher-order circle barrier
        ho_circle_barrier = Barrier().assign(circle_func, rel_deg=3)
        ho_circle_barrier.assign_dynamics(self.dynamics)
        barriers['ho_circle'] = ho_circle_barrier
        
        return barriers
    
    def generate_test_states(self, batch_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Generate test state tensors of various batch sizes."""
        test_states = {}
        
        for batch_size in batch_sizes:
            # Random states for unicycle dynamics (x, y, v, theta)
            states = torch.randn(batch_size, 4, dtype=torch.float64)
            test_states[f'batch_{batch_size}'] = states
            
        return test_states
    
    def benchmark_barrier_evaluation(
        self, 
        batch_sizes: List[int] = [1, 10, 50, 100, 500],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark basic barrier function evaluation.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        barriers = self.setup_test_barriers()
        test_states = self.generate_test_states(batch_sizes)
        
        results = {}
        
        for barrier_name, barrier in barriers.items():
            results[barrier_name] = {}
            
            for state_name, states in test_states.items():
                # Benchmark barrier evaluation
                timing_result = benchmark_function(
                    barrier.barrier,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Memory profiling
                with self.profiler.profile_context():
                    _ = barrier.barrier(states)
                memory_result = self.profiler.last_results
                
                results[barrier_name][state_name] = {
                    'timing': timing_result,
                    'memory': memory_result,
                    'states_shape': states.shape
                }
        
        return results
    
    def benchmark_hocbf_evaluation(
        self,
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark higher-order CBF evaluation.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        barriers = self.setup_test_barriers()
        test_states = self.generate_test_states(batch_sizes)
        
        results = {}
        
        for barrier_name, barrier in barriers.items():
            results[barrier_name] = {}
            
            for state_name, states in test_states.items():
                # Benchmark HOCBF evaluation
                timing_result = benchmark_function(
                    barrier.hocbf,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Memory profiling
                with self.profiler.profile_context():
                    _ = barrier.hocbf(states)
                memory_result = self.profiler.last_results
                
                results[barrier_name][state_name] = {
                    'timing': timing_result,
                    'memory': memory_result,
                    'relative_degree': barrier.rel_deg,
                    'states_shape': states.shape
                }
        
        return results
    
    def benchmark_lie_derivatives(
        self,
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark Lie derivative computations.
        
        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of benchmark runs
            
        Returns:
            Dictionary with benchmark results
        """
        barriers = self.setup_test_barriers()
        test_states = self.generate_test_states(batch_sizes)
        
        results = {}
        
        for barrier_name, barrier in barriers.items():
            results[barrier_name] = {}
            
            for state_name, states in test_states.items():
                # Benchmark Lf_hocbf
                timing_lf = benchmark_function(
                    barrier.Lf_hocbf,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Benchmark Lg_hocbf  
                timing_lg = benchmark_function(
                    barrier.Lg_hocbf,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Benchmark combined computation
                timing_combined = benchmark_function(
                    barrier.get_hocbf_and_lie_derivs,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Memory profiling for combined computation
                with self.profiler.profile_context():
                    _ = barrier.get_hocbf_and_lie_derivs(states)
                memory_result = self.profiler.last_results
                
                results[barrier_name][state_name] = {
                    'timing_Lf': timing_lf,
                    'timing_Lg': timing_lg,
                    'timing_combined': timing_combined,
                    'memory': memory_result,
                    'relative_degree': barrier.rel_deg,
                    'states_shape': states.shape
                }
        
        return results
    
    def run_full_benchmark_suite(
        self,
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """Run complete benchmark suite for barrier functions."""
        return {
            'barrier_evaluation': self.benchmark_barrier_evaluation(batch_sizes, num_runs),
            'hocbf_evaluation': self.benchmark_hocbf_evaluation(batch_sizes, num_runs),
            'lie_derivatives': self.benchmark_lie_derivatives(batch_sizes, num_runs)
        }