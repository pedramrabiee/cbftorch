"""
Benchmarks for higher-order Control Barrier Function (HOCBF) generation and evaluation.
"""

import torch
from typing import List, Dict, Any
from ..utils.timing import benchmark_function
from ..utils.profiling import MemoryProfiler
from cbftorch.barriers.barrier import Barrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional, make_linear_alpha_function_form_list_of_coef


class HOCBFBenchmarks:
    """Benchmarks for higher-order CBF operations."""
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        self.dynamics = UnicycleDynamics()
        
    def setup_test_barriers_with_varying_rel_deg(self, max_rel_deg: int = 5) -> Dict[str, Barrier]:
        """Setup barriers with different relative degrees."""
        barriers = {}
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        for rel_deg in range(1, max_rel_deg + 1):
            # Create alphas for higher-order barriers
            alphas = make_linear_alpha_function_form_list_of_coef([1.0] * (rel_deg - 1))
            
            barrier = Barrier().assign(circle_func, rel_deg=rel_deg, alphas=alphas)
            barrier.assign_dynamics(self.dynamics)
            barriers[f'rel_deg_{rel_deg}'] = barrier
            
        return barriers
    
    def benchmark_hocbf_generation(
        self,
        max_rel_deg: int = 5,
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark the generation of higher-order CBF series.
        
        This measures the overhead of the _make_hocbf_series method.
        """
        results = {}
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        for rel_deg in range(1, max_rel_deg + 1):
            alphas = make_linear_alpha_function_form_list_of_coef([1.0] * (rel_deg - 1))
            
            def create_barrier_with_rel_deg():
                barrier = Barrier().assign(circle_func, rel_deg=rel_deg, alphas=alphas)
                barrier.assign_dynamics(self.dynamics)
                return barrier
            
            # Benchmark barrier creation (includes HOCBF generation)
            timing_result = benchmark_function(
                create_barrier_with_rel_deg,
                num_runs=num_runs
            )
            
            # Memory profiling
            with self.profiler.profile_context():
                barrier = create_barrier_with_rel_deg()
            memory_result = self.profiler.last_results
            
            results[f'rel_deg_{rel_deg}'] = {
                'timing': timing_result,
                'memory': memory_result,
                'relative_degree': rel_deg,
                'num_barriers_generated': len(barrier.barriers)
            }
        
        return results
    
    def benchmark_hocbf_evaluation_scaling(
        self,
        max_rel_deg: int = 5,
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark how HOCBF evaluation scales with relative degree and batch size.
        """
        barriers = self.setup_test_barriers_with_varying_rel_deg(max_rel_deg)
        results = {}
        
        for barrier_name, barrier in barriers.items():
            results[barrier_name] = {}
            
            for batch_size in batch_sizes:
                states = torch.randn(batch_size, self.dynamics.state_dim, dtype=torch.float64)
                
                # Benchmark HOCBF evaluation
                timing_hocbf = benchmark_function(
                    barrier.hocbf,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Benchmark combined HOCBF and Lie derivatives
                timing_combined = benchmark_function(
                    barrier.get_hocbf_and_lie_derivs,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Memory profiling
                with self.profiler.profile_context():
                    _ = barrier.get_hocbf_and_lie_derivs(states)
                memory_result = self.profiler.last_results
                
                results[barrier_name][f'batch_{batch_size}'] = {
                    'timing_hocbf': timing_hocbf,
                    'timing_combined': timing_combined,
                    'memory': memory_result,
                    'relative_degree': barrier.rel_deg,
                    'batch_size': batch_size,
                    'states_shape': states.shape
                }
        
        return results
    
    def benchmark_lambda_chain_overhead(
        self,
        max_rel_deg: int = 5,
        batch_size: int = 50,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark the overhead of lambda function chains in HOCBF evaluation.
        
        This measures the cost of the current implementation's approach
        of creating nested lambda functions.
        """
        results = {}
        states = torch.randn(batch_size, self.dynamics.state_dim, dtype=torch.float64)
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        for rel_deg in range(1, max_rel_deg + 1):
            alphas = make_linear_alpha_function_form_list_of_coef([1.0] * (rel_deg - 1))
            barrier = Barrier().assign(circle_func, rel_deg=rel_deg, alphas=alphas)
            barrier.assign_dynamics(self.dynamics)
            
            # Benchmark each barrier in the series
            barrier_timings = {}
            for i, barrier_func in enumerate(barrier.barriers):
                timing_result = benchmark_function(
                    lambda x, func=barrier_func: func(x),
                    args=(states,),
                    num_runs=num_runs
                )
                barrier_timings[f'barrier_level_{i}'] = timing_result
            
            # Benchmark direct barrier function (level 0)
            timing_direct = benchmark_function(
                circle_func,
                args=(states,),
                num_runs=num_runs
            )
            
            # Memory profiling for the highest order barrier
            with self.profiler.profile_context():
                _ = barrier.hocbf(states)
            memory_result = self.profiler.last_results
            
            results[f'rel_deg_{rel_deg}'] = {
                'barrier_timings': barrier_timings,
                'timing_direct_barrier': timing_direct,
                'memory': memory_result,
                'relative_degree': rel_deg,
                'overhead_factor': (
                    barrier_timings[f'barrier_level_{rel_deg-1}'].mean_time / 
                    timing_direct.mean_time if rel_deg > 0 else 1.0
                )
            }
        
        return results
    
    def benchmark_raise_rel_deg_operation(
        self,
        initial_rel_deg: int = 2,
        raise_by_steps: List[int] = [1, 2, 3],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark the raise_rel_deg operation.
        
        This operation extends an existing HOCBF to higher relative degree.
        """
        results = {}
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        for raise_by in raise_by_steps:
            # Create initial barrier
            initial_alphas = make_linear_alpha_function_form_list_of_coef([1.0] * (initial_rel_deg - 1))
            barrier = Barrier().assign(circle_func, rel_deg=initial_rel_deg, alphas=initial_alphas)
            barrier.assign_dynamics(self.dynamics)
            
            # Create test states
            states = torch.randn(10, self.dynamics.state_dim, dtype=torch.float64)
            
            # Benchmark the raise_rel_deg operation
            raise_alphas = make_linear_alpha_function_form_list_of_coef([1.0] * raise_by)
            
            def raise_rel_deg_operation():
                # Note: This modifies the barrier in place
                barrier_copy = Barrier().assign(circle_func, rel_deg=initial_rel_deg, alphas=initial_alphas)
                barrier_copy.assign_dynamics(self.dynamics)
                barrier_copy.raise_rel_deg(states, raise_rel_deg_by=raise_by, alphas=raise_alphas)
                return barrier_copy
            
            timing_result = benchmark_function(
                raise_rel_deg_operation,
                num_runs=num_runs
            )
            
            # Memory profiling
            with self.profiler.profile_context():
                new_barrier = raise_rel_deg_operation()
            memory_result = self.profiler.last_results
            
            results[f'raise_by_{raise_by}'] = {
                'timing': timing_result,
                'memory': memory_result,
                'initial_rel_deg': initial_rel_deg,
                'raise_by': raise_by,
                'final_rel_deg': initial_rel_deg + raise_by
            }
        
        return results
    
    def run_full_benchmark_suite(
        self,
        max_rel_deg: int = 5,
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """Run complete HOCBF benchmark suite."""
        return {
            'hocbf_generation': self.benchmark_hocbf_generation(max_rel_deg, num_runs),
            'evaluation_scaling': self.benchmark_hocbf_evaluation_scaling(max_rel_deg, batch_sizes, num_runs),
            'lambda_chain_overhead': self.benchmark_lambda_chain_overhead(max_rel_deg, num_runs=num_runs),
            'raise_rel_deg': self.benchmark_raise_rel_deg_operation(num_runs=num_runs)
        }