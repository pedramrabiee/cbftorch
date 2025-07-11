"""
Focused benchmarks for Lie derivative computations.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Callable
from ..utils.timing import benchmark_function, TimingResult
from ..utils.profiling import MemoryProfiler
from cbftorch.utils.utils import lie_deriv, get_func_deriv, lie_deriv_from_values
from cbftorch.dynamics import UnicycleDynamics, DIDynamics, SIDynamics
from cbftorch.barriers.barrier import Barrier
from cbftorch.utils.utils import make_circle_barrier_functional


class LieDerivativeBenchmarks:
    """Focused benchmarks for Lie derivative computation efficiency."""
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        
    def setup_test_functions_and_dynamics(self) -> Dict[str, Any]:
        """Setup test functions and dynamics systems."""
        test_setup = {}
        
        # Simple quadratic function
        def quadratic_func(x):
            return torch.sum(x**2, dim=-1, keepdim=True)
        
        # Circle barrier function
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # More complex polynomial function
        def poly_func(x):
            return (x[..., 0]**3 + x[..., 1]**2 + x[..., 2] + 1).unsqueeze(-1)
        
        test_setup['functions'] = {
            'quadratic': quadratic_func,
            'circle': circle_func,
            'polynomial': poly_func
        }
        
        # Different dynamics systems
        test_setup['dynamics'] = {
            'unicycle': UnicycleDynamics(),
            'double_integrator': DIDynamics(),
            'single_integrator': SIDynamics()
        }
        
        return test_setup
    
    def generate_test_states(self, dynamics, batch_sizes: List[int]) -> Dict[str, torch.Tensor]:
        """Generate test states for given dynamics."""
        test_states = {}
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, dynamics.state_dim, dtype=torch.float64)
            test_states[f'batch_{batch_size}'] = states
            
        return test_states
    
    def benchmark_lie_deriv_components(
        self,
        batch_sizes: List[int] = [1, 10, 50, 100, 500],
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark individual components of Lie derivative computation.
        
        This helps identify which part of the computation is the bottleneck.
        """
        test_setup = self.setup_test_functions_and_dynamics()
        results = {}
        
        for func_name, func in test_setup['functions'].items():
            results[func_name] = {}
            
            for dyn_name, dynamics in test_setup['dynamics'].items():
                results[func_name][dyn_name] = {}
                test_states = self.generate_test_states(dynamics, batch_sizes)
                
                for state_name, states in test_states.items():
                    # Benchmark gradient computation
                    timing_grad = benchmark_function(
                        get_func_deriv,
                        args=(states, func),
                        num_runs=num_runs
                    )
                    
                    # Benchmark field evaluation
                    timing_f_field = benchmark_function(
                        dynamics.f,
                        args=(states,),
                        num_runs=num_runs
                    )
                    
                    timing_g_field = benchmark_function(
                        dynamics.g,
                        args=(states,),
                        num_runs=num_runs
                    )
                    
                    # Benchmark lie_deriv_from_values (the einsum operations)
                    func_deriv = get_func_deriv(states, func)
                    f_val = dynamics.f(states)
                    g_val = dynamics.g(states)
                    
                    timing_lie_from_vals_f = benchmark_function(
                        lie_deriv_from_values,
                        args=(func_deriv, f_val),
                        num_runs=num_runs
                    )
                    
                    timing_lie_from_vals_g = benchmark_function(
                        lie_deriv_from_values,
                        args=(func_deriv, g_val),
                        num_runs=num_runs
                    )
                    
                    # Benchmark full lie_deriv computation
                    timing_full_f = benchmark_function(
                        lie_deriv,
                        args=(states, func, dynamics.f),
                        num_runs=num_runs
                    )
                    
                    timing_full_g = benchmark_function(
                        lie_deriv,
                        args=(states, func, dynamics.g),
                        num_runs=num_runs
                    )
                    
                    # Memory profiling for full computation
                    with self.profiler.profile_context():
                        _ = lie_deriv(states, func, dynamics.f)
                    memory_f = self.profiler.last_results
                    
                    with self.profiler.profile_context():
                        _ = lie_deriv(states, func, dynamics.g)
                    memory_g = self.profiler.last_results
                    
                    results[func_name][dyn_name][state_name] = {
                        'timing_gradient': timing_grad,
                        'timing_f_field': timing_f_field,
                        'timing_g_field': timing_g_field,
                        'timing_lie_from_vals_f': timing_lie_from_vals_f,
                        'timing_lie_from_vals_g': timing_lie_from_vals_g,
                        'timing_full_f': timing_full_f,
                        'timing_full_g': timing_full_g,
                        'memory_f': memory_f,
                        'memory_g': memory_g,
                        'states_shape': states.shape,
                        'state_dim': dynamics.state_dim,
                        'action_dim': dynamics.action_dim
                    }
        
        return results
    
    def benchmark_gradient_computation_scaling(
        self,
        output_dims: List[int] = [1, 5, 10, 20],
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark how gradient computation scales with output dimension.
        
        The current implementation computes gradients for each output separately,
        which may be inefficient for multi-output functions.
        """
        results = {}
        dynamics = UnicycleDynamics()
        
        for output_dim in output_dims:
            results[f'output_dim_{output_dim}'] = {}
            
            # Create function with specified output dimension
            def multi_output_func(x, out_dim=output_dim):
                # Simple function that outputs out_dim values
                base = torch.sum(x**2, dim=-1, keepdim=True)
                return base.repeat(1, out_dim)
            
            for batch_size in batch_sizes:
                states = torch.randn(batch_size, dynamics.state_dim, dtype=torch.float64)
                
                # Benchmark current gradient computation method
                timing_current = benchmark_function(
                    get_func_deriv,
                    args=(states, lambda x: multi_output_func(x, output_dim)),
                    num_runs=num_runs
                )
                
                # Memory profiling
                with self.profiler.profile_context():
                    _ = get_func_deriv(states, lambda x: multi_output_func(x, output_dim))
                memory_result = self.profiler.last_results
                
                results[f'output_dim_{output_dim}'][f'batch_{batch_size}'] = {
                    'timing': timing_current,
                    'memory': memory_result,
                    'output_dim': output_dim,
                    'batch_size': batch_size
                }
        
        return results
    
    def benchmark_repeated_computations(
        self,
        num_repeated_calls: List[int] = [1, 5, 10, 20],
        batch_size: int = 50,
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark repeated Lie derivative computations to identify caching opportunities.
        
        In practice, the same function may be differentiated multiple times
        with the same states but different vector fields.
        """
        results = {}
        dynamics = UnicycleDynamics()
        states = torch.randn(batch_size, dynamics.state_dim, dtype=torch.float64)
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        for num_calls in num_repeated_calls:
            # Benchmark repeated independent calls (current behavior)
            def repeated_independent_calls():
                for _ in range(num_calls):
                    _ = lie_deriv(states, circle_func, dynamics.f)
                    _ = lie_deriv(states, circle_func, dynamics.g)
            
            timing_independent = benchmark_function(
                repeated_independent_calls,
                num_runs=num_runs
            )
            
            # Benchmark with potential for gradient reuse
            def repeated_with_gradient_reuse():
                func_deriv = get_func_deriv(states, circle_func)
                for _ in range(num_calls):
                    _ = lie_deriv_from_values(func_deriv, dynamics.f(states))
                    _ = lie_deriv_from_values(func_deriv, dynamics.g(states))
            
            timing_reuse = benchmark_function(
                repeated_with_gradient_reuse,
                num_runs=num_runs
            )
            
            # Memory profiling
            with self.profiler.profile_context():
                repeated_independent_calls()
            memory_independent = self.profiler.last_results
            
            with self.profiler.profile_context():
                repeated_with_gradient_reuse()
            memory_reuse = self.profiler.last_results
            
            results[f'calls_{num_calls}'] = {
                'timing_independent': timing_independent,
                'timing_reuse': timing_reuse,
                'memory_independent': memory_independent,
                'memory_reuse': memory_reuse,
                'speedup_factor': timing_independent.mean_time / timing_reuse.mean_time,
                'num_calls': num_calls
            }
        
        return results
    
    def run_full_benchmark_suite(
        self,
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """Run complete Lie derivative benchmark suite."""
        return {
            'component_breakdown': self.benchmark_lie_deriv_components(batch_sizes, num_runs),
            'gradient_scaling': self.benchmark_gradient_computation_scaling(num_runs=num_runs),
            'repeated_computations': self.benchmark_repeated_computations(num_runs=num_runs)
        }