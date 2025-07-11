"""
Benchmarks for composite barrier functions.
"""

import torch
from typing import List, Dict, Any
from ..utils.timing import benchmark_function
from ..utils.profiling import MemoryProfiler
from cbftorch.barriers.barrier import Barrier
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier, NonSmoothCompositionBarrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional, make_norm_rectangular_barrier_functional
from box import Box as AD


class CompositeBenchmarks:
    """Benchmarks for composite barrier function operations."""
    
    def __init__(self):
        self.profiler = MemoryProfiler()
        self.dynamics = UnicycleDynamics()
        
    def setup_individual_barriers(self, num_barriers: int) -> List[Barrier]:
        """Setup individual barriers to be composed."""
        barriers = []
        
        # Create circle barriers at different positions
        for i in range(num_barriers):
            center = torch.tensor([2.0 * i - num_barriers, 2.0 * (i % 2)], dtype=torch.float64)
            circle_func = make_circle_barrier_functional(center=center, radius=1.5)
            
            barrier = Barrier().assign(circle_func, rel_deg=1)
            barrier.assign_dynamics(self.dynamics)
            barriers.append(barrier)
        
        return barriers
    
    def setup_composite_barriers(
        self, 
        num_individual_barriers: List[int],
        composition_rules: List[str] = ['intersection', 'union']
    ) -> Dict[str, Any]:
        """Setup composite barriers with varying numbers of individual barriers."""
        composite_barriers = {}
        
        cfg = AD(softmax_rho=20, softmin_rho=20)
        
        for num_barriers in num_individual_barriers:
            individual_barriers = self.setup_individual_barriers(num_barriers)
            
            for rule in composition_rules:
                # Soft composition
                soft_composite = SoftCompositionBarrier(cfg=cfg)
                soft_composite.assign_barriers_and_rule(
                    barriers=individual_barriers, 
                    rule=rule,
                    infer_dynamics=True
                )
                
                # Non-smooth composition
                nonsmooth_composite = NonSmoothCompositionBarrier()
                nonsmooth_composite.assign_barriers_and_rule(
                    barriers=individual_barriers,
                    rule=rule,
                    infer_dynamics=True
                )
                
                composite_barriers[f'soft_{rule}_{num_barriers}'] = soft_composite
                composite_barriers[f'nonsmooth_{rule}_{num_barriers}'] = nonsmooth_composite
        
        return composite_barriers
    
    def benchmark_composite_creation(
        self,
        num_individual_barriers: List[int] = [2, 5, 10, 20],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark the creation of composite barriers.
        """
        results = {}
        cfg = AD(softmax_rho=20, softmin_rho=20)
        
        for num_barriers in num_individual_barriers:
            individual_barriers = self.setup_individual_barriers(num_barriers)
            
            # Benchmark soft composition creation
            def create_soft_intersection():
                composite = SoftCompositionBarrier(cfg=cfg)
                composite.assign_barriers_and_rule(
                    barriers=individual_barriers,
                    rule='intersection',
                    infer_dynamics=True
                )
                return composite
            
            timing_soft = benchmark_function(
                create_soft_intersection,
                num_runs=num_runs
            )
            
            # Benchmark non-smooth composition creation
            def create_nonsmooth_intersection():
                composite = NonSmoothCompositionBarrier()
                composite.assign_barriers_and_rule(
                    barriers=individual_barriers,
                    rule='intersection',
                    infer_dynamics=True
                )
                return composite
            
            timing_nonsmooth = benchmark_function(
                create_nonsmooth_intersection,
                num_runs=num_runs
            )
            
            # Memory profiling
            with self.profiler.profile_context():
                _ = create_soft_intersection()
            memory_soft = self.profiler.last_results
            
            with self.profiler.profile_context():
                _ = create_nonsmooth_intersection()
            memory_nonsmooth = self.profiler.last_results
            
            results[f'num_barriers_{num_barriers}'] = {
                'timing_soft': timing_soft,
                'timing_nonsmooth': timing_nonsmooth,
                'memory_soft': memory_soft,
                'memory_nonsmooth': memory_nonsmooth,
                'num_individual_barriers': num_barriers
            }
        
        return results
    
    def benchmark_composite_evaluation(
        self,
        num_individual_barriers: List[int] = [2, 5, 10, 20],
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark evaluation of composite barriers.
        """
        composite_barriers = self.setup_composite_barriers(num_individual_barriers)
        results = {}
        
        for barrier_name, barrier in composite_barriers.items():
            results[barrier_name] = {}
            
            for batch_size in batch_sizes:
                states = torch.randn(batch_size, self.dynamics.state_dim, dtype=torch.float64)
                
                # Benchmark barrier evaluation
                timing_barrier = benchmark_function(
                    barrier.barrier,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Benchmark HOCBF evaluation
                timing_hocbf = benchmark_function(
                    barrier.hocbf,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Benchmark combined computation
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
                    'timing_barrier': timing_barrier,
                    'timing_hocbf': timing_hocbf,
                    'timing_combined': timing_combined,
                    'memory': memory_result,
                    'batch_size': batch_size,
                    'num_barriers': barrier.num_barriers
                }
        
        return results
    
    def benchmark_composition_rules(
        self,
        num_barriers: int = 10,
        batch_size: int = 50,
        num_runs: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark different composition rules (intersection vs union, soft vs non-smooth).
        """
        individual_barriers = self.setup_individual_barriers(num_barriers)
        states = torch.randn(batch_size, self.dynamics.state_dim, dtype=torch.float64)
        
        cfg = AD(softmax_rho=20, softmin_rho=20)
        results = {}
        
        compositions = {
            'soft_intersection': (SoftCompositionBarrier(cfg=cfg), 'intersection'),
            'soft_union': (SoftCompositionBarrier(cfg=cfg), 'union'),
            'nonsmooth_intersection': (NonSmoothCompositionBarrier(), 'intersection'),
            'nonsmooth_union': (NonSmoothCompositionBarrier(), 'union')
        }
        
        for comp_name, (composite_class, rule) in compositions.items():
            composite_class.assign_barriers_and_rule(
                barriers=individual_barriers,
                rule=rule,
                infer_dynamics=True
            )
            
            # Benchmark the composition rule application
            # Get individual barrier values first
            individual_values = torch.hstack([
                barrier.hocbf(states) for barrier in individual_barriers
            ])
            
            timing_composition = benchmark_function(
                composite_class.compose(rule),
                args=(individual_values,),
                num_runs=num_runs
            )
            
            # Benchmark full HOCBF evaluation
            timing_full = benchmark_function(
                composite_class.hocbf,
                args=(states,),
                num_runs=num_runs
            )
            
            # Memory profiling
            with self.profiler.profile_context():
                _ = composite_class.hocbf(states)
            memory_result = self.profiler.last_results
            
            results[comp_name] = {
                'timing_composition_rule': timing_composition,
                'timing_full_hocbf': timing_full,
                'memory': memory_result,
                'rule': rule,
                'num_individual_barriers': num_barriers
            }
        
        return results
    
    def benchmark_scaling_with_barriers(
        self,
        max_num_barriers: int = 50,
        step_size: int = 5,
        batch_size: int = 50,
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark how performance scales with the number of individual barriers.
        """
        results = {}
        states = torch.randn(batch_size, self.dynamics.state_dim, dtype=torch.float64)
        cfg = AD(softmax_rho=20, softmin_rho=20)
        
        barrier_counts = list(range(2, max_num_barriers + 1, step_size))
        
        for num_barriers in barrier_counts:
            individual_barriers = self.setup_individual_barriers(num_barriers)
            
            # Test soft intersection (most common case)
            composite = SoftCompositionBarrier(cfg=cfg)
            composite.assign_barriers_and_rule(
                barriers=individual_barriers,
                rule='intersection',
                infer_dynamics=True
            )
            
            # Benchmark evaluation
            timing_hocbf = benchmark_function(
                composite.hocbf,
                args=(states,),
                num_runs=num_runs
            )
            
            timing_combined = benchmark_function(
                composite.get_hocbf_and_lie_derivs,
                args=(states,),
                num_runs=num_runs
            )
            
            # Memory profiling
            with self.profiler.profile_context():
                _ = composite.get_hocbf_and_lie_derivs(states)
            memory_result = self.profiler.last_results
            
            results[f'num_barriers_{num_barriers}'] = {
                'timing_hocbf': timing_hocbf,
                'timing_combined': timing_combined,
                'memory': memory_result,
                'num_barriers': num_barriers,
                'per_barrier_time': timing_hocbf.mean_time / num_barriers
            }
        
        return results
    
    def run_full_benchmark_suite(
        self,
        max_num_barriers: int = 20,
        batch_sizes: List[int] = [1, 10, 50, 100],
        num_runs: int = 50
    ) -> Dict[str, Any]:
        """Run complete composite barrier benchmark suite."""
        barrier_counts = [2, 5, 10, max_num_barriers]
        
        return {
            'creation': self.benchmark_composite_creation(barrier_counts, num_runs),
            'evaluation': self.benchmark_composite_evaluation(barrier_counts, batch_sizes, num_runs),
            'composition_rules': self.benchmark_composition_rules(num_runs=num_runs),
            'scaling': self.benchmark_scaling_with_barriers(max_num_barriers, num_runs=num_runs)
        }