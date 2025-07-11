#!/usr/bin/env python3
"""
Simple script to create performance baselines for the current CBFtorch implementation.
This runs a subset of benchmarks to establish baseline measurements.
"""

import os
import sys
import time
import torch
from pathlib import Path

# Add cbftorch to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cbftorch.barriers.barrier import Barrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional
from benchmarks.utils.timing import benchmark_function
from benchmarks.utils.profiling import MemoryProfiler


def create_simple_baseline():
    """Create a simple baseline measurement for the current implementation."""
    print("Creating CBFtorch Performance Baseline")
    print("="*50)
    
    # Setup
    dynamics = UnicycleDynamics()
    circle_func = make_circle_barrier_functional(
        center=torch.tensor([0.0, 0.0]), 
        radius=2.0
    )
    
    # Create barriers with different relative degrees
    barriers = {}
    for rel_deg in [1, 2, 3]:
        barrier = Barrier().assign(circle_func, rel_deg=rel_deg)
        barrier.assign_dynamics(dynamics)
        barriers[f'rel_deg_{rel_deg}'] = barrier
    
    # Test configurations
    batch_sizes = [1, 10, 50, 100]
    num_runs = 20
    
    profiler = MemoryProfiler()
    baseline_results = {}
    
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Number of runs per test: {num_runs}")
    print()
    
    # Barrier evaluation tests
    print("Testing barrier evaluations...")
    baseline_results['barrier_evaluation'] = {}
    
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        for barrier_name, barrier in barriers.items():
            test_name = f"{barrier_name}_batch_{batch_size}"
            
            # Time barrier evaluation
            timing_result = benchmark_function(
                barrier.barrier,
                args=(states,),
                num_runs=num_runs
            )
            
            # Memory profiling
            with profiler.profile_context():
                _ = barrier.barrier(states)
            memory_result = profiler.last_results
            
            baseline_results['barrier_evaluation'][test_name] = {
                'mean_time': timing_result.mean_time,
                'std_time': timing_result.std_time,
                'memory_mb': memory_result.memory_delta['rss_mb'],
                'batch_size': batch_size,
                'rel_deg': barrier.rel_deg
            }
            
            print(f"  {test_name}: {timing_result.mean_time:.6f}s ± {timing_result.std_time:.6f}s")
    
    # HOCBF evaluation tests
    print("\nTesting HOCBF evaluations...")
    baseline_results['hocbf_evaluation'] = {}
    
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        for barrier_name, barrier in barriers.items():
            test_name = f"{barrier_name}_batch_{batch_size}"
            
            # Time HOCBF evaluation
            timing_result = benchmark_function(
                barrier.hocbf,
                args=(states,),
                num_runs=num_runs
            )
            
            # Memory profiling
            with profiler.profile_context():
                _ = barrier.hocbf(states)
            memory_result = profiler.last_results
            
            baseline_results['hocbf_evaluation'][test_name] = {
                'mean_time': timing_result.mean_time,
                'std_time': timing_result.std_time,
                'memory_mb': memory_result.memory_delta['rss_mb'],
                'batch_size': batch_size,
                'rel_deg': barrier.rel_deg
            }
            
            print(f"  {test_name}: {timing_result.mean_time:.6f}s ± {timing_result.std_time:.6f}s")
    
    # Lie derivative tests
    print("\nTesting Lie derivative computations...")
    baseline_results['lie_derivatives'] = {}
    
    for batch_size in [1, 10, 50]:  # Fewer tests for Lie derivatives
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        for barrier_name, barrier in barriers.items():
            if barrier.rel_deg <= 2:  # Only test lower relative degrees
                test_name = f"{barrier_name}_batch_{batch_size}"
                
                # Time Lf computation
                timing_lf = benchmark_function(
                    barrier.Lf_hocbf,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Time Lg computation
                timing_lg = benchmark_function(
                    barrier.Lg_hocbf,
                    args=(states,),
                    num_runs=num_runs
                )
                
                # Time combined computation
                timing_combined = benchmark_function(
                    barrier.get_hocbf_and_lie_derivs,
                    args=(states,),
                    num_runs=num_runs
                )
                
                baseline_results['lie_derivatives'][test_name] = {
                    'lf_time': timing_lf.mean_time,
                    'lg_time': timing_lg.mean_time,
                    'combined_time': timing_combined.mean_time,
                    'batch_size': batch_size,
                    'rel_deg': barrier.rel_deg
                }
                
                print(f"  {test_name}: Lf={timing_lf.mean_time:.6f}s, Lg={timing_lg.mean_time:.6f}s, Combined={timing_combined.mean_time:.6f}s")
    
    # Save baseline
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    baseline_file = f"benchmarks/baseline_original_{timestamp}.json"
    
    os.makedirs("benchmarks", exist_ok=True)
    with open(baseline_file, 'w') as f:
        json.dump(baseline_results, f, indent=2)
    
    print(f"\nBaseline saved to: {baseline_file}")
    print("="*50)
    
    # Print summary
    print("BASELINE SUMMARY:")
    
    # Calculate averages
    barrier_times = [result['mean_time'] for result in baseline_results['barrier_evaluation'].values()]
    hocbf_times = [result['mean_time'] for result in baseline_results['hocbf_evaluation'].values()]
    lie_times = [result['combined_time'] for result in baseline_results['lie_derivatives'].values()]
    
    print(f"Average barrier evaluation time: {sum(barrier_times)/len(barrier_times):.6f}s")
    print(f"Average HOCBF evaluation time: {sum(hocbf_times)/len(hocbf_times):.6f}s")
    print(f"Average Lie derivative time: {sum(lie_times)/len(lie_times):.6f}s")
    
    return baseline_results


if __name__ == "__main__":
    create_simple_baseline()