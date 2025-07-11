#!/usr/bin/env python3
"""
Simple, reliable benchmark script that avoids complex tensor operations.
This creates comprehensive baseline measurements with proper result saving.
"""

import os
import sys
import json
import time
import torch
import psutil
from pathlib import Path

# Add cbftorch to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cbftorch.barriers.barrier import Barrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional
from benchmarks.utils.timing import benchmark_function


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'cpu_percent': process.cpu_percent()
    }


def run_comprehensive_benchmark():
    """Run comprehensive benchmark with proper result saving."""
    print("CBFtorch Comprehensive Benchmark")
    print("="*50)
    
    # Create results directory
    results_dir = "benchmarks/results"
    os.makedirs(results_dir, exist_ok=True)
    
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
    batch_sizes = [1, 5, 10, 25, 50, 100]
    num_runs = 50
    warmup_runs = 10
    
    results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'batch_sizes': batch_sizes,
            'num_runs': num_runs,
            'warmup_runs': warmup_runs,
            'torch_version': torch.__version__,
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        },
        'barrier_evaluation': {},
        'hocbf_evaluation': {},
        'lie_derivatives': {},
        'memory_analysis': {}
    }
    
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Number of runs per test: {num_runs}")
    print(f"Device: {results['metadata']['device']}")
    print()
    
    # 1. Barrier evaluation benchmarks
    print("1. Testing barrier evaluations...")
    memory_before = get_memory_usage()
    
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        for barrier_name, barrier in barriers.items():
            test_name = f"{barrier_name}_batch_{batch_size}"
            
            # Memory before test
            mem_before = get_memory_usage()
            
            # Time barrier evaluation
            timing_result = benchmark_function(
                barrier.barrier,
                args=(states,),
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
            
            # Memory after test
            mem_after = get_memory_usage()
            memory_delta = mem_after['rss_mb'] - mem_before['rss_mb']
            
            results['barrier_evaluation'][test_name] = {
                'mean_time': timing_result.mean_time,
                'std_time': timing_result.std_time,
                'min_time': timing_result.min_time,
                'max_time': timing_result.max_time,
                'total_time': timing_result.total_time,
                'num_runs': timing_result.num_runs,
                'memory_delta_mb': memory_delta,
                'batch_size': batch_size,
                'rel_deg': barrier.rel_deg,
                'time_per_sample': timing_result.mean_time / batch_size
            }
            
            print(f"  {test_name}: {timing_result.mean_time:.6f}s ± {timing_result.std_time:.6f}s (Δmem: {memory_delta:.2f}MB)")
    
    # 2. HOCBF evaluation benchmarks
    print("\n2. Testing HOCBF evaluations...")
    
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        for barrier_name, barrier in barriers.items():
            test_name = f"{barrier_name}_batch_{batch_size}"
            
            # Memory before test
            mem_before = get_memory_usage()
            
            # Time HOCBF evaluation
            timing_result = benchmark_function(
                barrier.hocbf,
                args=(states,),
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
            
            # Memory after test
            mem_after = get_memory_usage()
            memory_delta = mem_after['rss_mb'] - mem_before['rss_mb']
            
            results['hocbf_evaluation'][test_name] = {
                'mean_time': timing_result.mean_time,
                'std_time': timing_result.std_time,
                'min_time': timing_result.min_time,
                'max_time': timing_result.max_time,
                'total_time': timing_result.total_time,
                'num_runs': timing_result.num_runs,
                'memory_delta_mb': memory_delta,
                'batch_size': batch_size,
                'rel_deg': barrier.rel_deg,
                'time_per_sample': timing_result.mean_time / batch_size,
                'overhead_vs_barrier': 0.0  # Will calculate below
            }
            
            # Calculate overhead vs basic barrier
            barrier_time = results['barrier_evaluation'][test_name]['mean_time']
            hocbf_time = timing_result.mean_time
            overhead = (hocbf_time - barrier_time) / barrier_time * 100
            results['hocbf_evaluation'][test_name]['overhead_vs_barrier'] = overhead
            
            print(f"  {test_name}: {timing_result.mean_time:.6f}s ± {timing_result.std_time:.6f}s (Δmem: {memory_delta:.2f}MB, overhead: {overhead:.1f}%)")
    
    # 3. Lie derivative benchmarks (simplified to avoid tensor iteration issues)
    print("\n3. Testing Lie derivative computations...")
    
    for batch_size in [1, 10, 50]:  # Fewer tests for complex operations
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        for barrier_name, barrier in barriers.items():
            if barrier.rel_deg <= 2:  # Only test lower relative degrees to avoid complexity
                test_name = f"{barrier_name}_batch_{batch_size}"
                
                # Memory before test
                mem_before = get_memory_usage()
                
                # Time individual Lie derivatives
                try:
                    timing_lf = benchmark_function(
                        barrier.Lf_hocbf,
                        args=(states,),
                        num_runs=num_runs//2,  # Fewer runs for expensive operations
                        warmup_runs=warmup_runs//2
                    )
                    
                    timing_lg = benchmark_function(
                        barrier.Lg_hocbf,
                        args=(states,),
                        num_runs=num_runs//2,
                        warmup_runs=warmup_runs//2
                    )
                    
                    # Memory after test
                    mem_after = get_memory_usage()
                    memory_delta = mem_after['rss_mb'] - mem_before['rss_mb']
                    
                    results['lie_derivatives'][test_name] = {
                        'lf_mean_time': timing_lf.mean_time,
                        'lf_std_time': timing_lf.std_time,
                        'lg_mean_time': timing_lg.mean_time,
                        'lg_std_time': timing_lg.std_time,
                        'total_time': timing_lf.mean_time + timing_lg.mean_time,
                        'memory_delta_mb': memory_delta,
                        'batch_size': batch_size,
                        'rel_deg': barrier.rel_deg
                    }
                    
                    print(f"  {test_name}: Lf={timing_lf.mean_time:.6f}s, Lg={timing_lg.mean_time:.6f}s (Δmem: {memory_delta:.2f}MB)")
                
                except Exception as e:
                    print(f"  {test_name}: FAILED - {e}")
                    results['lie_derivatives'][test_name] = {
                        'error': str(e),
                        'batch_size': batch_size,
                        'rel_deg': barrier.rel_deg
                    }
    
    # 4. Memory analysis summary
    memory_after = get_memory_usage()
    results['memory_analysis'] = {
        'initial_memory_mb': memory_before['rss_mb'],
        'final_memory_mb': memory_after['rss_mb'],
        'total_memory_increase_mb': memory_after['rss_mb'] - memory_before['rss_mb'],
        'peak_cpu_percent': memory_after['cpu_percent']
    }
    
    # Save results with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_filename = f"performance_results_{timestamp}.json"
    detailed_filepath = os.path.join(results_dir, detailed_filename)
    
    with open(detailed_filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {detailed_filepath}")
    
    # Save summary results
    summary = create_summary(results)
    summary_filename = f"performance_summary_{timestamp}.json"
    summary_filepath = os.path.join(results_dir, summary_filename)
    
    with open(summary_filepath, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary results saved to: {summary_filepath}")
    
    # Print final summary
    print_summary(summary)
    
    return results


def create_summary(results):
    """Create a summary of benchmark results."""
    summary = {
        'timestamp': results['metadata']['timestamp'],
        'device': results['metadata']['device'],
        'overview': {},
        'scaling_analysis': {},
        'performance_hotspots': []
    }
    
    # Calculate overview statistics
    barrier_times = [v['mean_time'] for v in results['barrier_evaluation'].values()]
    hocbf_times = [v['mean_time'] for v in results['hocbf_evaluation'].values()]
    
    summary['overview'] = {
        'total_tests': len(barrier_times) + len(hocbf_times),
        'avg_barrier_time': sum(barrier_times) / len(barrier_times) if barrier_times else 0,
        'avg_hocbf_time': sum(hocbf_times) / len(hocbf_times) if hocbf_times else 0,
        'max_execution_time': max(barrier_times + hocbf_times) if (barrier_times + hocbf_times) else 0,
        'total_memory_used_mb': results['memory_analysis']['total_memory_increase_mb']
    }
    
    # Scaling analysis
    batch_1_times = [v['mean_time'] for k, v in results['hocbf_evaluation'].items() if 'batch_1' in k]
    batch_100_times = [v['mean_time'] for k, v in results['hocbf_evaluation'].items() if 'batch_100' in k]
    
    if batch_1_times and batch_100_times:
        avg_scaling = (sum(batch_100_times) / len(batch_100_times)) / (sum(batch_1_times) / len(batch_1_times))
        summary['scaling_analysis']['batch_1_to_100_scaling'] = avg_scaling
        summary['scaling_analysis']['scaling_efficiency'] = 100 / avg_scaling  # Ideal would be 100x
    
    # Performance hotspots
    all_times = [(k, v['mean_time']) for k, v in results['hocbf_evaluation'].items()]
    all_times.sort(key=lambda x: x[1], reverse=True)
    summary['performance_hotspots'] = all_times[:10]  # Top 10 slowest
    
    return summary


def print_summary(summary):
    """Print a formatted summary."""
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    
    print(f"Device: {summary['device']}")
    print(f"Timestamp: {summary['timestamp']}")
    print()
    
    overview = summary['overview']
    print(f"Total tests: {overview['total_tests']}")
    print(f"Average barrier time: {overview['avg_barrier_time']:.6f}s")
    print(f"Average HOCBF time: {overview['avg_hocbf_time']:.6f}s")
    print(f"Maximum execution time: {overview['max_execution_time']:.6f}s")
    print(f"Total memory used: {overview['total_memory_used_mb']:.2f}MB")
    print()
    
    if 'batch_1_to_100_scaling' in summary['scaling_analysis']:
        scaling = summary['scaling_analysis']
        print(f"Batch scaling (1→100): {scaling['batch_1_to_100_scaling']:.1f}x")
        print(f"Scaling efficiency: {scaling['scaling_efficiency']:.1f}%")
        print()
    
    print("Top 5 slowest operations:")
    for i, (test_name, time_val) in enumerate(summary['performance_hotspots'][:5]):
        print(f"  {i+1}. {test_name}: {time_val:.6f}s")
    
    print("="*50)


if __name__ == "__main__":
    run_comprehensive_benchmark()