#!/usr/bin/env python3
"""
CBFtorch Main Benchmark Script

This is the ONLY benchmark script you need to run.
It captures all optimization targets and saves comprehensive results.

Usage:
    python benchmarks/benchmark.py              # Run full benchmark
    python benchmarks/benchmark.py --quick      # Quick test (fewer runs)
    python benchmarks/benchmark.py --baseline   # Create baseline only
"""

import os
import sys
import json
import time
import torch
import psutil
import argparse
from pathlib import Path

# Add cbftorch to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cbftorch.barriers.barrier import Barrier
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional
from benchmarks.utils.timing import benchmark_function
from box import Box as AD


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def run_cbftorch_benchmark(quick_mode=False, baseline_only=False):
    """Run the CBFtorch benchmark suite."""
    
    print("CBFtorch Performance Benchmark")
    print("=" * 50)
    
    # Configuration
    if quick_mode:
        print("🚀 Quick mode: Fewer runs for fast testing")
        num_runs, warmup_runs = 10, 2
        batch_sizes = [1, 10, 50]
        barrier_counts = [1, 5, 10]
        sim_steps = 20
    else:
        print("🔬 Full mode: Comprehensive performance analysis")
        num_runs, warmup_runs = 30, 5
        batch_sizes = [1, 10, 50, 100]
        barrier_counts = [1, 5, 10, 20]
        sim_steps = 50
    
    # Create results directory
    results_dir = "benchmarks/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results structure
    results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'torch_version': torch.__version__,
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'mode': 'quick' if quick_mode else 'full',
            'baseline_only': baseline_only
        },
        'basic_performance': {},      # Basic barrier/HOCBF evaluation
        'optimization_targets': {},   # Key inefficiencies for optimization
        'summary': {}                # High-level summary
    }
    
    print(f"Device: {results['metadata']['device']}")
    print(f"Runs per test: {num_runs} (warmup: {warmup_runs})")
    print()
    
    # Setup basic components
    dynamics = UnicycleDynamics()
    circle_func = make_circle_barrier_functional(
        center=torch.tensor([0.0, 0.0]), 
        radius=2.0
    )
    
    # === BASIC PERFORMANCE TESTS ===
    print("1️⃣  Basic Performance Tests")
    print("-" * 30)
    
    # Test different relative degrees
    for rel_deg in [1, 2, 3]:
        barrier = Barrier().assign(circle_func, rel_deg=rel_deg)
        barrier.assign_dynamics(dynamics)
        
        rel_deg_results = {}
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, 4, dtype=torch.float64)
            
            # Basic barrier evaluation
            barrier_timing = benchmark_function(
                barrier.barrier, args=(states,), 
                num_runs=num_runs, warmup_runs=warmup_runs
            )
            
            # HOCBF evaluation  
            hocbf_timing = benchmark_function(
                barrier.hocbf, args=(states,),
                num_runs=num_runs, warmup_runs=warmup_runs
            )
            
            # Lie derivatives (if not baseline_only)
            if not baseline_only:
                lie_timing = benchmark_function(
                    barrier.get_hocbf_and_lie_derivs, args=(states,),
                    num_runs=num_runs//2, warmup_runs=warmup_runs//2
                )
                lie_time = lie_timing.mean_time
            else:
                lie_time = 0.0
            
            # Memory measurement
            mem_before = get_memory_usage()
            _ = barrier.hocbf(states)
            mem_after = get_memory_usage()
            
            rel_deg_results[f'batch_{batch_size}'] = {
                'barrier_time': barrier_timing.mean_time,
                'hocbf_time': hocbf_timing.mean_time,
                'lie_derivs_time': lie_time,
                'overhead_vs_barrier': (hocbf_timing.mean_time / barrier_timing.mean_time - 1) * 100,
                'time_per_sample': hocbf_timing.mean_time / batch_size,
                'memory_delta_mb': mem_after - mem_before,
                'batch_size': batch_size
            }
            
            print(f"  Rel deg {rel_deg}, batch {batch_size:3d}: "
                  f"Barrier={barrier_timing.mean_time:.6f}s, "
                  f"HOCBF={hocbf_timing.mean_time:.6f}s "
                  f"({(hocbf_timing.mean_time/barrier_timing.mean_time-1)*100:+5.0f}%)")
        
        results['basic_performance'][f'rel_deg_{rel_deg}'] = rel_deg_results
    
    # Skip optimization tests if baseline_only
    if baseline_only:
        print("\n✅ Baseline measurements complete!")
    else:
        # === OPTIMIZATION TARGET TESTS ===
        print(f"\n2️⃣  Optimization Target Analysis")
        print("-" * 30)
        
        # Test gradient computation redundancy
        print("🎯 Gradient redundancy...")
        barrier = Barrier().assign(circle_func, rel_deg=2)
        barrier.assign_dynamics(dynamics)
        
        redundancy_results = {}
        for batch_size in [1, 10, 50, 100]:
            states = torch.randn(batch_size, 4, dtype=torch.float64)
            
            # Efficient: Combined computation
            combined_timing = benchmark_function(
                barrier.get_hocbf_and_lie_derivs, args=(states,),
                num_runs=num_runs//2, warmup_runs=warmup_runs//2
            )
            
            # Inefficient: Separate computations
            def separate_calls(x):
                return barrier.hocbf(x), barrier.Lf_hocbf(x), barrier.Lg_hocbf(x)
            
            separate_timing = benchmark_function(
                separate_calls, args=(states,),
                num_runs=num_runs//2, warmup_runs=warmup_runs//2
            )
            
            inefficiency = separate_timing.mean_time / combined_timing.mean_time
            potential_savings = (1 - combined_timing.mean_time / separate_timing.mean_time) * 100
            
            redundancy_results[f'batch_{batch_size}'] = {
                'combined_time': combined_timing.mean_time,
                'separate_time': separate_timing.mean_time,
                'inefficiency_factor': inefficiency,
                'potential_savings_percent': potential_savings
            }
            
            print(f"    Batch {batch_size:3d}: {inefficiency:.2f}x inefficiency, "
                  f"{potential_savings:.1f}% potential savings")
        
        results['optimization_targets']['gradient_redundancy'] = redundancy_results
        
        # Test multiple barriers scaling
        print("🎯 Multiple barriers scaling...")
        scaling_results = {}
        
        for num_barriers in barrier_counts:
            # Create multiple barriers
            barriers = []
            for i in range(num_barriers):
                center = torch.tensor([2.0 * i - num_barriers, 2.0 * (i % 2)], dtype=torch.float64)
                func = make_circle_barrier_functional(center=center, radius=1.5)
                b = Barrier().assign(func, rel_deg=1)
                b.assign_dynamics(dynamics)
                barriers.append(b)
            
            # Create composite
            cfg = AD(softmax_rho=20, softmin_rho=20)
            composite = SoftCompositionBarrier(cfg=cfg)
            composite.assign_barriers_and_rule(barriers=barriers, rule='intersection', infer_dynamics=True)
            
            states = torch.randn(25, 4, dtype=torch.float64)
            
            composite_timing = benchmark_function(
                composite.hocbf, args=(states,),
                num_runs=num_runs//2, warmup_runs=warmup_runs//2
            )
            
            time_per_barrier = composite_timing.mean_time / num_barriers
            
            scaling_results[f'barriers_{num_barriers}'] = {
                'total_time': composite_timing.mean_time,
                'time_per_barrier': time_per_barrier,
                'num_barriers': num_barriers
            }
            
            print(f"    {num_barriers:2d} barriers: {composite_timing.mean_time:.6f}s "
                  f"({time_per_barrier:.6f}s per barrier)")
        
        results['optimization_targets']['multiple_barriers'] = scaling_results
        
        # Test simulation pattern (repeated computations)
        print("🎯 Simulation performance...")
        states_sequence = [torch.randn(10, 4, dtype=torch.float64) for _ in range(sim_steps)]
        
        def simulation_pattern():
            for states in states_sequence:
                _ = barrier.get_hocbf_and_lie_derivs(states)
        
        sim_timing = benchmark_function(
            simulation_pattern, num_runs=5, warmup_runs=1
        )
        
        time_per_step = sim_timing.mean_time / sim_steps
        
        results['optimization_targets']['simulation'] = {
            'total_time': sim_timing.mean_time,
            'time_per_step': time_per_step,
            'steps': sim_steps
        }
        
        print(f"    {sim_steps} steps: {sim_timing.mean_time:.4f}s "
              f"({time_per_step:.6f}s per step)")
    
    # === GENERATE SUMMARY ===
    print(f"\n3️⃣  Performance Summary")
    print("-" * 30)
    
    # Calculate key metrics
    basic_perf = results['basic_performance']
    
    # Average times across all tests
    all_barrier_times = []
    all_hocbf_times = []
    all_lie_times = []
    
    for rel_deg_data in basic_perf.values():
        for batch_data in rel_deg_data.values():
            all_barrier_times.append(batch_data['barrier_time'])
            all_hocbf_times.append(batch_data['hocbf_time'])
            if batch_data['lie_derivs_time'] > 0:
                all_lie_times.append(batch_data['lie_derivs_time'])
    
    avg_barrier_time = sum(all_barrier_times) / len(all_barrier_times)
    avg_hocbf_time = sum(all_hocbf_times) / len(all_hocbf_times)
    avg_lie_time = sum(all_lie_times) / len(all_lie_times) if all_lie_times else 0
    
    summary = {
        'avg_barrier_time_us': avg_barrier_time * 1e6,
        'avg_hocbf_time_us': avg_hocbf_time * 1e6,
        'avg_lie_derivs_time_us': avg_lie_time * 1e6,
        'hocbf_vs_barrier_overhead': (avg_hocbf_time / avg_barrier_time - 1) * 100,
        'total_tests': len(all_barrier_times)
    }
    
    if not baseline_only and 'gradient_redundancy' in results['optimization_targets']:
        redundancy = results['optimization_targets']['gradient_redundancy']
        avg_inefficiency = sum(r['inefficiency_factor'] for r in redundancy.values()) / len(redundancy)
        avg_savings = sum(r['potential_savings_percent'] for r in redundancy.values()) / len(redundancy)
        
        summary['optimization_opportunities'] = {
            'gradient_inefficiency_factor': avg_inefficiency,
            'potential_time_savings_percent': avg_savings
        }
    
    results['summary'] = summary
    
    # Print summary
    print(f"Average barrier evaluation: {summary['avg_barrier_time_us']:.0f}μs")
    print(f"Average HOCBF evaluation: {summary['avg_hocbf_time_us']:.0f}μs")
    if avg_lie_time > 0:
        print(f"Average Lie derivatives: {summary['avg_lie_derivs_time_us']:.0f}μs")
    print(f"HOCBF overhead vs barrier: {summary['hocbf_vs_barrier_overhead']:.0f}%")
    
    if 'optimization_opportunities' in summary:
        opt = summary['optimization_opportunities']
        print(f"🎯 Optimization potential: {opt['potential_time_savings_percent']:.1f}% time savings available")
    
    # === SAVE RESULTS ===
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_quick" if quick_mode else "_baseline" if baseline_only else "_full"
    filename = f"cbftorch_benchmark{mode_suffix}_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {filepath}")
    
    if not baseline_only:
        print(f"\n🚀 Ready for optimization! Key targets identified:")
        if 'optimization_opportunities' in summary:
            print(f"   • Gradient computation: {summary['optimization_opportunities']['potential_time_savings_percent']:.1f}% potential savings")
        print(f"   • HOCBF evaluation: {summary['hocbf_vs_barrier_overhead']:.0f}% overhead to optimize")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CBFtorch Performance Benchmark")
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode: fewer runs for fast testing')
    parser.add_argument('--baseline', action='store_true',
                       help='Baseline only: skip optimization analysis')
    
    args = parser.parse_args()
    
    try:
        results = run_cbftorch_benchmark(
            quick_mode=args.quick,
            baseline_only=args.baseline
        )
        print("\n✅ Benchmark completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()