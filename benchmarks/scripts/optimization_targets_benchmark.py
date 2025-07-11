#!/usr/bin/env python3
"""
Optimization targets benchmark - focused on the key inefficiencies we identified.
This version is more robust and focuses on the critical optimization opportunities.
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
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional
from benchmarks.utils.timing import benchmark_function
from box import Box as AD


def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024


def run_optimization_targets_benchmark():
    """Run benchmark focused on key optimization targets."""
    print("CBFtorch Optimization Targets Benchmark")
    print("="*50)
    
    # Create results directory
    results_dir = "benchmarks/results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'torch_version': torch.__version__,
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'benchmark_type': 'optimization_targets'
        },
        'gradient_redundancy': {},
        'multiple_barriers_scaling': {},
        'repeated_computations': {},
        'hocbf_generation_overhead': {},
        'memory_patterns': {}
    }
    
    num_runs = 20
    warmup_runs = 3
    
    # Setup basic components
    dynamics = UnicycleDynamics()
    circle_func = make_circle_barrier_functional(
        center=torch.tensor([0.0, 0.0]), 
        radius=2.0
    )
    
    # 1. GRADIENT REDUNDANCY BENCHMARK (Critical Target #1)
    print("\n1. Testing gradient computation redundancy...")
    
    barrier = Barrier().assign(circle_func, rel_deg=2)
    barrier.assign_dynamics(dynamics)
    
    batch_sizes = [1, 10, 50, 100]
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        # Method A: Combined (efficient)
        timing_combined = benchmark_function(
            barrier.get_hocbf_and_lie_derivs,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Method B: Separate calls (inefficient)
        def separate_calls(x):
            h = barrier.hocbf(x)
            lf = barrier.Lf_hocbf(x)
            lg = barrier.Lg_hocbf(x)
            return h, lf, lg
        
        timing_separate = benchmark_function(
            separate_calls,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Method C: Alternative version
        timing_v2 = benchmark_function(
            barrier.get_hocbf_and_lie_derivs_v2,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        inefficiency_factor = timing_separate.mean_time / timing_combined.mean_time
        v2_factor = timing_combined.mean_time / timing_v2.mean_time
        
        results['gradient_redundancy'][f'batch_{batch_size}'] = {
            'combined_time': timing_combined.mean_time,
            'separate_time': timing_separate.mean_time,
            'v2_time': timing_v2.mean_time,
            'inefficiency_factor': inefficiency_factor,
            'v2_speedup': v2_factor,
            'potential_savings': (timing_separate.mean_time - timing_combined.mean_time) / timing_separate.mean_time,
            'batch_size': batch_size
        }
        
        print(f"  Batch {batch_size}: Combined={timing_combined.mean_time:.6f}s, "
              f"Separate={timing_separate.mean_time:.6f}s "
              f"(Inefficiency: {inefficiency_factor:.2f}x, Potential savings: {((timing_separate.mean_time - timing_combined.mean_time) / timing_separate.mean_time * 100):.1f}%)")
    
    # 2. MULTIPLE BARRIERS SCALING (Critical Target #2)
    print("\n2. Testing multiple barriers scaling...")
    
    def create_barriers(num_barriers):
        barriers = []
        for i in range(num_barriers):
            center = torch.tensor([2.0 * i - num_barriers, 2.0 * (i % 2)], dtype=torch.float64)
            func = make_circle_barrier_functional(center=center, radius=1.5)
            b = Barrier().assign(func, rel_deg=1)
            b.assign_dynamics(dynamics)
            barriers.append(b)
        return barriers
    
    states = torch.randn(25, 4, dtype=torch.float64)
    cfg = AD(softmax_rho=20, softmin_rho=20)
    barrier_counts = [1, 5, 10, 20]
    
    for num_barriers in barrier_counts:
        try:
            individual_barriers = create_barriers(num_barriers)
            
            composite = SoftCompositionBarrier(cfg=cfg)
            composite.assign_barriers_and_rule(
                barriers=individual_barriers,
                rule='intersection',
                infer_dynamics=True
            )
            
            # Benchmark composite evaluation
            mem_before = get_memory_usage()
            
            timing_composite = benchmark_function(
                composite.hocbf,
                args=(states,),
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
            
            timing_lie_derivs = benchmark_function(
                composite.get_hocbf_and_lie_derivs,
                args=(states,),
                num_runs=num_runs//2,  # Fewer for expensive operation
                warmup_runs=warmup_runs//2
            )
            
            mem_after = get_memory_usage()
            memory_delta = mem_after - mem_before
            
            # Calculate scaling efficiency
            single_barrier_time = results['multiple_barriers_scaling'].get('barriers_1', {}).get('hocbf_time', timing_composite.mean_time)
            scaling_efficiency = single_barrier_time * num_barriers / timing_composite.mean_time if num_barriers > 1 else 1.0
            
            results['multiple_barriers_scaling'][f'barriers_{num_barriers}'] = {
                'hocbf_time': timing_composite.mean_time,
                'lie_derivs_time': timing_lie_derivs.mean_time,
                'time_per_barrier': timing_composite.mean_time / num_barriers,
                'scaling_efficiency': scaling_efficiency,
                'memory_delta_mb': memory_delta,
                'num_barriers': num_barriers
            }
            
            print(f"  {num_barriers} barriers: HOCBF={timing_composite.mean_time:.6f}s "
                  f"({timing_composite.mean_time/num_barriers:.6f}s per barrier), "
                  f"Efficiency: {scaling_efficiency:.2f}")
            
        except Exception as e:
            print(f"  {num_barriers} barriers: FAILED - {str(e)[:50]}...")
            results['multiple_barriers_scaling'][f'barriers_{num_barriers}'] = {
                'error': str(e),
                'num_barriers': num_barriers
            }
    
    # 3. REPEATED COMPUTATIONS (Critical Target #3)
    print("\n3. Testing repeated computations (simulation pattern)...")
    
    # Simulate control loop pattern
    simulation_steps = 50
    states_sequence = [torch.randn(5, 4, dtype=torch.float64) for _ in range(simulation_steps)]
    
    def simulation_combined():
        for states in states_sequence:
            _ = barrier.get_hocbf_and_lie_derivs(states)
    
    def simulation_separate():
        for states in states_sequence:
            _ = barrier.hocbf(states)
            _ = barrier.Lf_hocbf(states)
            _ = barrier.Lg_hocbf(states)
    
    timing_sim_combined = benchmark_function(
        simulation_combined,
        num_runs=5,
        warmup_runs=1
    )
    
    timing_sim_separate = benchmark_function(
        simulation_separate,
        num_runs=5,
        warmup_runs=1
    )
    
    efficiency_gain = timing_sim_separate.mean_time / timing_sim_combined.mean_time
    time_per_step = timing_sim_combined.mean_time / simulation_steps
    
    results['repeated_computations'] = {
        'simulation_steps': simulation_steps,
        'combined_total_time': timing_sim_combined.mean_time,
        'separate_total_time': timing_sim_separate.mean_time,
        'time_per_step': time_per_step,
        'efficiency_gain': efficiency_gain,
        'potential_time_savings_per_step': (timing_sim_separate.mean_time - timing_sim_combined.mean_time) / simulation_steps
    }
    
    print(f"  {simulation_steps} steps: Combined={timing_sim_combined.mean_time:.4f}s "
          f"({time_per_step:.6f}s/step), "
          f"Efficiency gain vs separate: {efficiency_gain:.2f}x")
    
    # 4. HOCBF GENERATION OVERHEAD (Critical Target #4)
    print("\n4. Testing HOCBF generation overhead...")
    
    rel_degrees = [1, 2, 3, 4]
    states = torch.randn(20, 4, dtype=torch.float64)
    
    for rel_deg in rel_degrees:
        try:
            # Time barrier creation
            def create_barrier():
                b = Barrier().assign(circle_func, rel_deg=rel_deg)
                b.assign_dynamics(dynamics)
                return b
            
            mem_before = get_memory_usage()
            timing_creation = benchmark_function(
                create_barrier,
                num_runs=10,
                warmup_runs=2
            )
            mem_after = get_memory_usage()
            
            # Time evaluation of created barrier
            test_barrier = create_barrier()
            timing_evaluation = benchmark_function(
                test_barrier.hocbf,
                args=(states,),
                num_runs=num_runs,
                warmup_runs=warmup_runs
            )
            
            # Calculate overhead
            creation_overhead = timing_creation.mean_time
            evaluation_time = timing_evaluation.mean_time
            overhead_ratio = creation_overhead / evaluation_time if evaluation_time > 0 else 0
            
            results['hocbf_generation_overhead'][f'rel_deg_{rel_deg}'] = {
                'creation_time': timing_creation.mean_time,
                'evaluation_time': timing_evaluation.mean_time,
                'overhead_ratio': overhead_ratio,
                'memory_delta_mb': mem_after - mem_before,
                'rel_deg': rel_deg
            }
            
            print(f"  Rel deg {rel_deg}: Creation={timing_creation.mean_time:.6f}s, "
                  f"Evaluation={timing_evaluation.mean_time:.6f}s "
                  f"(Overhead ratio: {overhead_ratio:.2f})")
            
        except Exception as e:
            print(f"  Rel deg {rel_deg}: FAILED - {str(e)[:50]}...")
    
    # 5. MEMORY PATTERNS
    print("\n5. Testing memory allocation patterns...")
    
    # Test memory growth over multiple evaluations
    initial_memory = get_memory_usage()
    memory_samples = [initial_memory]
    
    large_states = torch.randn(100, 4, dtype=torch.float64)
    
    for i in range(20):
        _ = barrier.get_hocbf_and_lie_derivs(large_states)
        if i % 5 == 0:  # Sample every 5 iterations
            memory_samples.append(get_memory_usage())
    
    final_memory = get_memory_usage()
    memory_growth = final_memory - initial_memory
    
    results['memory_patterns'] = {
        'initial_memory_mb': initial_memory,
        'final_memory_mb': final_memory,
        'memory_growth_mb': memory_growth,
        'memory_samples': memory_samples,
        'evaluations_tested': 20
    }
    
    print(f"  Memory growth over 20 evaluations: {memory_growth:.2f}MB")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"optimization_targets_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOptimization targets benchmark saved to: {filepath}")
    
    # Print summary
    print_optimization_summary(results)
    
    return results


def print_optimization_summary(results):
    """Print summary focusing on optimization opportunities."""
    print("\n" + "="*50)
    print("OPTIMIZATION TARGETS SUMMARY")
    print("="*50)
    
    print(f"Device: {results['metadata']['device']}")
    print(f"Timestamp: {results['metadata']['timestamp']}")
    print()
    
    # Gradient redundancy opportunities
    print("🎯 GRADIENT COMPUTATION REDUNDANCY:")
    print("-" * 40)
    redundancy = results['gradient_redundancy']
    avg_inefficiency = sum(r['inefficiency_factor'] for r in redundancy.values()) / len(redundancy)
    avg_savings = sum(r['potential_savings'] for r in redundancy.values()) / len(redundancy) * 100
    
    print(f"Average inefficiency factor: {avg_inefficiency:.2f}x")
    print(f"Average potential time savings: {avg_savings:.1f}%")
    print("🔧 Optimization: Cache gradients, batch computations")
    print()
    
    # Multiple barriers scaling
    print("🎯 MULTIPLE BARRIERS SCALING:")
    print("-" * 40)
    scaling = results['multiple_barriers_scaling']
    
    working_results = {k: v for k, v in scaling.items() if 'error' not in v}
    if working_results:
        worst_efficiency = min(r['scaling_efficiency'] for r in working_results.values())
        print(f"Worst scaling efficiency: {worst_efficiency:.2f}")
        print("🔧 Optimization: Vectorize barrier computations, optimize composition")
    else:
        print("❌ Multiple barriers test failed - needs investigation")
    print()
    
    # Repeated computations
    print("🎯 REPEATED COMPUTATIONS:")
    print("-" * 40)
    repeated = results['repeated_computations']
    efficiency_gain = repeated['efficiency_gain']
    time_savings = repeated['potential_time_savings_per_step'] * 1000  # ms
    
    print(f"Efficiency gain over separate calls: {efficiency_gain:.2f}x")
    print(f"Time savings per simulation step: {time_savings:.3f}ms")
    print("🔧 Optimization: Gradient caching, reduced redundant computations")
    print()
    
    # HOCBF generation
    print("🎯 HOCBF GENERATION OVERHEAD:")
    print("-" * 40)
    hocbf_gen = results['hocbf_generation_overhead']
    
    working_hocbf = {k: v for k, v in hocbf_gen.items() if 'error' not in v}
    if working_hocbf:
        avg_overhead = sum(r['overhead_ratio'] for r in working_hocbf.values()) / len(working_hocbf)
        print(f"Average creation/evaluation overhead: {avg_overhead:.2f}")
        print("🔧 Optimization: Direct computation instead of lambda chains")
    else:
        print("❌ HOCBF generation test failed - needs investigation")
    print()
    
    # Memory patterns
    print("🎯 MEMORY ALLOCATION PATTERNS:")
    print("-" * 40)
    memory = results['memory_patterns']
    memory_growth = memory['memory_growth_mb']
    
    print(f"Memory growth over 20 evaluations: {memory_growth:.2f}MB")
    
    if memory_growth > 10:
        print("⚠️  Significant memory growth detected - potential leak")
    elif memory_growth > 1:
        print("ℹ️  Moderate memory growth - monitor in long simulations")
    else:
        print("✅ Minimal memory growth - good allocation patterns")
    
    print("🔧 Optimization: In-place operations, tensor reuse")
    print()
    
    print("="*50)
    print("🚀 READY FOR OPTIMIZATION IMPLEMENTATION!")
    print("="*50)


if __name__ == "__main__":
    run_optimization_targets_benchmark()