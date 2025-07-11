#!/usr/bin/env python3
"""
Comprehensive benchmark covering all identified optimization targets.
This benchmark fills the gaps in simple_benchmark.py and covers:
1. Gradient computation redundancy
2. Multiple barrier scaling  
3. QP control performance
4. Repeated computation patterns
5. State dimension scaling
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
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier, NonSmoothCompositionBarrier
from cbftorch.safe_controls.qp_safe_control import MinIntervQPSafeControl
from cbftorch.dynamics import UnicycleDynamics, BicycleDynamics, DIDynamics
from cbftorch.utils.utils import make_circle_barrier_functional, make_norm_rectangular_barrier_functional
from benchmarks.utils.timing import benchmark_function
from box import Box as AD


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
    """Run comprehensive benchmark covering all optimization targets."""
    print("CBFtorch Comprehensive Optimization Benchmark")
    print("="*60)
    
    # Create results directory
    results_dir = "benchmarks/results"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'torch_version': torch.__version__,
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'benchmark_type': 'comprehensive_optimization'
        },
        'gradient_redundancy': {},
        'multiple_barriers_scaling': {},
        'qp_control_performance': {},
        'repeated_computations': {},
        'state_dimension_scaling': {},
        'end_to_end_simulation': {}
    }
    
    num_runs = 30
    warmup_runs = 5
    
    # 1. GRADIENT COMPUTATION REDUNDANCY BENCHMARK
    print("\n1. Testing gradient computation redundancy...")
    
    dynamics = UnicycleDynamics()
    circle_func = make_circle_barrier_functional(
        center=torch.tensor([0.0, 0.0]), 
        radius=2.0
    )
    barrier = Barrier().assign(circle_func, rel_deg=2)
    barrier.assign_dynamics(dynamics)
    
    batch_sizes = [1, 10, 50, 100]
    
    for batch_size in batch_sizes:
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        test_name = f"batch_{batch_size}"
        
        # Method 1: Combined computation (current implementation)
        timing_combined = benchmark_function(
            barrier.get_hocbf_and_lie_derivs,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Method 2: Separate computations (what people might do manually)
        def separate_computation(x):
            hocbf = barrier.hocbf(x)
            lf = barrier.Lf_hocbf(x) 
            lg = barrier.Lg_hocbf(x)
            return hocbf, lf, lg
        
        timing_separate = benchmark_function(
            separate_computation,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Method 3: Alternative implementation (if available)
        timing_v2 = benchmark_function(
            barrier.get_hocbf_and_lie_derivs_v2,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        results['gradient_redundancy'][test_name] = {
            'combined_time': timing_combined.mean_time,
            'separate_time': timing_separate.mean_time,
            'v2_time': timing_v2.mean_time,
            'redundancy_overhead': timing_separate.mean_time / timing_combined.mean_time,
            'v2_speedup': timing_combined.mean_time / timing_v2.mean_time,
            'batch_size': batch_size
        }
        
        print(f"  Batch {batch_size}: Combined={timing_combined.mean_time:.6f}s, "
              f"Separate={timing_separate.mean_time:.6f}s (overhead: {timing_separate.mean_time/timing_combined.mean_time:.2f}x), "
              f"V2={timing_v2.mean_time:.6f}s (speedup: {timing_combined.mean_time/timing_v2.mean_time:.2f}x)")
    
    # 2. MULTIPLE BARRIERS SCALING BENCHMARK
    print("\n2. Testing multiple barriers scaling...")
    
    def create_multiple_barriers(num_barriers):
        """Create multiple barriers for composition."""
        barriers = []
        for i in range(num_barriers):
            center = torch.tensor([2.0 * i - num_barriers, 2.0 * (i % 2)], dtype=torch.float64)
            barrier_func = make_circle_barrier_functional(center=center, radius=1.5)
            barrier = Barrier().assign(barrier_func, rel_deg=1)
            barrier.assign_dynamics(dynamics)
            barriers.append(barrier)
        return barriers
    
    barrier_counts = [1, 5, 10, 20]
    states = torch.randn(50, 4, dtype=torch.float64)  # Fixed batch size
    cfg = AD(softmax_rho=20, softmin_rho=20)
    
    for num_barriers in barrier_counts:
        individual_barriers = create_multiple_barriers(num_barriers)
        
        # Test soft composition (most common)
        soft_composite = SoftCompositionBarrier(cfg=cfg)
        soft_composite.assign_barriers_and_rule(
            barriers=individual_barriers,
            rule='intersection',
            infer_dynamics=True
        )
        
        # Benchmark composite barrier evaluation
        timing_composite = benchmark_function(
            soft_composite.hocbf,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Benchmark combined lie derivatives  
        timing_lie_derivs = benchmark_function(
            soft_composite.get_hocbf_and_lie_derivs,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Memory usage
        mem_before = get_memory_usage()
        _ = soft_composite.get_hocbf_and_lie_derivs(states)
        mem_after = get_memory_usage()
        memory_delta = mem_after['rss_mb'] - mem_before['rss_mb']
        
        results['multiple_barriers_scaling'][f'barriers_{num_barriers}'] = {
            'composite_hocbf_time': timing_composite.mean_time,
            'lie_derivs_time': timing_lie_derivs.mean_time,
            'time_per_barrier': timing_composite.mean_time / num_barriers,
            'memory_delta_mb': memory_delta,
            'num_barriers': num_barriers,
            'scaling_efficiency': (timing_composite.mean_time / num_barriers) if num_barriers > 1 else 1.0
        }
        
        print(f"  {num_barriers} barriers: HOCBF={timing_composite.mean_time:.6f}s, "
              f"LieDerivs={timing_lie_derivs.mean_time:.6f}s, "
              f"Per-barrier={timing_composite.mean_time/num_barriers:.6f}s")
    
    # 3. QP CONTROL PERFORMANCE BENCHMARK
    print("\n3. Testing QP control performance...")
    
    # Create QP safe controller
    qp_controller = MinIntervQPSafeControl(
        action_dim=dynamics.action_dim,
        alpha=lambda x: 0.5 * x,
        params=AD(slacked=False)
    )
    qp_controller.assign_dynamics(dynamics)
    qp_controller.assign_state_barrier(barrier)
    
    # Define desired control
    def desired_control(x):
        return torch.zeros(x.shape[0], 2, dtype=torch.float64)
    
    qp_controller.assign_desired_control(desired_control)
    
    for batch_size in [1, 10, 25, 50]:
        states = torch.randn(batch_size, 4, dtype=torch.float64)
        
        # Benchmark QP solving
        timing_qp = benchmark_function(
            qp_controller.safe_optimal_control,
            args=(states,),
            num_runs=num_runs//2,  # Fewer runs for expensive QP
            warmup_runs=warmup_runs//2
        )
        
        # Memory usage
        mem_before = get_memory_usage()
        _ = qp_controller.safe_optimal_control(states)
        mem_after = get_memory_usage()
        memory_delta = mem_after['rss_mb'] - mem_before['rss_mb']
        
        results['qp_control_performance'][f'batch_{batch_size}'] = {
            'qp_solve_time': timing_qp.mean_time,
            'time_per_sample': timing_qp.mean_time / batch_size,
            'memory_delta_mb': memory_delta,
            'batch_size': batch_size
        }
        
        print(f"  QP Batch {batch_size}: {timing_qp.mean_time:.6f}s "
              f"({timing_qp.mean_time/batch_size:.6f}s per sample)")
    
    # 4. REPEATED COMPUTATIONS BENCHMARK (Simulation-like)
    print("\n4. Testing repeated computations (simulation pattern)...")
    
    # Simulate repeated barrier evaluations (like in a control loop)
    states_sequence = [torch.randn(10, 4, dtype=torch.float64) for _ in range(100)]
    
    def repeated_barrier_calls():
        """Simulate 100 control steps."""
        for states in states_sequence:
            _ = barrier.get_hocbf_and_lie_derivs(states)
    
    def repeated_separate_calls():
        """Simulate 100 control steps with separate calls."""
        for states in states_sequence:
            _ = barrier.hocbf(states)
            _ = barrier.Lf_hocbf(states)
            _ = barrier.Lg_hocbf(states)
    
    timing_repeated = benchmark_function(
        repeated_barrier_calls,
        num_runs=5,  # Fewer runs for long simulation
        warmup_runs=1
    )
    
    timing_separate_repeated = benchmark_function(
        repeated_separate_calls,
        num_runs=5,
        warmup_runs=1
    )
    
    results['repeated_computations'] = {
        'simulation_steps': 100,
        'combined_total_time': timing_repeated.mean_time,
        'separate_total_time': timing_separate_repeated.mean_time,
        'time_per_step_combined': timing_repeated.mean_time / 100,
        'time_per_step_separate': timing_separate_repeated.mean_time / 100,
        'efficiency_gain': timing_separate_repeated.mean_time / timing_repeated.mean_time
    }
    
    print(f"  100 steps: Combined={timing_repeated.mean_time:.4f}s "
          f"({timing_repeated.mean_time/100:.6f}s/step), "
          f"Separate={timing_separate_repeated.mean_time:.4f}s "
          f"({timing_separate_repeated.mean_time/100:.6f}s/step)")
    
    # 5. STATE DIMENSION SCALING BENCHMARK
    print("\n5. Testing state dimension scaling...")
    
    # Test different dynamics with different state dimensions
    dynamics_configs = [
        ('2D', DIDynamics(), 4),          # Double integrator: 4D (x, y, vx, vy)
        ('unicycle', UnicycleDynamics(), 4),  # Unicycle: 4D (x, y, v, theta)
        ('bicycle', BicycleDynamics(), 6),    # Bicycle: 6D (x, y, v, theta, delta, beta)
    ]
    
    for dyn_name, dyn, state_dim in dynamics_configs:
        # Create barrier for this dynamics
        barrier_func = make_circle_barrier_functional(
            center=torch.zeros(2), radius=2.0
        )
        test_barrier = Barrier().assign(barrier_func, rel_deg=2)
        test_barrier.assign_dynamics(dyn)
        
        states = torch.randn(25, state_dim, dtype=torch.float64)
        
        # Benchmark barrier evaluation
        timing_barrier = benchmark_function(
            test_barrier.hocbf,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        # Benchmark lie derivatives
        timing_lie = benchmark_function(
            test_barrier.get_hocbf_and_lie_derivs,
            args=(states,),
            num_runs=num_runs,
            warmup_runs=warmup_runs
        )
        
        results['state_dimension_scaling'][dyn_name] = {
            'state_dim': state_dim,
            'hocbf_time': timing_barrier.mean_time,
            'lie_derivs_time': timing_lie.mean_time,
            'time_per_state_dim': timing_lie.mean_time / state_dim
        }
        
        print(f"  {dyn_name} ({state_dim}D): HOCBF={timing_barrier.mean_time:.6f}s, "
              f"LieDerivs={timing_lie.mean_time:.6f}s")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_benchmark_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nComprehensive benchmark results saved to: {filepath}")
    
    # Print summary
    print_comprehensive_summary(results)
    
    return results


def print_comprehensive_summary(results):
    """Print summary of comprehensive benchmark results."""
    print("\n" + "="*60)
    print("COMPREHENSIVE BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"Device: {results['metadata']['device']}")
    print(f"Timestamp: {results['metadata']['timestamp']}")
    print()
    
    # Gradient redundancy analysis
    print("GRADIENT COMPUTATION EFFICIENCY:")
    print("-" * 40)
    redundancy = results['gradient_redundancy']
    avg_overhead = sum(r['redundancy_overhead'] for r in redundancy.values()) / len(redundancy)
    avg_v2_speedup = sum(r['v2_speedup'] for r in redundancy.values()) / len(redundancy)
    print(f"Average redundancy overhead: {avg_overhead:.2f}x")
    print(f"Average V2 speedup: {avg_v2_speedup:.2f}x")
    print()
    
    # Multiple barriers scaling
    print("MULTIPLE BARRIERS SCALING:")
    print("-" * 40)
    scaling = results['multiple_barriers_scaling']
    for key, data in scaling.items():
        print(f"{data['num_barriers']} barriers: {data['composite_hocbf_time']:.6f}s "
              f"({data['time_per_barrier']:.6f}s per barrier)")
    print()
    
    # QP performance
    print("QP CONTROL PERFORMANCE:")
    print("-" * 40)
    qp_perf = results['qp_control_performance']
    for key, data in qp_perf.items():
        print(f"Batch {data['batch_size']}: {data['qp_solve_time']:.6f}s "
              f"({data['time_per_sample']:.6f}s per sample)")
    print()
    
    # Repeated computations
    print("SIMULATION PERFORMANCE:")
    print("-" * 40)
    repeated = results['repeated_computations']
    print(f"100 simulation steps: {repeated['time_per_step_combined']:.6f}s per step")
    print(f"Efficiency vs separate calls: {repeated['efficiency_gain']:.2f}x")
    print()
    
    # State dimension scaling
    print("STATE DIMENSION SCALING:")
    print("-" * 40)
    state_scaling = results['state_dimension_scaling']
    for dyn_name, data in state_scaling.items():
        print(f"{dyn_name} ({data['state_dim']}D): {data['lie_derivs_time']:.6f}s "
              f"({data['time_per_state_dim']:.6f}s per dim)")
    
    print("="*60)


if __name__ == "__main__":
    run_comprehensive_benchmark()