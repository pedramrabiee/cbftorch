#!/usr/bin/env python3
"""
CBFtorch Benchmark Analysis Tool

Analyze benchmark results and generate reports.

Usage:
    python benchmarks/analyze.py                    # Analyze latest results
    python benchmarks/analyze.py --summary          # Print text summary only
    python benchmarks/analyze.py --plot             # Generate plots  
    python benchmarks/analyze.py --compare file1 file2  # Compare two result files
"""

import os
import json
import glob
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def find_latest_results(results_dir="benchmarks/results"):
    """Find the most recent benchmark results."""
    result_files = glob.glob(os.path.join(results_dir, "cbftorch_benchmark_*.json"))
    if not result_files:
        print("❌ No benchmark results found!")
        print("   Run: python benchmarks/benchmark.py")
        return None
    
    latest = max(result_files, key=os.path.getctime)
    return latest


def load_results(filepath):
    """Load benchmark results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_summary(results):
    """Print a text summary of benchmark results."""
    metadata = results['metadata']
    summary = results['summary']
    
    print("\n" + "="*50)
    print("CBFTORCH BENCHMARK SUMMARY")
    print("="*50)
    
    print(f"Timestamp: {metadata['timestamp']}")
    print(f"Device: {metadata['device']}")
    print(f"Mode: {metadata['mode']}")
    print(f"Total tests: {summary['total_tests']}")
    print()
    
    print("PERFORMANCE METRICS:")
    print("-" * 30)
    print(f"Average barrier evaluation: {summary['avg_barrier_time_us']:.0f}μs")
    print(f"Average HOCBF evaluation: {summary['avg_hocbf_time_us']:.0f}μs")
    
    if summary.get('avg_lie_derivs_time_us', 0) > 0:
        print(f"Average Lie derivatives: {summary['avg_lie_derivs_time_us']:.0f}μs")
    
    print(f"HOCBF overhead: {summary['hocbf_vs_barrier_overhead']:.0f}%")
    print()
    
    if 'optimization_opportunities' in summary:
        opt = summary['optimization_opportunities']
        print("OPTIMIZATION OPPORTUNITIES:")
        print("-" * 30)
        print(f"Gradient inefficiency: {opt['gradient_inefficiency_factor']:.2f}x")
        print(f"Potential time savings: {opt['potential_time_savings_percent']:.1f}%")
        print()
    
    # Show performance by relative degree
    if 'basic_performance' in results:
        print("PERFORMANCE BY RELATIVE DEGREE:")
        print("-" * 30)
        
        for rel_deg_key, rel_deg_data in results['basic_performance'].items():
            rel_deg = rel_deg_key.split('_')[-1]
            
            # Get average times for this relative degree
            barrier_times = [data['barrier_time'] for data in rel_deg_data.values()]
            hocbf_times = [data['hocbf_time'] for data in rel_deg_data.values()]
            
            avg_barrier = np.mean(barrier_times) * 1e6  # Convert to μs
            avg_hocbf = np.mean(hocbf_times) * 1e6
            
            print(f"Rel deg {rel_deg}: Barrier={avg_barrier:.0f}μs, HOCBF={avg_hocbf:.0f}μs")
    
    print("="*50)


def create_plots(results, save_path=None):
    """Create performance visualization plots."""
    if 'basic_performance' not in results:
        print("❌ No basic performance data found for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('CBFtorch Performance Analysis', fontsize=14)
    
    # Extract data for plotting
    rel_degs = []
    barrier_times = []
    hocbf_times = []
    batch_sizes = []
    batch_perf = []
    
    for rel_deg_key, rel_deg_data in results['basic_performance'].items():
        rel_deg = int(rel_deg_key.split('_')[-1])
        
        for batch_key, batch_data in rel_deg_data.items():
            rel_degs.append(rel_deg)
            barrier_times.append(batch_data['barrier_time'] * 1e6)  # Convert to μs
            hocbf_times.append(batch_data['hocbf_time'] * 1e6)
            
            batch_size = batch_data['batch_size']
            batch_sizes.append(batch_size)
            batch_perf.append(batch_data['time_per_sample'] * 1e6)
    
    # Plot 1: Performance by relative degree
    rel_deg_groups = {}
    for i, rd in enumerate(rel_degs):
        if rd not in rel_deg_groups:
            rel_deg_groups[rd] = {'barrier': [], 'hocbf': []}
        rel_deg_groups[rd]['barrier'].append(barrier_times[i])
        rel_deg_groups[rd]['hocbf'].append(hocbf_times[i])
    
    x_pos = list(rel_deg_groups.keys())
    barrier_means = [np.mean(rel_deg_groups[rd]['barrier']) for rd in x_pos]
    hocbf_means = [np.mean(rel_deg_groups[rd]['hocbf']) for rd in x_pos]
    
    width = 0.35
    x = np.arange(len(x_pos))
    
    axes[0, 0].bar(x - width/2, barrier_means, width, label='Barrier', alpha=0.8)
    axes[0, 0].bar(x + width/2, hocbf_means, width, label='HOCBF', alpha=0.8)
    axes[0, 0].set_xlabel('Relative Degree')
    axes[0, 0].set_ylabel('Time (μs)')
    axes[0, 0].set_title('Performance by Relative Degree')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(x_pos)
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    
    # Plot 2: Batch size scaling
    batch_groups = {}
    for i, bs in enumerate(batch_sizes):
        if bs not in batch_groups:
            batch_groups[bs] = []
        batch_groups[bs].append(batch_perf[i])
    
    batch_x = sorted(batch_groups.keys())
    batch_means = [np.mean(batch_groups[bs]) for bs in batch_x]
    
    axes[0, 1].plot(batch_x, batch_means, 'o-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Time per Sample (μs)')
    axes[0, 1].set_title('Batch Size Scaling')
    axes[0, 1].set_xscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Overhead analysis
    overheads = []
    rel_deg_labels = []
    for rel_deg_key, rel_deg_data in results['basic_performance'].items():
        rel_deg = rel_deg_key.split('_')[-1]
        avg_overhead = np.mean([data['overhead_vs_barrier'] for data in rel_deg_data.values()])
        overheads.append(avg_overhead)
        rel_deg_labels.append(f"Rel deg {rel_deg}")
    
    axes[1, 0].bar(rel_deg_labels, overheads, alpha=0.8, color='orange')
    axes[1, 0].set_ylabel('Overhead (%)')
    axes[1, 0].set_title('HOCBF vs Barrier Overhead')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Summary metrics
    summary_data = results['summary']
    metrics = ['Barrier', 'HOCBF', 'Lie Derivs']
    times = [
        summary_data['avg_barrier_time_us'],
        summary_data['avg_hocbf_time_us'],
        summary_data.get('avg_lie_derivs_time_us', 0)
    ]
    
    # Filter out zero times
    non_zero_metrics = []
    non_zero_times = []
    for m, t in zip(metrics, times):
        if t > 0:
            non_zero_metrics.append(m)
            non_zero_times.append(t)
    
    axes[1, 1].bar(non_zero_metrics, non_zero_times, alpha=0.8, color='green')
    axes[1, 1].set_ylabel('Average Time (μs)')
    axes[1, 1].set_title('Average Operation Times')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to: {save_path}")
    
    return fig


def compare_results(file1, file2):
    """Compare two benchmark result files."""
    results1 = load_results(file1)
    results2 = load_results(file2)
    
    summary1 = results1['summary']
    summary2 = results2['summary']
    
    print("\n" + "="*50)
    print("BENCHMARK COMPARISON")
    print("="*50)
    
    print(f"File 1: {os.path.basename(file1)} ({results1['metadata']['timestamp']})")
    print(f"File 2: {os.path.basename(file2)} ({results2['metadata']['timestamp']})")
    print()
    
    # Compare key metrics
    metrics = [
        ('avg_barrier_time_us', 'Barrier time'),
        ('avg_hocbf_time_us', 'HOCBF time'),
        ('avg_lie_derivs_time_us', 'Lie derivatives time')
    ]
    
    print("PERFORMANCE COMPARISON:")
    print("-" * 30)
    
    for metric_key, metric_name in metrics:
        val1 = summary1.get(metric_key, 0)
        val2 = summary2.get(metric_key, 0)
        
        if val1 > 0 and val2 > 0:
            speedup = val1 / val2
            change_pct = (val2 - val1) / val1 * 100
            
            print(f"{metric_name}:")
            print(f"  File 1: {val1:.0f}μs")
            print(f"  File 2: {val2:.0f}μs")
            
            if speedup > 1.05:
                print(f"  🚀 {speedup:.2f}x speedup ({-change_pct:+.1f}%)")
            elif speedup < 0.95:
                print(f"  🐌 {1/speedup:.2f}x slower ({-change_pct:+.1f}%)")
            else:
                print(f"  ➡️  Similar performance ({-change_pct:+.1f}%)")
            print()
    
    print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CBFtorch Benchmark Analysis")
    parser.add_argument('--file', help='Specific result file to analyze')
    parser.add_argument('--summary', action='store_true', help='Print summary only')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--compare', nargs=2, metavar=('file1', 'file2'),
                       help='Compare two result files')
    parser.add_argument('--output-dir', default='benchmarks/analysis',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.compare:
        compare_results(args.compare[0], args.compare[1])
        return
    
    # Find results file
    if args.file:
        results_file = args.file
    else:
        results_file = find_latest_results()
        if not results_file:
            return
    
    print(f"📁 Analyzing: {os.path.basename(results_file)}")
    
    # Load and analyze results
    results = load_results(results_file)
    
    if args.summary or not (args.plot):
        print_summary(results)
    
    if args.plot:
        timestamp = results['metadata']['timestamp'].replace(':', '-').replace(' ', '_')
        plot_path = os.path.join(args.output_dir, f'benchmark_plots_{timestamp}.png')
        create_plots(results, plot_path)
        
        # Also show the plot
        plt.show()


if __name__ == "__main__":
    main()