#!/usr/bin/env python3
"""
Results analysis and visualization tool for CBFtorch benchmarks.

This tool helps analyze saved benchmark results and create detailed reports.
"""

import os
import json
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
# import seaborn as sns  # Optional dependency


class BenchmarkAnalyzer:
    """Analyzer for benchmark results."""
    
    def __init__(self, results_dir: str = "benchmarks/results"):
        """Initialize analyzer with results directory."""
        self.results_dir = results_dir
        self.results = {}
        
    def load_results(self, pattern: str = "*_results_*.json") -> Dict[str, Any]:
        """Load all result files matching pattern."""
        result_files = glob.glob(os.path.join(self.results_dir, pattern))
        
        for filepath in result_files:
            filename = os.path.basename(filepath)
            print(f"Loading: {filename}")
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.results[filename] = data
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"Loaded {len(self.results)} result files")
        return self.results
    
    def extract_timing_data(self) -> pd.DataFrame:
        """Extract timing data into a pandas DataFrame for analysis."""
        timing_data = []
        
        for filename, results in self.results.items():
            # Determine test type
            if 'performance' in filename:
                # Extract from performance results
                timing_data.extend(self._extract_performance_timings(results, filename))
            elif 'baseline' in filename:
                # Extract from baseline results
                timing_data.extend(self._extract_baseline_timings(results, filename))
        
        return pd.DataFrame(timing_data)
    
    def _extract_performance_timings(self, results: Dict[str, Any], source: str) -> List[Dict]:
        """Extract timing data from performance benchmark results."""
        timings = []
        
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                timings.extend(self._extract_timings_recursive(
                    category_results, category, source, prefix=""
                ))
        
        return timings
    
    def _extract_baseline_timings(self, results: Dict[str, Any], source: str) -> List[Dict]:
        """Extract timing data from baseline results."""
        timings = []
        
        for category, category_results in results.items():
            if isinstance(category_results, dict):
                for test_name, test_data in category_results.items():
                    if isinstance(test_data, dict) and 'mean_time' in test_data:
                        timings.append({
                            'test_name': test_name,
                            'category': category,
                            'mean_time': test_data['mean_time'],
                            'std_time': test_data.get('std_time', 0),
                            'batch_size': test_data.get('batch_size', 1),
                            'rel_deg': test_data.get('rel_deg', 1),
                            'memory_mb': test_data.get('memory_mb', 0),
                            'source': source
                        })
        
        return timings
    
    def _extract_timings_recursive(
        self, 
        data: Dict[str, Any], 
        category: str, 
        source: str, 
        prefix: str
    ) -> List[Dict]:
        """Recursively extract timing data from nested dictionaries."""
        timings = []
        
        for key, value in data.items():
            current_prefix = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                # Check if this is a timing result object
                if 'mean_time' in value:
                    timings.append({
                        'test_name': current_prefix,
                        'category': category,
                        'mean_time': value['mean_time'],
                        'std_time': value.get('std_time', 0),
                        'min_time': value.get('min_time', 0),
                        'max_time': value.get('max_time', 0),
                        'num_runs': value.get('num_runs', 1),
                        'source': source
                    })
                else:
                    # Recurse into nested dictionary
                    timings.extend(self._extract_timings_recursive(
                        value, category, source, current_prefix
                    ))
        
        return timings
    
    def create_performance_summary(self) -> str:
        """Create a text summary of performance results."""
        df = self.extract_timing_data()
        
        if df.empty:
            return "No timing data found in results."
        
        summary = []
        summary.append("BENCHMARK PERFORMANCE SUMMARY")
        summary.append("=" * 50)
        
        # Overall statistics
        summary.append(f"Total tests: {len(df)}")
        summary.append(f"Average execution time: {df['mean_time'].mean():.6f}s")
        summary.append(f"Slowest test: {df['mean_time'].max():.6f}s")
        summary.append(f"Fastest test: {df['mean_time'].min():.6f}s")
        summary.append("")
        
        # By category
        summary.append("BY CATEGORY:")
        summary.append("-" * 20)
        
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            summary.append(f"{category}:")
            summary.append(f"  Tests: {len(cat_data)}")
            summary.append(f"  Avg time: {cat_data['mean_time'].mean():.6f}s")
            summary.append(f"  Max time: {cat_data['mean_time'].max():.6f}s")
            summary.append("")
        
        # Slowest tests
        summary.append("SLOWEST TESTS:")
        summary.append("-" * 20)
        slowest = df.nlargest(10, 'mean_time')
        for _, row in slowest.iterrows():
            summary.append(f"{row['test_name']}: {row['mean_time']:.6f}s ({row['category']})")
        
        return "\n".join(summary)
    
    def plot_performance_comparison(self, save_path: str = None):
        """Create performance comparison plots."""
        df = self.extract_timing_data()
        
        if df.empty:
            print("No data to plot")
            return
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CBFtorch Benchmark Results', fontsize=16)
        
        # Plot 1: Performance by category
        if 'category' in df.columns:
            category_means = df.groupby('category')['mean_time'].mean().sort_values(ascending=False)
            axes[0, 0].bar(range(len(category_means)), category_means.values)
            axes[0, 0].set_xticks(range(len(category_means)))
            axes[0, 0].set_xticklabels(category_means.index, rotation=45, ha='right')
            axes[0, 0].set_ylabel('Mean Time (s)')
            axes[0, 0].set_title('Average Performance by Category')
        
        # Plot 2: Batch size scaling (if available)
        if 'batch_size' in df.columns and df['batch_size'].nunique() > 1:
            batch_data = df.groupby('batch_size')['mean_time'].mean()
            axes[0, 1].plot(batch_data.index, batch_data.values, 'o-')
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Mean Time (s)')
            axes[0, 1].set_title('Performance vs Batch Size')
            axes[0, 1].set_xscale('log')
        else:
            axes[0, 1].text(0.5, 0.5, 'No batch size data', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Batch Size Scaling (No Data)')
        
        # Plot 3: Relative degree scaling (if available)
        if 'rel_deg' in df.columns and df['rel_deg'].nunique() > 1:
            rel_deg_data = df.groupby('rel_deg')['mean_time'].mean()
            axes[1, 0].bar(rel_deg_data.index, rel_deg_data.values)
            axes[1, 0].set_xlabel('Relative Degree')
            axes[1, 0].set_ylabel('Mean Time (s)')
            axes[1, 0].set_title('Performance vs Relative Degree')
        else:
            axes[1, 0].text(0.5, 0.5, 'No relative degree data', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Relative Degree Scaling (No Data)')
        
        # Plot 4: Distribution of execution times
        axes[1, 1].hist(df['mean_time'], bins=30, alpha=0.7)
        axes[1, 1].set_xlabel('Mean Time (s)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Execution Times')
        axes[1, 1].set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        return fig
    
    def create_detailed_report(self, output_file: str = None) -> str:
        """Create a detailed HTML report."""
        df = self.extract_timing_data()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>CBFtorch Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                         background-color: #e7f3ff; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>CBFtorch Benchmark Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <div class="metric"><strong>Total Tests:</strong> {len(df)}</div>
                <div class="metric"><strong>Avg Time:</strong> {df['mean_time'].mean():.6f}s</div>
                <div class="metric"><strong>Max Time:</strong> {df['mean_time'].max():.6f}s</div>
                <div class="metric"><strong>Min Time:</strong> {df['mean_time'].min():.6f}s</div>
            </div>
            
            <h2>Performance by Category</h2>
            {self._create_category_table(df)}
            
            <h2>Detailed Results</h2>
            {df.to_html(index=False, float_format='{:.6f}'.format)}
            
            <h2>Analysis</h2>
            <pre>{self.create_performance_summary()}</pre>
        </body>
        </html>
        """
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(html_content)
            print(f"Report saved to: {output_file}")
        
        return html_content
    
    def _create_category_table(self, df: pd.DataFrame) -> str:
        """Create HTML table for category performance."""
        if df.empty or 'category' not in df.columns:
            return "<p>No category data available</p>"
        
        category_stats = df.groupby('category').agg({
            'mean_time': ['count', 'mean', 'std', 'min', 'max']
        }).round(6)
        
        return category_stats.to_html()
    
    def compare_results(self, baseline_file: str, optimized_file: str) -> Dict[str, Any]:
        """Compare baseline vs optimized results."""
        comparison = {}
        
        # Load specific files
        with open(os.path.join(self.results_dir, baseline_file), 'r') as f:
            baseline = json.load(f)
        
        with open(os.path.join(self.results_dir, optimized_file), 'r') as f:
            optimized = json.load(f)
        
        # Extract timing data for both
        baseline_df = pd.DataFrame(self._extract_baseline_timings(baseline, "baseline"))
        optimized_df = pd.DataFrame(self._extract_baseline_timings(optimized, "optimized"))
        
        # Merge on test_name and category
        merged = pd.merge(baseline_df, optimized_df, 
                         on=['test_name', 'category'], 
                         suffixes=('_baseline', '_optimized'))
        
        # Calculate speedup
        merged['speedup'] = merged['mean_time_baseline'] / merged['mean_time_optimized']
        
        comparison['summary'] = {
            'average_speedup': merged['speedup'].mean(),
            'max_speedup': merged['speedup'].max(),
            'min_speedup': merged['speedup'].min(),
            'tests_improved': (merged['speedup'] > 1.0).sum(),
            'tests_regressed': (merged['speedup'] < 1.0).sum(),
            'total_tests': len(merged)
        }
        
        comparison['detailed'] = merged.to_dict('records')
        
        return comparison


def main():
    """Main entry point for results analyzer."""
    parser = argparse.ArgumentParser(description="Analyze CBFtorch benchmark results")
    
    parser.add_argument('--results-dir', default='benchmarks/results',
                       help='Directory containing result files')
    parser.add_argument('--summary', action='store_true',
                       help='Print performance summary')
    parser.add_argument('--plot', action='store_true',
                       help='Create performance plots')
    parser.add_argument('--report', action='store_true',
                       help='Generate HTML report')
    parser.add_argument('--compare', nargs=2, metavar=('baseline', 'optimized'),
                       help='Compare two result files')
    parser.add_argument('--output-dir', default='benchmarks/analysis',
                       help='Output directory for generated files')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = BenchmarkAnalyzer(args.results_dir)
    analyzer.load_results()
    
    if args.summary:
        print(analyzer.create_performance_summary())
    
    if args.plot:
        plot_path = os.path.join(args.output_dir, 'performance_plots.png')
        analyzer.plot_performance_comparison(plot_path)
    
    if args.report:
        report_path = os.path.join(args.output_dir, 'benchmark_report.html')
        analyzer.create_detailed_report(report_path)
    
    if args.compare:
        baseline_file, optimized_file = args.compare
        comparison = analyzer.compare_results(baseline_file, optimized_file)
        
        print("COMPARISON RESULTS:")
        print("=" * 40)
        print(f"Average speedup: {comparison['summary']['average_speedup']:.2f}x")
        print(f"Max speedup: {comparison['summary']['max_speedup']:.2f}x")
        print(f"Tests improved: {comparison['summary']['tests_improved']}")
        print(f"Tests regressed: {comparison['summary']['tests_regressed']}")
        
        # Save detailed comparison
        comparison_path = os.path.join(args.output_dir, 'comparison_results.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"Detailed comparison saved to: {comparison_path}")


if __name__ == "__main__":
    main()