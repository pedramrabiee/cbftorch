#!/usr/bin/env python3
"""
Main benchmark runner for CBFtorch optimization testing.

This script provides a unified interface for running all benchmark tests,
including correctness validation, performance measurement, and regression testing.
"""

import os
import sys
import argparse
import json
import datetime
from pathlib import Path

# Add cbftorch to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.configs.benchmark_configs import (
    get_default_config, 
    get_quick_test_config, 
    get_comprehensive_config,
    get_optimization_config
)
from benchmarks.tests.correctness_tests import CorrectnessValidator
from benchmarks.tests.regression_tests import RegressionTester
from benchmarks.tests.numerical_stability import NumericalStabilityTester
from benchmarks.core.barrier_benchmarks import BarrierBenchmarks
from benchmarks.core.lie_deriv_benchmarks import LieDerivativeBenchmarks
from benchmarks.core.hocbf_benchmarks import HOCBFBenchmarks
from benchmarks.core.composite_benchmarks import CompositeBenchmarks


class BenchmarkRunner:
    """Main benchmark runner class."""
    
    def __init__(self, config):
        """Initialize benchmark runner with configuration."""
        self.config = config
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize test components
        self.correctness_validator = CorrectnessValidator(
            tolerance={
                'absolute': self.config.absolute_tolerance,
                'relative': self.config.relative_tolerance
            }
        )
        self.stability_tester = NumericalStabilityTester()
        self.regression_tester = RegressionTester(
            baseline_dir=os.path.join(self.config.output_dir, "baselines")
        )
        
        # Initialize benchmark components
        self.barrier_benchmarks = BarrierBenchmarks()
        self.lie_deriv_benchmarks = LieDerivativeBenchmarks()
        self.hocbf_benchmarks = HOCBFBenchmarks()
        self.composite_benchmarks = CompositeBenchmarks()
        
    def run_correctness_tests(self) -> dict:
        """Run correctness validation tests."""
        print("="*60)
        print("RUNNING CORRECTNESS TESTS")
        print("="*60)
        
        results = self.correctness_validator.run_all_correctness_tests()
        self.correctness_validator.print_results_summary(results)
        
        return results
    
    def run_stability_tests(self) -> dict:
        """Run numerical stability tests."""
        print("\n" + "="*60)
        print("RUNNING NUMERICAL STABILITY TESTS")
        print("="*60)
        
        results = self.stability_tester.run_all_stability_tests()
        self.stability_tester.print_stability_summary(results)
        
        return results
    
    def run_performance_tests(self) -> dict:
        """Run performance benchmark tests."""
        print("\n" + "="*60)
        print("RUNNING PERFORMANCE BENCHMARKS")
        print("="*60)
        
        results = {}
        
        print("Running barrier benchmarks...")
        results['barrier_benchmarks'] = self.barrier_benchmarks.run_full_benchmark_suite(
            batch_sizes=self.config.batch_sizes,
            num_runs=self.config.num_runs
        )
        
        print("Running Lie derivative benchmarks...")
        results['lie_deriv_benchmarks'] = self.lie_deriv_benchmarks.run_full_benchmark_suite(
            batch_sizes=self.config.batch_sizes,
            num_runs=self.config.num_runs
        )
        
        print("Running HOCBF benchmarks...")
        results['hocbf_benchmarks'] = self.hocbf_benchmarks.run_full_benchmark_suite(
            max_rel_deg=self.config.max_rel_deg,
            batch_sizes=self.config.batch_sizes,
            num_runs=self.config.num_runs
        )
        
        print("Running composite barrier benchmarks...")
        results['composite_benchmarks'] = self.composite_benchmarks.run_full_benchmark_suite(
            max_num_barriers=self.config.max_num_barriers,
            batch_sizes=self.config.batch_sizes,
            num_runs=self.config.num_runs
        )
        
        print("Performance benchmarks completed!")
        return results
    
    def run_regression_tests(self, baseline_version: str = "original") -> dict:
        """Run regression tests against baseline."""
        print("\n" + "="*60)
        print("RUNNING REGRESSION TESTS")
        print("="*60)
        
        try:
            results = self.regression_tester.run_regression_test(
                baseline_version=baseline_version,
                current_version="current"
            )
            self.regression_tester.print_regression_summary(results)
            return results
        except ValueError as e:
            print(f"Regression test failed: {e}")
            print("Creating baseline first...")
            self.regression_tester.create_baseline(baseline_version)
            print("Baseline created. Re-run to perform regression testing.")
            return {}
    
    def create_baseline(self, version: str = "original") -> dict:
        """Create baseline measurements."""
        print("\n" + "="*60)
        print(f"CREATING BASELINE FOR VERSION: {version}")
        print("="*60)
        
        return self.regression_tester.create_baseline(version)
    
    def save_results(self, results: dict, test_type: str):
        """Save results to JSON file."""
        if not self.config.save_results:
            return
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_type}_results_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Convert results to JSON-serializable format
        json_results = self._convert_to_json_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        # Handle torch tensors and numpy arrays
        if hasattr(obj, 'detach'):  # PyTorch tensor
            if obj.numel() == 1:
                return obj.item()
            else:
                return obj.detach().cpu().numpy().tolist()
        elif hasattr(obj, 'tolist'):  # NumPy array
            return obj.tolist()
        elif hasattr(obj, 'item') and hasattr(obj, 'numel'):  # PyTorch scalar
            return obj.item()
        
        # Handle custom dataclasses and objects with __dict__
        elif hasattr(obj, '__dict__'):
            return {key: self._convert_to_json_serializable(value) 
                   for key, value in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {key: self._convert_to_json_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_json_serializable(item) for item in obj]
        
        # Handle primitive types and objects that can't be serialized
        try:
            json.dumps(obj)  # Test if it's already JSON serializable
            return obj
        except (TypeError, ValueError):
            # Fallback: convert to string representation
            return str(obj)
    
    def run_full_suite(self, baseline_version: str = None) -> dict:
        """Run complete benchmark suite."""
        all_results = {}
        
        # Run correctness tests
        if self.config.run_correctness_tests:
            correctness_results = self.run_correctness_tests()
            all_results['correctness'] = correctness_results
            self.save_results(correctness_results, 'correctness')
        
        # Run stability tests
        if self.config.run_stability_tests:
            stability_results = self.run_stability_tests()
            all_results['stability'] = stability_results
            self.save_results(stability_results, 'stability')
        
        # Run performance tests
        if self.config.run_performance_tests:
            performance_results = self.run_performance_tests()
            all_results['performance'] = performance_results
            self.save_results(performance_results, 'performance')
        
        # Run regression tests
        if self.config.run_regression_tests and baseline_version:
            regression_results = self.run_regression_tests(baseline_version)
            all_results['regression'] = regression_results
            self.save_results(regression_results, 'regression')
        
        # Print summary
        self._print_final_summary(all_results)
        
        return all_results
    
    def _print_final_summary(self, results: dict):
        """Print final summary of all test results."""
        print("\n" + "="*60)
        print("FINAL BENCHMARK SUMMARY")
        print("="*60)
        
        if 'correctness' in results:
            # Count correctness results
            total_correctness = 0
            passed_correctness = 0
            
            for category, tests in results['correctness'].items():
                for test in tests:
                    total_correctness += 1
                    if test.passed:
                        passed_correctness += 1
            
            print(f"Correctness Tests: {passed_correctness}/{total_correctness} passed")
        
        if 'stability' in results:
            # Count stability results
            total_stability = 0
            stable_tests = 0
            
            for category, tests in results['stability'].items():
                for test in tests:
                    total_stability += 1
                    if test.stability_score > 0.7:
                        stable_tests += 1
            
            print(f"Stability Tests: {stable_tests}/{total_stability} stable")
        
        if 'performance' in results:
            print("Performance Tests: Completed")
        
        if 'regression' in results:
            if results['regression']:
                status = results['regression']['summary']['overall_status']
                print(f"Regression Tests: {status}")
            else:
                print("Regression Tests: Baseline created")
        
        print("="*60)


def main():
    """Main entry point for benchmark runner."""
    parser = argparse.ArgumentParser(description="CBFtorch Benchmark Runner")
    
    parser.add_argument(
        '--config', 
        choices=['default', 'quick', 'comprehensive', 'optimization'],
        default='default',
        help='Benchmark configuration to use'
    )
    
    parser.add_argument(
        '--create-baseline',
        action='store_true',
        help='Create baseline measurements'
    )
    
    parser.add_argument(
        '--baseline-version',
        default='original',
        help='Version name for baseline'
    )
    
    parser.add_argument(
        '--output-dir',
        default='benchmarks/results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--no-correctness',
        action='store_true',
        help='Skip correctness tests'
    )
    
    parser.add_argument(
        '--no-performance',
        action='store_true',
        help='Skip performance tests'
    )
    
    parser.add_argument(
        '--no-stability',
        action='store_true',
        help='Skip stability tests'
    )
    
    parser.add_argument(
        '--enable-regression',
        action='store_true',
        help='Enable regression testing'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_map = {
        'default': get_default_config,
        'quick': get_quick_test_config,
        'comprehensive': get_comprehensive_config,
        'optimization': get_optimization_config
    }
    
    config = config_map[args.config]()
    
    # Override config with command line arguments
    config.output_dir = args.output_dir
    config.run_correctness_tests = not args.no_correctness
    config.run_performance_tests = not args.no_performance
    config.run_stability_tests = not args.no_stability
    config.run_regression_tests = args.enable_regression
    
    # Initialize runner
    runner = BenchmarkRunner(config)
    
    # Create baseline if requested
    if args.create_baseline:
        runner.create_baseline(args.baseline_version)
        return
    
    # Run full benchmark suite
    results = runner.run_full_suite(
        baseline_version=args.baseline_version if args.enable_regression else None
    )
    
    print(f"\nBenchmark run completed. Results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()