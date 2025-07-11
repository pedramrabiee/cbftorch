"""
Regression tests to ensure optimizations don't break existing functionality.
"""

import torch
import pickle
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from ..core.barrier_benchmarks import BarrierBenchmarks
from ..core.lie_deriv_benchmarks import LieDerivativeBenchmarks
from ..core.hocbf_benchmarks import HOCBFBenchmarks
from .correctness_tests import CorrectnessValidator


@dataclass
class RegressionBaseline:
    """Baseline results for regression testing."""
    version: str
    test_name: str
    results: Dict[str, Any]
    timestamp: str
    
    
class RegressionTester:
    """Tester for ensuring optimizations don't cause regressions."""
    
    def __init__(self, baseline_dir: str = None):
        """
        Initialize regression tester.
        
        Args:
            baseline_dir: Directory to store baseline results
        """
        self.baseline_dir = baseline_dir or "benchmarks/baselines"
        os.makedirs(self.baseline_dir, exist_ok=True)
        
        # Initialize validators and benchmarks
        self.correctness_validator = CorrectnessValidator()
        self.barrier_benchmarks = BarrierBenchmarks()
        self.lie_deriv_benchmarks = LieDerivativeBenchmarks()
        self.hocbf_benchmarks = HOCBFBenchmarks()
        
    def create_baseline(self, version: str = "original") -> Dict[str, Any]:
        """
        Create baseline results for regression testing.
        
        Args:
            version: Version identifier for this baseline
            
        Returns:
            Dictionary with all baseline results
        """
        print(f"Creating baseline for version: {version}")
        
        baseline_results = {}
        
        # Run correctness tests
        print("Running correctness tests...")
        baseline_results['correctness'] = self.correctness_validator.run_all_correctness_tests()
        
        # Run performance benchmarks with smaller test sizes for baseline
        print("Running barrier benchmarks...")
        baseline_results['barrier_benchmarks'] = self.barrier_benchmarks.run_full_benchmark_suite(
            batch_sizes=[1, 10, 50],
            num_runs=20
        )
        
        print("Running Lie derivative benchmarks...")
        baseline_results['lie_deriv_benchmarks'] = self.lie_deriv_benchmarks.run_full_benchmark_suite(
            batch_sizes=[1, 10, 50],
            num_runs=20
        )
        
        print("Running HOCBF benchmarks...")
        baseline_results['hocbf_benchmarks'] = self.hocbf_benchmarks.run_full_benchmark_suite(
            max_rel_deg=3,
            batch_sizes=[1, 10, 50],
            num_runs=20
        )
        
        # Save baseline
        baseline = RegressionBaseline(
            version=version,
            test_name="full_suite",
            results=baseline_results,
            timestamp=torch.datetime.now().isoformat()
        )
        
        self._save_baseline(baseline)
        print(f"Baseline saved for version {version}")
        
        return baseline_results
    
    def _save_baseline(self, baseline: RegressionBaseline):
        """Save baseline results to disk."""
        filename = f"{baseline.version}_{baseline.test_name}_baseline.pkl"
        filepath = os.path.join(self.baseline_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(baseline, f)
    
    def _load_baseline(self, version: str, test_name: str = "full_suite") -> Optional[RegressionBaseline]:
        """Load baseline results from disk."""
        filename = f"{version}_{test_name}_baseline.pkl"
        filepath = os.path.join(self.baseline_dir, filename)
        
        if not os.path.exists(filepath):
            return None
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def run_regression_test(
        self, 
        baseline_version: str = "original",
        current_version: str = "optimized"
    ) -> Dict[str, Any]:
        """
        Run regression test against baseline.
        
        Args:
            baseline_version: Version to compare against
            current_version: Current version being tested
            
        Returns:
            Dictionary with regression test results
        """
        print(f"Running regression test: {current_version} vs {baseline_version}")
        
        # Load baseline
        baseline = self._load_baseline(baseline_version)
        if baseline is None:
            raise ValueError(f"Baseline for version {baseline_version} not found. "
                           f"Create baseline first using create_baseline()")
        
        # Run current tests
        current_results = self.create_baseline(current_version)
        
        # Compare results
        regression_results = self._compare_with_baseline(
            baseline.results, 
            current_results,
            baseline_version,
            current_version
        )
        
        return regression_results
    
    def _compare_with_baseline(
        self,
        baseline_results: Dict[str, Any],
        current_results: Dict[str, Any],
        baseline_version: str,
        current_version: str
    ) -> Dict[str, Any]:
        """Compare current results with baseline."""
        regression_results = {
            'baseline_version': baseline_version,
            'current_version': current_version,
            'correctness_regression': self._check_correctness_regression(
                baseline_results.get('correctness', {}),
                current_results.get('correctness', {})
            ),
            'performance_regression': self._check_performance_regression(
                baseline_results,
                current_results
            ),
            'summary': {}
        }
        
        # Generate summary
        correctness_ok = regression_results['correctness_regression']['all_tests_passed']
        performance_ok = regression_results['performance_regression']['no_significant_regression']
        
        regression_results['summary'] = {
            'overall_status': 'PASS' if correctness_ok and performance_ok else 'FAIL',
            'correctness_status': 'PASS' if correctness_ok else 'FAIL',
            'performance_status': 'PASS' if performance_ok else 'FAIL'
        }
        
        return regression_results
    
    def _check_correctness_regression(
        self,
        baseline_correctness: Dict[str, Any],
        current_correctness: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check for correctness regressions."""
        regression_info = {
            'all_tests_passed': True,
            'failed_tests': [],
            'new_failures': [],
            'fixed_tests': []
        }
        
        # Compare test results category by category
        for category in baseline_correctness.keys():
            if category not in current_correctness:
                continue
                
            baseline_tests = baseline_correctness[category]
            current_tests = current_correctness[category]
            
            # Create lookup for current test results
            current_lookup = {test.test_name: test for test in current_tests}
            
            for baseline_test in baseline_tests:
                test_name = baseline_test.test_name
                
                if test_name in current_lookup:
                    current_test = current_lookup[test_name]
                    
                    # Check if test status changed
                    if baseline_test.passed and not current_test.passed:
                        regression_info['new_failures'].append({
                            'test_name': test_name,
                            'category': category,
                            'baseline_error': baseline_test.max_absolute_error,
                            'current_error': current_test.max_absolute_error
                        })
                        regression_info['all_tests_passed'] = False
                    
                    elif not baseline_test.passed and current_test.passed:
                        regression_info['fixed_tests'].append({
                            'test_name': test_name,
                            'category': category
                        })
                    
                    # Track all current failures
                    if not current_test.passed:
                        regression_info['failed_tests'].append({
                            'test_name': test_name,
                            'category': category,
                            'error': current_test.max_absolute_error
                        })
        
        return regression_info
    
    def _check_performance_regression(
        self,
        baseline_results: Dict[str, Any],
        current_results: Dict[str, Any],
        regression_threshold: float = 2.0  # 2x slower is considered regression
    ) -> Dict[str, Any]:
        """Check for performance regressions."""
        regression_info = {
            'no_significant_regression': True,
            'performance_regressions': [],
            'performance_improvements': [],
            'regression_threshold': regression_threshold
        }
        
        # Compare timing results from different benchmark categories
        benchmark_categories = ['barrier_benchmarks', 'lie_deriv_benchmarks', 'hocbf_benchmarks']
        
        for category in benchmark_categories:
            if category not in baseline_results or category not in current_results:
                continue
            
            baseline_bench = baseline_results[category]
            current_bench = current_results[category]
            
            # Extract timing information recursively
            baseline_timings = self._extract_timings(baseline_bench)
            current_timings = self._extract_timings(current_bench)
            
            # Compare timings
            for timing_key in baseline_timings.keys():
                if timing_key in current_timings:
                    baseline_time = baseline_timings[timing_key]
                    current_time = current_timings[timing_key]
                    
                    slowdown_factor = current_time / baseline_time
                    
                    if slowdown_factor > regression_threshold:
                        regression_info['performance_regressions'].append({
                            'category': category,
                            'test': timing_key,
                            'baseline_time': baseline_time,
                            'current_time': current_time,
                            'slowdown_factor': slowdown_factor
                        })
                        regression_info['no_significant_regression'] = False
                    
                    elif slowdown_factor < 0.9:  # More than 10% improvement
                        regression_info['performance_improvements'].append({
                            'category': category,
                            'test': timing_key,
                            'baseline_time': baseline_time,
                            'current_time': current_time,
                            'speedup_factor': baseline_time / current_time
                        })
        
        return regression_info
    
    def _extract_timings(self, benchmark_results: Dict[str, Any], prefix: str = "") -> Dict[str, float]:
        """Recursively extract timing information from benchmark results."""
        timings = {}
        
        for key, value in benchmark_results.items():
            current_prefix = f"{prefix}.{key}" if prefix else key
            
            if hasattr(value, 'mean_time'):
                # This is a TimingResult object
                timings[current_prefix] = value.mean_time
            elif isinstance(value, dict):
                # Recurse into nested dictionary
                nested_timings = self._extract_timings(value, current_prefix)
                timings.update(nested_timings)
        
        return timings
    
    def print_regression_summary(self, regression_results: Dict[str, Any]):
        """Print a summary of regression test results."""
        print("\n" + "="*60)
        print("REGRESSION TEST RESULTS")
        print("="*60)
        
        summary = regression_results['summary']
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Correctness: {summary['correctness_status']}")
        print(f"Performance: {summary['performance_status']}")
        
        # Correctness details
        correctness = regression_results['correctness_regression']
        print(f"\nCorrectness Details:")
        print(f"  New Failures: {len(correctness['new_failures'])}")
        print(f"  Fixed Tests: {len(correctness['fixed_tests'])}")
        print(f"  Total Failed: {len(correctness['failed_tests'])}")
        
        if correctness['new_failures']:
            print("  New Failure Details:")
            for failure in correctness['new_failures']:
                print(f"    - {failure['test_name']} ({failure['category']})")
        
        # Performance details
        performance = regression_results['performance_regression']
        print(f"\nPerformance Details:")
        print(f"  Regressions: {len(performance['performance_regressions'])}")
        print(f"  Improvements: {len(performance['performance_improvements'])}")
        
        if performance['performance_regressions']:
            print("  Regression Details:")
            for regression in performance['performance_regressions']:
                print(f"    - {regression['test']} ({regression['category']}): "
                      f"{regression['slowdown_factor']:.2f}x slower")
        
        if performance['performance_improvements']:
            print("  Improvement Details:")
            for improvement in performance['performance_improvements'][:5]:  # Show top 5
                print(f"    - {improvement['test']} ({improvement['category']}): "
                      f"{improvement['speedup_factor']:.2f}x faster")
        
        print("="*60)