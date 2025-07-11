#!/usr/bin/env python3
"""
Benchmark coverage validation script.
This script checks if our benchmarks cover all the key optimization targets identified.
"""

import json
import sys
from pathlib import Path

# Add cbftorch to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class BenchmarkCoverageValidator:
    """Validates that benchmarks cover all optimization targets."""
    
    def __init__(self):
        self.optimization_targets = {
            'lie_derivative_computation': {
                'description': 'Redundant gradient computations in Lie derivatives',
                'impact': 'HIGH',
                'requirements': [
                    'Compare combined vs separate gradient computations',
                    'Test with different batch sizes',
                    'Measure memory usage during gradient computation',
                    'Test alternative implementation methods'
                ]
            },
            'higher_order_cbf_generation': {
                'description': 'Lambda function closure overhead in HOCBF generation',
                'impact': 'HIGH', 
                'requirements': [
                    'Test HOCBF generation for different relative degrees',
                    'Measure evaluation time vs generation time',
                    'Test memory usage during HOCBF creation',
                    'Compare nested lambda vs direct computation'
                ]
            },
            'multiple_barriers_scaling': {
                'description': 'Composite barrier function performance with many barriers',
                'impact': 'HIGH',
                'requirements': [
                    'Test with 1, 5, 10, 20+ barriers',
                    'Measure per-barrier computation time',
                    'Test different composition rules (intersection/union)',
                    'Memory usage scaling'
                ]
            },
            'gradient_computation_redundancy': {
                'description': 'Multiple gradient computations for same functions',
                'impact': 'HIGH',
                'requirements': [
                    'Test repeated gradient computations',
                    'Measure caching opportunities',
                    'Compare get_hocbf_and_lie_derivs vs separate calls',
                    'Test with simulation-like repeated calls'
                ]
            },
            'qp_control_performance': {
                'description': 'QP solver and matrix assembly overhead',
                'impact': 'MEDIUM',
                'requirements': [
                    'Test QP solving time vs batch size',
                    'Measure matrix assembly overhead', 
                    'Test with different numbers of constraints',
                    'Memory usage during QP solving'
                ]
            },
            'memory_allocation_patterns': {
                'description': 'Tensor creation and memory fragmentation',
                'impact': 'MEDIUM',
                'requirements': [
                    'Track memory usage over long simulations',
                    'Measure memory allocation/deallocation patterns',
                    'Test memory growth with problem size',
                    'Identify memory leaks'
                ]
            },
            'state_dimension_scaling': {
                'description': 'Performance scaling with state dimension',
                'impact': 'MEDIUM',
                'requirements': [
                    'Test with different state dimensions (2D, 4D, 6D, 8D+)',
                    'Measure computation time per state dimension',
                    'Test with different dynamics types',
                    'Memory usage scaling'
                ]
            },
            'batch_processing_efficiency': {
                'description': 'Vectorization and batch processing effectiveness',
                'impact': 'MEDIUM',
                'requirements': [
                    'Test batch sizes from 1 to 500+',
                    'Measure per-sample computation time',
                    'Calculate vectorization efficiency',
                    'Memory usage per sample'
                ]
            }
        }
    
    def validate_simple_benchmark(self, results_file: str) -> dict:
        """Validate simple benchmark coverage."""
        print("Validating simple_benchmark.py coverage...")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        coverage = {}
        
        # Check what's covered
        coverage['lie_derivative_computation'] = {
            'covered': 'lie_derivatives' in results,
            'metrics': ['lf_mean_time', 'lg_mean_time'] if 'lie_derivatives' in results else [],
            'missing': ['combined vs separate comparison', 'repeated computation patterns']
        }
        
        coverage['higher_order_cbf_generation'] = {
            'covered': 'hocbf_evaluation' in results,
            'metrics': ['mean_time', 'overhead_vs_barrier'] if 'hocbf_evaluation' in results else [],
            'missing': ['generation time vs evaluation time', 'lambda closure overhead']
        }
        
        coverage['multiple_barriers_scaling'] = {
            'covered': False,
            'metrics': [],
            'missing': ['composite barriers', 'multiple barrier performance']
        }
        
        coverage['gradient_computation_redundancy'] = {
            'covered': False,
            'metrics': [],
            'missing': ['redundancy comparison', 'caching opportunities']
        }
        
        coverage['qp_control_performance'] = {
            'covered': False,
            'metrics': [],
            'missing': ['QP solving', 'matrix assembly', 'constraint scaling']
        }
        
        coverage['batch_processing_efficiency'] = {
            'covered': True,
            'metrics': ['time_per_sample', 'batch_size scaling'],
            'missing': []
        }
        
        coverage['state_dimension_scaling'] = {
            'covered': False,
            'metrics': [],
            'missing': ['multiple dynamics types', 'state dimension comparison']
        }
        
        return coverage
    
    def validate_comprehensive_benchmark(self, results_file: str) -> dict:
        """Validate comprehensive benchmark coverage."""
        print("Validating comprehensive_benchmark.py coverage...")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        coverage = {}
        
        # Check comprehensive coverage
        coverage['lie_derivative_computation'] = {
            'covered': 'gradient_redundancy' in results,
            'metrics': ['combined_time', 'separate_time', 'redundancy_overhead'],
            'missing': []
        }
        
        coverage['multiple_barriers_scaling'] = {
            'covered': 'multiple_barriers_scaling' in results,
            'metrics': ['time_per_barrier', 'scaling_efficiency'],
            'missing': []
        }
        
        coverage['gradient_computation_redundancy'] = {
            'covered': 'repeated_computations' in results,
            'metrics': ['efficiency_gain', 'time_per_step_combined'],
            'missing': []
        }
        
        coverage['qp_control_performance'] = {
            'covered': 'qp_control_performance' in results,
            'metrics': ['qp_solve_time', 'time_per_sample'],
            'missing': []
        }
        
        coverage['state_dimension_scaling'] = {
            'covered': 'state_dimension_scaling' in results,
            'metrics': ['time_per_state_dim', 'state_dim'],
            'missing': []
        }
        
        return coverage
    
    def print_coverage_report(self, simple_coverage: dict, comprehensive_coverage: dict):
        """Print detailed coverage report."""
        print("\n" + "="*60)
        print("BENCHMARK COVERAGE VALIDATION REPORT")
        print("="*60)
        
        all_targets = set(self.optimization_targets.keys())
        simple_covered = set(k for k, v in simple_coverage.items() if v['covered'])
        comprehensive_covered = set(k for k, v in comprehensive_coverage.items() if v['covered'])
        
        print(f"Optimization targets identified: {len(all_targets)}")
        print(f"Simple benchmark coverage: {len(simple_covered)}/{len(all_targets)}")
        print(f"Comprehensive benchmark coverage: {len(comprehensive_covered)}/{len(all_targets)}")
        print(f"Total coverage: {len(simple_covered | comprehensive_covered)}/{len(all_targets)}")
        print()
        
        print("COVERAGE BY TARGET:")
        print("-" * 40)
        
        for target, info in self.optimization_targets.items():
            simple_status = "✅" if simple_coverage.get(target, {}).get('covered', False) else "❌"
            comp_status = "✅" if comprehensive_coverage.get(target, {}).get('covered', False) else "❌"
            overall_status = "✅" if (simple_status == "✅" or comp_status == "✅") else "❌"
            
            print(f"{overall_status} {target}")
            print(f"   Impact: {info['impact']}")
            print(f"   Simple: {simple_status} | Comprehensive: {comp_status}")
            
            # Show missing requirements
            simple_missing = simple_coverage.get(target, {}).get('missing', [])
            comp_missing = comprehensive_coverage.get(target, {}).get('missing', [])
            all_missing = list(set(simple_missing + comp_missing))
            
            if all_missing:
                print(f"   Missing: {', '.join(all_missing[:2])}{'...' if len(all_missing) > 2 else ''}")
            print()
        
        # Critical gaps
        uncovered = all_targets - (simple_covered | comprehensive_covered)
        if uncovered:
            print("🚨 CRITICAL GAPS:")
            print("-" * 20)
            for target in uncovered:
                if self.optimization_targets[target]['impact'] == 'HIGH':
                    print(f"❌ HIGH IMPACT: {target}")
            print()
        
        # Recommendations
        print("RECOMMENDATIONS:")
        print("-" * 20)
        
        if len(simple_covered | comprehensive_covered) < len(all_targets):
            print("1. Run both simple_benchmark.py AND comprehensive_benchmark.py")
            print("2. Focus on HIGH impact targets for optimization")
            if uncovered:
                print("3. Add benchmarks for uncovered targets:")
                for target in uncovered:
                    print(f"   - {target}")
        else:
            print("✅ Complete coverage achieved!")
            print("1. Use simple_benchmark.py for quick development testing")
            print("2. Use comprehensive_benchmark.py for detailed optimization analysis")
            print("3. Ready to implement optimizations with confidence!")
        
        print("="*60)
    
    def check_results_quality(self, results_file: str):
        """Check if benchmark results have sufficient quality/detail."""
        print(f"\nValidating result quality in {results_file}...")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        quality_checks = {
            'has_metadata': 'metadata' in results,
            'has_timing_stats': False,
            'has_memory_tracking': False,
            'has_batch_scaling': False,
            'sufficient_runs': False
        }
        
        # Check for detailed timing statistics
        for category in results.values():
            if isinstance(category, dict):
                for test_data in category.values():
                    if isinstance(test_data, dict):
                        if all(k in test_data for k in ['mean_time', 'std_time', 'num_runs']):
                            quality_checks['has_timing_stats'] = True
                        if 'memory_delta_mb' in test_data:
                            quality_checks['has_memory_tracking'] = True
                        if 'batch_size' in test_data:
                            quality_checks['has_batch_scaling'] = True
                        if test_data.get('num_runs', 0) >= 20:
                            quality_checks['sufficient_runs'] = True
        
        print("Quality checks:")
        for check, passed in quality_checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}")
        
        return quality_checks


def main():
    """Main validation function."""
    import glob
    
    results_dir = "benchmarks/results"
    
    # Find latest results files
    simple_results = glob.glob(f"{results_dir}/performance_results_*.json")
    comprehensive_results = glob.glob(f"{results_dir}/comprehensive_benchmark_*.json")
    
    validator = BenchmarkCoverageValidator()
    
    simple_coverage = {}
    comprehensive_coverage = {}
    
    if simple_results:
        latest_simple = max(simple_results)
        print(f"Found simple benchmark results: {latest_simple}")
        simple_coverage = validator.validate_simple_benchmark(latest_simple)
        validator.check_results_quality(latest_simple)
    else:
        print("❌ No simple benchmark results found!")
    
    if comprehensive_results:
        latest_comprehensive = max(comprehensive_results)
        print(f"Found comprehensive benchmark results: {latest_comprehensive}")
        comprehensive_coverage = validator.validate_comprehensive_benchmark(latest_comprehensive)
        validator.check_results_quality(latest_comprehensive)
    else:
        print("❌ No comprehensive benchmark results found!")
        print("   Run: python benchmarks/comprehensive_benchmark.py")
    
    # Generate coverage report
    validator.print_coverage_report(simple_coverage, comprehensive_coverage)


if __name__ == "__main__":
    main()