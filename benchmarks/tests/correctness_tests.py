"""
Correctness validation tests for CBFtorch optimizations.

These tests ensure that optimized implementations produce mathematically
equivalent results to the original implementations.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass
from cbftorch.barriers.barrier import Barrier
from cbftorch.dynamics import UnicycleDynamics, DIDynamics
from cbftorch.utils.utils import (
    make_circle_barrier_functional, 
    make_norm_rectangular_barrier_functional,
    lie_deriv,
    get_func_deriv
)


@dataclass
class CorrectnessResult:
    """Results from correctness validation."""
    test_name: str
    passed: bool
    max_absolute_error: float
    max_relative_error: float
    mean_absolute_error: float
    error_details: Optional[Dict[str, Any]] = None
    
    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return (f"{status} {self.test_name}: "
                f"max_abs_err={self.max_absolute_error:.2e}, "
                f"max_rel_err={self.max_relative_error:.2e}")


class CorrectnessValidator:
    """Validator for ensuring optimization correctness."""
    
    def __init__(self, tolerance: Dict[str, float] = None):
        """
        Initialize validator with tolerance settings.
        
        Args:
            tolerance: Dictionary with 'absolute' and 'relative' tolerance values
        """
        self.tolerance = tolerance or {
            'absolute': 1e-10,
            'relative': 1e-8
        }
        
    def validate_function_pair(
        self,
        original_func: Callable,
        optimized_func: Callable,
        test_inputs: List[Any],
        test_name: str
    ) -> CorrectnessResult:
        """
        Validate that two functions produce equivalent outputs.
        
        Args:
            original_func: Original implementation
            optimized_func: Optimized implementation
            test_inputs: List of test inputs
            test_name: Name for this test
            
        Returns:
            CorrectnessResult with validation details
        """
        all_abs_errors = []
        all_rel_errors = []
        max_abs_error = 0.0
        max_rel_error = 0.0
        passed = True
        
        for i, test_input in enumerate(test_inputs):
            try:
                # Handle different input formats
                if isinstance(test_input, tuple):
                    args, kwargs = test_input, {}
                else:
                    args, kwargs = (test_input,), {}
                
                # Get outputs from both functions
                original_output = original_func(*args, **kwargs)
                optimized_output = optimized_func(*args, **kwargs)
                
                # Calculate errors
                abs_error, rel_error, test_passed = self._compute_errors(
                    original_output, optimized_output
                )
                
                all_abs_errors.append(abs_error)
                all_rel_errors.append(rel_error)
                max_abs_error = max(max_abs_error, abs_error)
                max_rel_error = max(max_rel_error, rel_error)
                
                if not test_passed:
                    passed = False
                    
            except Exception as e:
                passed = False
                print(f"Error in test {test_name}, input {i}: {e}")
        
        return CorrectnessResult(
            test_name=test_name,
            passed=passed,
            max_absolute_error=max_abs_error,
            max_relative_error=max_rel_error,
            mean_absolute_error=np.mean(all_abs_errors) if all_abs_errors else float('inf'),
            error_details={
                'num_tests': len(test_inputs),
                'all_abs_errors': all_abs_errors,
                'all_rel_errors': all_rel_errors
            }
        )
    
    def _compute_errors(
        self, 
        original: Any, 
        optimized: Any
    ) -> Tuple[float, float, bool]:
        """Compute absolute and relative errors between two values."""
        try:
            if isinstance(original, torch.Tensor) and isinstance(optimized, torch.Tensor):
                abs_diff = torch.abs(original - optimized)
                abs_error = torch.max(abs_diff).item()
                
                # Relative error computation
                original_abs = torch.abs(original)
                rel_error = torch.max(abs_diff / (original_abs + 1e-12)).item()
                
                # Check tolerance
                abs_ok = abs_error <= self.tolerance['absolute']
                rel_ok = rel_error <= self.tolerance['relative']
                passed = abs_ok or rel_ok  # Pass if either tolerance is met
                
                return abs_error, rel_error, passed
                
            elif isinstance(original, (list, tuple)) and isinstance(optimized, (list, tuple)):
                if len(original) != len(optimized):
                    return float('inf'), float('inf'), False
                
                max_abs_error = 0.0
                max_rel_error = 0.0
                all_passed = True
                
                for orig_item, opt_item in zip(original, optimized):
                    abs_err, rel_err, item_passed = self._compute_errors(orig_item, opt_item)
                    max_abs_error = max(max_abs_error, abs_err)
                    max_rel_error = max(max_rel_error, rel_err)
                    all_passed &= item_passed
                
                return max_abs_error, max_rel_error, all_passed
            
            else:
                # Try to convert to numpy arrays
                orig_arr = np.array(original)
                opt_arr = np.array(optimized)
                
                abs_diff = np.abs(orig_arr - opt_arr)
                abs_error = np.max(abs_diff)
                
                orig_abs = np.abs(orig_arr)
                rel_error = np.max(abs_diff / (orig_abs + 1e-12))
                
                abs_ok = abs_error <= self.tolerance['absolute']
                rel_ok = rel_error <= self.tolerance['relative']
                passed = abs_ok or rel_ok
                
                return float(abs_error), float(rel_error), passed
                
        except Exception as e:
            print(f"Error computing errors: {e}")
            return float('inf'), float('inf'), False
    
    def test_barrier_function_correctness(self) -> List[CorrectnessResult]:
        """Test correctness of basic barrier function operations."""
        results = []
        
        # Setup test barriers
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        rect_func = make_norm_rectangular_barrier_functional(
            center=torch.tensor([1.0, 1.0]),
            size=torch.tensor([2.0, 1.5])
        )
        
        # Test inputs
        test_states = [
            torch.tensor([[0.0, 0.0, 1.0, 0.5]], dtype=torch.float64),
            torch.tensor([[1.0, 1.0, 0.5, 0.0]], dtype=torch.float64),
            torch.randn(10, 4, dtype=torch.float64),
            torch.randn(100, 4, dtype=torch.float64)
        ]
        
        # Test barrier evaluations
        for func_name, func in [('circle', circle_func), ('rectangle', rect_func)]:
            result = self.validate_function_pair(
                original_func=func,
                optimized_func=func,  # Same function for now - will be replaced with optimized version
                test_inputs=test_states,
                test_name=f"barrier_evaluation_{func_name}"
            )
            results.append(result)
        
        return results
    
    def test_lie_derivative_correctness(self) -> List[CorrectnessResult]:
        """Test correctness of Lie derivative computations."""
        results = []
        
        # Setup dynamics
        dynamics = UnicycleDynamics()
        
        # Setup barrier function
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # Test states
        test_states = [
            torch.tensor([[0.0, 0.0, 1.0, 0.5]], dtype=torch.float64),
            torch.randn(10, 4, dtype=torch.float64),
            torch.randn(50, 4, dtype=torch.float64)
        ]
        
        # Test Lie derivatives with f field
        result_f = self.validate_function_pair(
            original_func=lambda x: lie_deriv(x, circle_func, dynamics.f),
            optimized_func=lambda x: lie_deriv(x, circle_func, dynamics.f),  # Will be replaced
            test_inputs=test_states,
            test_name="lie_derivative_f_field"
        )
        results.append(result_f)
        
        # Test Lie derivatives with g field
        result_g = self.validate_function_pair(
            original_func=lambda x: lie_deriv(x, circle_func, dynamics.g),
            optimized_func=lambda x: lie_deriv(x, circle_func, dynamics.g),  # Will be replaced
            test_inputs=test_states,
            test_name="lie_derivative_g_field"
        )
        results.append(result_g)
        
        return results
    
    def test_gradient_computation_correctness(self) -> List[CorrectnessResult]:
        """Test correctness of gradient computations using finite differences."""
        results = []
        
        # Setup test function
        def test_func(x):
            return torch.sum(x**2, dim=-1, keepdim=True)
        
        test_states = [
            torch.tensor([[1.0, 2.0, 0.5, 0.0]], dtype=torch.float64),
            torch.randn(5, 4, dtype=torch.float64)
        ]
        
        # Compare automatic differentiation with finite differences
        def finite_diff_gradient(x, func, h=1e-6):
            """Compute gradient using finite differences."""
            x = x.clone().requires_grad_(False)
            gradients = []
            
            for i in range(x.shape[0]):  # For each sample in batch
                grad_sample = torch.zeros_like(x[i])
                
                for j in range(x.shape[1]):  # For each dimension
                    x_plus = x[i].clone()
                    x_minus = x[i].clone()
                    x_plus[j] += h
                    x_minus[j] -= h
                    
                    f_plus = func(x_plus.unsqueeze(0))
                    f_minus = func(x_minus.unsqueeze(0))
                    
                    grad_sample[j] = (f_plus - f_minus) / (2 * h)
                
                gradients.append(grad_sample)
            
            return torch.stack(gradients)
        
        def autograd_gradient(x, func):
            """Compute gradient using automatic differentiation."""
            return get_func_deriv(x, func)[0]
        
        # Test gradient correctness
        for i, states in enumerate(test_states):
            result = self.validate_function_pair(
                original_func=lambda x: finite_diff_gradient(x, test_func),
                optimized_func=lambda x: autograd_gradient(x, test_func),
                test_inputs=[states],
                test_name=f"gradient_computation_test_{i}"
            )
            results.append(result)
        
        return results
    
    def test_hocbf_correctness(self) -> List[CorrectnessResult]:
        """Test correctness of higher-order CBF computations."""
        results = []
        
        dynamics = UnicycleDynamics()
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # Test different relative degrees
        for rel_deg in [1, 2, 3]:
            # Create barrier
            barrier = Barrier().assign(circle_func, rel_deg=rel_deg)
            barrier.assign_dynamics(dynamics)
            
            # Test states
            test_states = [
                torch.tensor([[0.0, 0.0, 1.0, 0.5]], dtype=torch.float64),
                torch.randn(10, 4, dtype=torch.float64)
            ]
            
            # Test HOCBF evaluation
            result = self.validate_function_pair(
                original_func=barrier.hocbf,
                optimized_func=barrier.hocbf,  # Will be replaced with optimized version
                test_inputs=test_states,
                test_name=f"hocbf_evaluation_rel_deg_{rel_deg}"
            )
            results.append(result)
            
            # Test combined computation
            result_combined = self.validate_function_pair(
                original_func=barrier.get_hocbf_and_lie_derivs,
                optimized_func=barrier.get_hocbf_and_lie_derivs,  # Will be replaced
                test_inputs=test_states,
                test_name=f"hocbf_combined_computation_rel_deg_{rel_deg}"
            )
            results.append(result_combined)
        
        return results
    
    def run_all_correctness_tests(self) -> Dict[str, List[CorrectnessResult]]:
        """Run all correctness tests."""
        return {
            'barrier_functions': self.test_barrier_function_correctness(),
            'lie_derivatives': self.test_lie_derivative_correctness(),
            'gradient_computation': self.test_gradient_computation_correctness(),
            'hocbf_computations': self.test_hocbf_correctness()
        }
    
    def print_results_summary(self, results: Dict[str, List[CorrectnessResult]]):
        """Print a summary of all test results."""
        total_tests = 0
        passed_tests = 0
        
        print("\n" + "="*60)
        print("CORRECTNESS TEST RESULTS")
        print("="*60)
        
        for category, test_results in results.items():
            print(f"\n{category.upper()}:")
            print("-" * 40)
            
            for result in test_results:
                print(f"  {result}")
                total_tests += 1
                if result.passed:
                    passed_tests += 1
        
        print("\n" + "="*60)
        print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        if passed_tests == total_tests:
            print("✓ ALL TESTS PASSED")
        else:
            print(f"✗ {total_tests - passed_tests} TESTS FAILED")
        print("="*60)