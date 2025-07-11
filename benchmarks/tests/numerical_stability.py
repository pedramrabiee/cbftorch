"""
Numerical stability tests for CBFtorch optimizations.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Callable
from dataclasses import dataclass
from cbftorch.barriers.barrier import Barrier
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.utils import make_circle_barrier_functional, lie_deriv


@dataclass
class StabilityResult:
    """Results from numerical stability tests."""
    test_name: str
    condition_numbers: List[float]
    max_condition_number: float
    gradient_norms: List[float]
    max_gradient_norm: float
    numerical_rank_deficient: bool
    stability_score: float  # 0-1 scale, 1 is most stable
    
    def __str__(self):
        status = "STABLE" if self.stability_score > 0.7 else "UNSTABLE"
        return (f"{status} {self.test_name}: "
                f"max_cond={self.max_condition_number:.2e}, "
                f"score={self.stability_score:.3f}")


class NumericalStabilityTester:
    """Tester for numerical stability of CBFtorch operations."""
    
    def __init__(self):
        self.dynamics = UnicycleDynamics()
        
    def test_gradient_stability(
        self,
        func: Callable,
        test_points: torch.Tensor,
        perturbation_scales: List[float] = None
    ) -> StabilityResult:
        """
        Test stability of gradient computations under small perturbations.
        
        Args:
            func: Function to test gradient stability for
            test_points: Points to evaluate gradients at
            perturbation_scales: Scales of perturbations to test
            
        Returns:
            StabilityResult with stability analysis
        """
        if perturbation_scales is None:
            perturbation_scales = [1e-8, 1e-6, 1e-4, 1e-2]
        
        test_points = test_points.clone().requires_grad_(True)
        
        # Compute original gradient
        original_output = func(test_points)
        original_grad = torch.autograd.grad(
            outputs=original_output.sum(),
            inputs=test_points,
            create_graph=True
        )[0]
        
        condition_numbers = []
        gradient_norms = []
        relative_errors = []
        
        for scale in perturbation_scales:
            # Add small perturbation
            perturbation = scale * torch.randn_like(test_points)
            perturbed_points = (test_points + perturbation).clone().requires_grad_(True)
            
            # Compute perturbed gradient
            perturbed_output = func(perturbed_points)
            perturbed_grad = torch.autograd.grad(
                outputs=perturbed_output.sum(),
                inputs=perturbed_points,
                create_graph=True
            )[0]
            
            # Analyze stability
            grad_diff = torch.norm(perturbed_grad - original_grad, dim=-1)
            input_diff = torch.norm(perturbation, dim=-1)
            
            # Condition number approximation
            condition_number = (grad_diff / (torch.norm(original_grad, dim=-1) + 1e-12)).mean().item()
            condition_numbers.append(condition_number)
            
            # Gradient norms
            grad_norm = torch.norm(perturbed_grad, dim=-1).mean().item()
            gradient_norms.append(grad_norm)
            
            # Relative error
            rel_error = (grad_diff / (torch.norm(original_grad, dim=-1) + 1e-12)).mean().item()
            relative_errors.append(rel_error)
        
        # Check for rank deficiency (very large condition numbers)
        max_condition = max(condition_numbers) if condition_numbers else 0.0
        rank_deficient = max_condition > 1e12
        
        # Compute stability score (lower condition numbers and consistent gradients = higher score)
        stability_score = 1.0 / (1.0 + np.log10(max_condition + 1))
        
        return StabilityResult(
            test_name=f"gradient_stability_{func.__name__ if hasattr(func, '__name__') else 'function'}",
            condition_numbers=condition_numbers,
            max_condition_number=max_condition,
            gradient_norms=gradient_norms,
            max_gradient_norm=max(gradient_norms) if gradient_norms else 0.0,
            numerical_rank_deficient=rank_deficient,
            stability_score=stability_score
        )
    
    def test_barrier_stability_near_boundary(self) -> List[StabilityResult]:
        """Test stability of barrier functions near the boundary (h ≈ 0)."""
        results = []
        
        # Circle barrier
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # Generate points near the boundary
        angles = torch.linspace(0, 2*np.pi, 20)
        boundary_points = []
        
        for radius_offset in [-0.1, -0.01, 0.01, 0.1]:  # Points near boundary
            radius = 2.0 + radius_offset
            x = radius * torch.cos(angles)
            y = radius * torch.sin(angles)
            # Add velocity and heading (for unicycle dynamics)
            v = torch.ones_like(x) * 0.5
            theta = angles
            points = torch.stack([x, y, v, theta], dim=1).to(torch.float64)
            boundary_points.append(points)
        
        test_points = torch.cat(boundary_points, dim=0)
        
        # Test barrier function stability
        result = self.test_gradient_stability(circle_func, test_points)
        result.test_name = "barrier_stability_near_boundary"
        results.append(result)
        
        return results
    
    def test_lie_derivative_stability(self) -> List[StabilityResult]:
        """Test stability of Lie derivative computations."""
        results = []
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # Test points including some that might be problematic
        test_points = torch.tensor([
            [0.0, 0.0, 1.0, 0.0],    # At center
            [2.0, 0.0, 1.0, np.pi/2],  # On boundary
            [1.9, 0.0, 0.1, 0.0],    # Near boundary, slow
            [10.0, 10.0, 2.0, np.pi], # Far from boundary
        ], dtype=torch.float64)
        
        # Test Lf stability
        lf_func = lambda x: lie_deriv(x, circle_func, self.dynamics.f)
        result_lf = self.test_gradient_stability(lf_func, test_points)
        result_lf.test_name = "lie_derivative_Lf_stability"
        results.append(result_lf)
        
        # Test Lg stability  
        lg_func = lambda x: lie_deriv(x, circle_func, self.dynamics.g)
        result_lg = self.test_gradient_stability(lg_func, test_points)
        result_lg.test_name = "lie_derivative_Lg_stability"
        results.append(result_lg)
        
        return results
    
    def test_hocbf_stability(self) -> List[StabilityResult]:
        """Test stability of higher-order CBF computations."""
        results = []
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # Test different relative degrees
        test_points = torch.tensor([
            [1.0, 1.0, 0.5, np.pi/4],
            [0.5, 0.5, 1.0, 0.0],
            [3.0, 0.0, 0.8, np.pi/2]
        ], dtype=torch.float64)
        
        for rel_deg in [1, 2, 3]:
            barrier = Barrier().assign(circle_func, rel_deg=rel_deg)
            barrier.assign_dynamics(self.dynamics)
            
            result = self.test_gradient_stability(barrier.hocbf, test_points)
            result.test_name = f"hocbf_stability_rel_deg_{rel_deg}"
            results.append(result)
        
        return results
    
    def test_extreme_values_handling(self) -> List[StabilityResult]:
        """Test handling of extreme values (very large/small numbers)."""
        results = []
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # Extreme test cases
        extreme_points = torch.tensor([
            [1e6, 1e6, 1.0, 0.0],     # Very large positions
            [1e-6, 1e-6, 1e-3, 0.0],  # Very small values
            [0.0, 0.0, 1e6, 0.0],     # Very large velocity
            [1000.0, 0.0, 0.001, 0.0], # Large position, small velocity
        ], dtype=torch.float64)
        
        # Test barrier function with extreme values
        result_barrier = self.test_gradient_stability(circle_func, extreme_points)
        result_barrier.test_name = "extreme_values_barrier"
        results.append(result_barrier)
        
        # Test Lie derivatives with extreme values
        lf_func = lambda x: lie_deriv(x, circle_func, self.dynamics.f)
        result_lf = self.test_gradient_stability(lf_func, extreme_points)
        result_lf.test_name = "extreme_values_lie_deriv"
        results.append(result_lf)
        
        return results
    
    def test_batch_consistency(self) -> List[StabilityResult]:
        """Test consistency of computations across different batch sizes."""
        results = []
        
        circle_func = make_circle_barrier_functional(
            center=torch.tensor([0.0, 0.0]), 
            radius=2.0
        )
        
        # Single test point
        single_point = torch.tensor([[1.0, 1.0, 0.5, 0.0]], dtype=torch.float64)
        
        # Same point repeated in batch
        batch_sizes = [1, 5, 10, 50]
        consistency_errors = []
        
        # Get reference result
        reference_result = circle_func(single_point)
        
        for batch_size in batch_sizes:
            batch_points = single_point.repeat(batch_size, 1)
            batch_result = circle_func(batch_points)
            
            # Check consistency
            consistency_error = torch.max(torch.abs(
                batch_result - reference_result.repeat(batch_size, 1)
            )).item()
            consistency_errors.append(consistency_error)
        
        # Create stability result
        max_error = max(consistency_errors)
        stability_score = 1.0 if max_error < 1e-12 else 1.0 / (1.0 + np.log10(max_error))
        
        result = StabilityResult(
            test_name="batch_consistency",
            condition_numbers=[max_error],
            max_condition_number=max_error,
            gradient_norms=[],
            max_gradient_norm=0.0,
            numerical_rank_deficient=max_error > 1e-6,
            stability_score=stability_score
        )
        
        results.append(result)
        return results
    
    def run_all_stability_tests(self) -> Dict[str, List[StabilityResult]]:
        """Run all numerical stability tests."""
        return {
            'barrier_boundary': self.test_barrier_stability_near_boundary(),
            'lie_derivatives': self.test_lie_derivative_stability(),
            'hocbf_stability': self.test_hocbf_stability(),
            'extreme_values': self.test_extreme_values_handling(),
            'batch_consistency': self.test_batch_consistency()
        }
    
    def print_stability_summary(self, results: Dict[str, List[StabilityResult]]):
        """Print summary of stability test results."""
        print("\n" + "="*60)
        print("NUMERICAL STABILITY TEST RESULTS")
        print("="*60)
        
        total_tests = 0
        stable_tests = 0
        
        for category, test_results in results.items():
            print(f"\n{category.upper()}:")
            print("-" * 40)
            
            for result in test_results:
                print(f"  {result}")
                total_tests += 1
                if result.stability_score > 0.7:
                    stable_tests += 1
        
        print("\n" + "="*60)
        print(f"SUMMARY: {stable_tests}/{total_tests} tests stable")
        stability_ratio = stable_tests / total_tests if total_tests > 0 else 0
        
        if stability_ratio > 0.9:
            print("✓ EXCELLENT NUMERICAL STABILITY")
        elif stability_ratio > 0.7:
            print("⚠ GOOD NUMERICAL STABILITY")
        else:
            print("✗ POOR NUMERICAL STABILITY - OPTIMIZATION NEEDED")
        print("="*60)