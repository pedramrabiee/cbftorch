"""
Tensor Management System for CBFtorch

This module provides centralized tensor handling with automatic batch dimension
management, dtype validation, and standardized input processing.
"""

import torch
import numpy as np
from typing import Union, Optional, Callable, Any
from functools import wraps
from ..config import DEFAULT_DTYPE
import warnings


class TensorManager:
    """
    Centralized tensor management for CBFtorch operations.
    
    Handles automatic batch dimension management, dtype conversion,
    and standardized tensor input processing.
    """
    
    def __init__(self, default_dtype: Optional[torch.dtype] = None):
        """
        Initialize TensorManager.
        
        Args:
            default_dtype: Default dtype for tensor operations. If None, uses DEFAULT_DTYPE.
        """
        self.default_dtype = default_dtype or DEFAULT_DTYPE
        self._tensor_cache = {}  # Cache for commonly used tensors
        
    def standardize_input(self, x: Union[torch.Tensor, np.ndarray, list, tuple], 
                         ensure_batch: bool = True, 
                         target_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Standardize input tensor with consistent batch dimension and dtype.
        
        Args:
            x: Input tensor or array-like object
            ensure_batch: If True, ensure first dimension is batch dimension
            target_dtype: Target dtype for conversion. If None, uses default_dtype.
            
        Returns:
            Standardized tensor with proper batch dimension and dtype
            
        Raises:
            ValueError: If input cannot be converted to tensor
            TypeError: If input type is not supported
        """
        # Convert to tensor if needed
        if not isinstance(x, torch.Tensor):
            x = self._convert_to_tensor(x, target_dtype or self.default_dtype)
        else:
            # Ensure correct dtype
            target_dtype = target_dtype or self.default_dtype
            if x.dtype != target_dtype:
                x = x.to(target_dtype)
        
        # Ensure batch dimension if requested
        if ensure_batch:
            x = self._ensure_batch_dimension(x)
            
        return x
    
    def _convert_to_tensor(self, obj: Any, dtype: torch.dtype) -> torch.Tensor:
        """Convert various input types to tensor."""
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        elif isinstance(obj, np.ndarray):
            return torch.from_numpy(obj).to(dtype)
        elif isinstance(obj, (list, tuple)):
            return torch.tensor(obj, dtype=dtype)
        elif isinstance(obj, (int, float)):
            return torch.tensor(obj, dtype=dtype)
        else:
            raise TypeError(f"Cannot convert {type(obj)} to tensor")
    
    def _ensure_batch_dimension(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure tensor has proper batch dimension."""
        if x.ndim == 0:
            # Scalar -> (1, 1)
            return x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 1:
            # Vector -> (1, n) - add batch dimension
            return x.unsqueeze(0)
        else:
            # Already has batch dimension
            return x
    
    def is_batched(self, x: torch.Tensor) -> bool:
        """Check if tensor has batch dimension (ndim >= 2)."""
        return x.ndim >= 2
    
    def get_batch_size(self, x: torch.Tensor) -> int:
        """Get batch size from tensor."""
        return x.shape[0] if self.is_batched(x) else 1
    
    def batch_apply(self, func: Callable, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Apply function to tensor, handling batch dimension automatically.
        
        Args:
            func: Function to apply
            x: Input tensor
            *args, **kwargs: Additional arguments for func
            
        Returns:
            Result tensor with proper batch dimension handling
        """
        x_std = self.standardize_input(x, ensure_batch=True)
        result = func(x_std, *args, **kwargs)
        
        # Handle result dimensions
        if isinstance(result, torch.Tensor):
            # If input was single sample but we added batch dim, remove it from result
            if x.ndim == 1 and result.ndim == 2 and result.shape[0] == 1:
                return result.squeeze(0)
        
        return result
    
    def validate_compatible_shapes(self, *tensors: torch.Tensor) -> bool:
        """Validate that tensors have compatible batch dimensions."""
        if not tensors:
            return True
            
        batch_sizes = [self.get_batch_size(t) for t in tensors]
        return all(bs == batch_sizes[0] or bs == 1 for bs in batch_sizes)
    
    def broadcast_to_common_batch(self, *tensors: torch.Tensor) -> tuple:
        """Broadcast tensors to common batch size."""
        if not tensors:
            return tensors
            
        batch_sizes = [self.get_batch_size(t) for t in tensors]
        max_batch_size = max(batch_sizes)
        
        result = []
        for t in tensors:
            if self.get_batch_size(t) == 1 and max_batch_size > 1:
                # Broadcast single sample to batch
                t = t.expand(max_batch_size, *t.shape[1:])
            result.append(t)
            
        return tuple(result)


class DtypeManager:
    """
    Robust dtype management for CBFtorch operations.
    
    Handles automatic dtype conversion, validation, and compatibility
    with clear error messages and user control over conversion behavior.
    """
    
    def __init__(self, default_dtype: Optional[torch.dtype] = None, 
                 strict_mode: bool = False, 
                 warn_on_conversion: bool = True):
        """
        Initialize DtypeManager.
        
        Args:
            default_dtype: Default dtype for operations. If None, uses DEFAULT_DTYPE.
            strict_mode: If True, raises errors on dtype mismatches instead of converting.
            warn_on_conversion: If True, warns when dtype conversion occurs.
        """
        self.default_dtype = default_dtype or DEFAULT_DTYPE
        self.strict_mode = strict_mode
        self.warn_on_conversion = warn_on_conversion
        
        # Supported dtype conversions
        self.supported_dtypes = {
            torch.float32, torch.float64, torch.int32, torch.int64
        }
        
        # Compatible dtype groups (can convert between these)
        self.compatible_groups = [
            {torch.float32, torch.float64},  # Float types
            {torch.int32, torch.int64},      # Integer types
        ]
    
    def validate_and_convert(self, x: torch.Tensor, 
                           target_dtype: Optional[torch.dtype] = None,
                           allow_conversion: bool = True) -> torch.Tensor:
        """
        Validate and convert tensor to target dtype.
        
        Args:
            x: Input tensor
            target_dtype: Target dtype. If None, uses default_dtype.
            allow_conversion: If False, raises error on dtype mismatch.
            
        Returns:
            Tensor with validated/converted dtype
            
        Raises:
            TypeError: If dtype conversion is not allowed or not supported
            ValueError: If strict mode is enabled and dtypes don't match
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        
        target_dtype = target_dtype or self.default_dtype
        
        # If dtypes match, return as-is
        if x.dtype == target_dtype:
            return x
        
        # Check if conversion is allowed
        if not allow_conversion or self.strict_mode:
            raise ValueError(f"Dtype mismatch: expected {target_dtype}, got {x.dtype}")
        
        # Check if conversion is supported
        if not self._is_conversion_supported(x.dtype, target_dtype):
            raise TypeError(f"Unsupported dtype conversion: {x.dtype} -> {target_dtype}")
        
        # Warn if requested
        if self.warn_on_conversion:
            warnings.warn(f"Converting tensor from {x.dtype} to {target_dtype}", 
                         UserWarning, stacklevel=3)
        
        return x.to(target_dtype)
    
    def _is_conversion_supported(self, from_dtype: torch.dtype, to_dtype: torch.dtype) -> bool:
        """Check if dtype conversion is supported."""
        # Check if both dtypes are in compatible groups
        for group in self.compatible_groups:
            if from_dtype in group and to_dtype in group:
                return True
        return False
    
    def ensure_compatible_dtypes(self, *tensors: torch.Tensor) -> tuple:
        """
        Ensure all tensors have compatible dtypes.
        
        Args:
            *tensors: Variable number of tensors
            
        Returns:
            Tuple of tensors with compatible dtypes
            
        Raises:
            TypeError: If tensors have incompatible dtypes
        """
        if not tensors:
            return tensors
        
        # Find the "best" dtype (prefer float64 over float32, etc.)
        target_dtype = self._get_best_dtype([t.dtype for t in tensors])
        
        # Convert all tensors to target dtype
        result = []
        for t in tensors:
            result.append(self.validate_and_convert(t, target_dtype))
        
        return tuple(result)
    
    def _get_best_dtype(self, dtypes: list) -> torch.dtype:
        """Get the best dtype from a list of dtypes."""
        # Priority order: float64 > float32 > int64 > int32
        priority_order = [torch.float64, torch.float32, torch.int64, torch.int32]
        
        for dtype in priority_order:
            if dtype in dtypes:
                return dtype
        
        # If no match found, return default
        return self.default_dtype
    
    def get_dtype_info(self, x: torch.Tensor) -> dict:
        """Get detailed dtype information for tensor."""
        return {
            'dtype': x.dtype,
            'is_floating_point': x.dtype.is_floating_point,
            'is_complex': x.dtype.is_complex,
            'is_signed': x.dtype.is_signed if hasattr(x.dtype, 'is_signed') else None,
            'itemsize': x.element_size(),
            'compatible_with_default': self._is_conversion_supported(x.dtype, self.default_dtype)
        }


# Global manager instances
_tensor_manager = TensorManager()
_dtype_manager = DtypeManager()


def ensure_batched(func: Callable) -> Callable:
    """
    Decorator to ensure function inputs have proper batch dimensions.
    
    Automatically standardizes the first argument (assumed to be tensor input)
    and restores original shape if needed.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not args:
            return func(*args, **kwargs)
        
        # Standardize first argument (assumed to be main tensor input)
        x = args[0]
        was_single_sample = isinstance(x, torch.Tensor) and x.ndim == 1
        
        # Standardize input
        x_std = _tensor_manager.standardize_input(x, ensure_batch=True)
        
        # Call function with standardized input
        result = func(x_std, *args[1:], **kwargs)
        
        # Handle result dimensions for single sample inputs
        if was_single_sample and isinstance(result, torch.Tensor):
            if result.ndim == 2 and result.shape[0] == 1:
                result = result.squeeze(0)
        
        return result
    
    return wrapper


def tensor_input(ensure_batch: bool = True, target_dtype: Optional[torch.dtype] = None, 
                input_arg_index: int = 0) -> Callable:
    """
    Decorator for automatic tensor input standardization.
    
    Args:
        ensure_batch: Whether to ensure batch dimension
        target_dtype: Target dtype for conversion
        input_arg_index: Index of the tensor argument to standardize (0 for functions, 1 for methods)
        
    Returns:
        Decorated function with automatic input standardization
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) <= input_arg_index:
                return func(*args, **kwargs)
            
            # Standardize specified argument
            args_list = list(args)
            x = args_list[input_arg_index]
            
            # Only standardize if it's a tensor-like object
            if isinstance(x, (torch.Tensor, np.ndarray, list, tuple, int, float)):
                x_std = _tensor_manager.standardize_input(x, ensure_batch=ensure_batch, 
                                                        target_dtype=target_dtype)
                args_list[input_arg_index] = x_std
            
            return func(*args_list, **kwargs)
        
        return wrapper
    return decorator


def get_tensor_manager() -> TensorManager:
    """Get the global tensor manager instance."""
    return _tensor_manager


def set_tensor_manager(manager: TensorManager) -> None:
    """Set the global tensor manager instance."""
    global _tensor_manager
    _tensor_manager = manager


def ensure_dtype(target_dtype: Optional[torch.dtype] = None, 
                allow_conversion: bool = True,
                input_arg_index: int = 0) -> Callable:
    """
    Decorator to ensure function inputs have correct dtype.
    
    Args:
        target_dtype: Target dtype for conversion. If None, uses default.
        allow_conversion: If False, raises error on dtype mismatch.
        input_arg_index: Index of the tensor argument to validate (0 for functions, 1 for methods)
        
    Returns:
        Decorated function with dtype validation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) <= input_arg_index:
                return func(*args, **kwargs)
            
            # Validate/convert specified argument
            args_list = list(args)
            x = args_list[input_arg_index]
            
            if isinstance(x, torch.Tensor):
                x = _dtype_manager.validate_and_convert(x, target_dtype, allow_conversion)
                args_list[input_arg_index] = x
            
            return func(*args_list, **kwargs)
        
        return wrapper
    return decorator


def validate_dtype(strict: bool = False) -> Callable:
    """
    Decorator for strict dtype validation.
    
    Args:
        strict: If True, raises error on any dtype mismatch
        
    Returns:
        Decorated function with strict dtype validation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not args:
                return func(*args, **kwargs)
            
            # Validate first argument
            x = args[0]
            if isinstance(x, torch.Tensor):
                x = _dtype_manager.validate_and_convert(x, allow_conversion=not strict)
            
            return func(x, *args[1:], **kwargs)
        
        return wrapper
    return decorator


def get_dtype_manager() -> DtypeManager:
    """Get the global dtype manager instance."""
    return _dtype_manager


def set_dtype_manager(manager: DtypeManager) -> None:
    """Set the global dtype manager instance."""
    global _dtype_manager
    _dtype_manager = manager


# Legacy compatibility functions (to be deprecated)
def vectorize_tensors_v2(arr: Any) -> torch.Tensor:
    """
    Improved version of vectorize_tensors using TensorManager.
    
    This function provides backward compatibility while using the new
    tensor management system.
    """
    return _tensor_manager.standardize_input(arr, ensure_batch=True)


def apply_and_batchize_v2(func: Callable, x: Any) -> torch.Tensor:
    """
    Improved version of apply_and_batchize using TensorManager.
    
    This function provides backward compatibility while using the new
    tensor management system.
    """
    return _tensor_manager.batch_apply(func, x)