"""
Configuration settings for CBFTorch.
"""

import torch

# Default precision for all CBFTorch computations
# CBF methods require high precision for numerical stability
DEFAULT_DTYPE = torch.float64

def get_default_dtype():
    """Get the current default dtype for CBFTorch computations."""
    return DEFAULT_DTYPE

def set_default_dtype(dtype):
    """
    Set the default dtype for CBFTorch computations.
    
    Parameters:
        dtype: torch.dtype
            The desired default dtype (e.g., torch.float32, torch.float64)
    
    Example:
      import cbftorch
      cbftorch.set_default_dtype(torch.float32)  # Use single precision
      cbftorch.set_default_dtype(torch.float64)  # Use double precision (default)
    """
    global DEFAULT_DTYPE
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"dtype must be a torch.dtype, got {type(dtype)}")
    DEFAULT_DTYPE = dtype
