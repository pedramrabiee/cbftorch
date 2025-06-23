"""
CBFTorch: Control Barrier Functions in PyTorch

A PyTorch-based library for implementing Control Barrier Functions (CBFs) 
and higher-order CBFs for safe control of dynamical systems.
"""

__version__ = "0.1.0"
__author__ = "Pedram Rabiee"
__email__ = "pedram.rabiee@gmail.com"

# Import main classes for easier access
from .barriers.barrier import Barrier
from .barriers.composite_barrier import (
    CompositionBarrier,
    SoftCompositionBarrier,
    NonSmoothCompositionBarrier
)
from .barriers.multi_barrier import MultiBarriers
from .barriers.backup_barrier import BackupBarrier

from .safe_controls.closed_form_safe_control import (
    CFSafeControl,
    MinIntervCFSafeControl,
    InputConstCFSafeControl,
    MinIntervInputConstCFSafeControl
)
from .safe_controls.qp_safe_control import QPSafeControl
from .safe_controls.backup_safe_control import BackupSafeControl

from .utils.dynamics import AffineInControlDynamics

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Barriers
    "Barrier",
    "CompositionBarrier",
    "SoftCompositionBarrier", 
    "NonSmoothCompositionBarrier",
    "MultiBarriers",
    "BackupBarrier",
    
    # Safe Controls
    "CFSafeControl",
    "MinIntervCFSafeControl",
    "InputConstCFSafeControl",
    "MinIntervInputConstCFSafeControl",
    "QPSafeControl",
    "BackupSafeControl",
    
    # Utils
    "AffineInControlDynamics",
]