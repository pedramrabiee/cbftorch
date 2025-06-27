import torch
from .base import AffineInControlDynamics
from cbftorch.config import DEFAULT_DTYPE


class SIDynamics(AffineInControlDynamics):
    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 2
        self._action_dim = 2

    def _f(self, x):
        return torch.zeros(*x.shape[:-1], 2, dtype=DEFAULT_DTYPE)

    def _g(self, x):
        return torch.eye(2, dtype=DEFAULT_DTYPE).repeat(x.shape[0], 1, 1)
