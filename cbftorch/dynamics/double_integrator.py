import torch
from .base import AffineInControlDynamics
from cbftorch.config import DEFAULT_DTYPE


class DIDynamics(AffineInControlDynamics):
    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 4
        self._action_dim = 2

    def _f(self, x):
        return torch.stack([x[:, 2],
                            x[:, 3],
                            torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])], dim=-1)

    def _g(self, x):
        return (torch.vstack([torch.zeros(2, 2, dtype=DEFAULT_DTYPE),
                              torch.eye(2, dtype=DEFAULT_DTYPE)])
                ).repeat(x.shape[0], 1, 1)
