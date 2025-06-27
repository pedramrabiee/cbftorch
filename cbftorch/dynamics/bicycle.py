import torch
from .base import AffineInControlDynamics
from cbftorch.config import DEFAULT_DTYPE


class BicycleDynamics(AffineInControlDynamics):
    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 4
        self._action_dim = 2

    def _f(self, x):
        return torch.stack([x[:, 2] * torch.cos(x[:, 3]),
                            x[:, 2] * torch.sin(x[:, 3]),
                            torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])], dim=-1)

    def _g(self, x):
        return (torch.stack([torch.zeros(*x.shape[:-1], 2, dtype=DEFAULT_DTYPE),
                             torch.zeros(*x.shape[:-1], 2, dtype=DEFAULT_DTYPE),
                             torch.tensor([1, 0], dtype=DEFAULT_DTYPE).repeat(x.shape[0], 1),
                             torch.hstack([torch.zeros(*x.shape[:-1], 1, dtype=DEFAULT_DTYPE),
                                           x[:, 2:3] / self._params.l])], dim=1)
                )
