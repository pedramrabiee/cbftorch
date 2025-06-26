import torch
from .base import AffineInControlDynamics


class BicycleDynamics(AffineInControlDynamics):
    def _f(self, x):
        return torch.stack([x[:, 2] * torch.cos(x[:, 3]),
                            x[:, 2] * torch.sin(x[:, 3]),
                            torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])], dim=-1)

    def _g(self, x):
        return (torch.stack([torch.zeros(*x.shape[:-1], 2, dtype=torch.float64),
                              torch.zeros(*x.shape[:-1], 2, dtype=torch.float64),
                              torch.tensor([1, 0]).repeat(x.shape[0], 1),
                              torch.hstack([torch.zeros(*x.shape[:-1], 1, dtype=torch.float64),
                                            x[:, 2:3] / self._params.l])], dim=1)
                )