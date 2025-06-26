import torch
from .base import AffineInControlDynamics


class InvertPendDynamics(AffineInControlDynamics):
    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 2
        self._action_dim = 1

    def _f(self, x):
        return torch.stack([x[:, 1],
                             torch.sin(x[:, 0])], dim=-1)

    def _g(self, x):
        batch_size = x.shape[0]
        base_tensor = torch.tensor([[0.], [1.]], dtype=torch.float64, requires_grad=x.requires_grad)
        return base_tensor.expand(batch_size, -1, -1)