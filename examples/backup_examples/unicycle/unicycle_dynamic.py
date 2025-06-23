import torch
from cbftorch.utils.dynamics import AffineInControlDynamics


class UnicycleReducedOrderDynamics(AffineInControlDynamics):
    def __init__(self, params=None, **kwargs):
        super().__init__(params, **kwargs)
        self._state_dim = 4
        self._action_dim = 2
        self.d = self._params.d

    def _f(self, x):
        return torch.stack([x[:, 2] * torch.cos(x[:, 3]),
                            x[:, 2] * torch.sin(x[:, 3]),
                            torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])], dim=-1)

    def _g(self, x):
        return torch.stack([
        torch.stack([torch.zeros_like(x[:, 0]),
                    torch.zeros_like(x[:, 0], dtype=torch.float64),
                    torch.ones_like(x[:, 0], dtype=torch.float64),
                    torch.zeros_like(x[:, 0], dtype=torch.float64)], dim=-1),
        torch.stack([-self.d * torch.sin(x[:, 3]),
                    self.d * torch.cos(x[:, 3]),
                    torch.zeros_like(x[:, 0], dtype=torch.float64),
                    torch.ones_like(x[:, 0], dtype=torch.float64)], dim=-1)
    ], dim=-1)
