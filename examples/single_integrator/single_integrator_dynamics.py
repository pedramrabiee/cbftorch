import torch
from cbftorch.utils.dynamics import AffineInControlDynamics

class SIDynamics(AffineInControlDynamics):
    def _f(self, x):
        return torch.zeros(*x.shape[:-1], 2, dtype=torch.float64)

    def _g(self, x):
        return torch.eye(2, dtype=torch.float64).repeat(x.shape[0], 1, 1)





