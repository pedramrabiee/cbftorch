import torch
from cbftorch.utils.dynamics import AffineInControlDynamics

class DIDynamics(AffineInControlDynamics):
    def _f(self, x):
        return torch.stack([x[:, 2],
                            x[:, 3],
                            torch.zeros_like(x[:, 0]),
                            torch.zeros_like(x[:, 0])], dim=-1)

    def _g(self, x):
        return (torch.vstack([torch.zeros(2, 2, dtype=torch.float64),
                              torch.eye(2, dtype=torch.float64)])
                ).repeat(x.shape[0], 1, 1)





