from typing import List
from cbftorch.utils.utils import *
from cbftorch.barriers.barrier import Barrier



class MultiBarriers(Barrier):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._barriers = []
        self._hocbf_funcs = []
        self._barrier_funcs = []

    def assign(self, barrier_func, rel_deg=1, alphas=None):
        raise "Use add_barriers method to add barriers for MultiBarriers class"

    def add_barriers(self, barriers: List[Barrier], infer_dynamics: bool = False):
        # Infer dynamics of the first barrier if infer_dynamics = True and dynamics is not already assigned
        if infer_dynamics:
            if self._dynamics is None:
                self._dynamics = barriers[0].dynamics

        # Define barrier functions and higher-order barrier function
        self._barrier_funcs.extend([barrier.barrier for barrier in barriers])
        self._hocbf_funcs.extend([barrier.hocbf for barrier in barriers])
        self._barriers.extend([barrier.barriers for barrier in barriers])

        return self

    def assign_dynamics(self, dynamics):
        """
        Assign dynamics
        """
        if self._dynamics is not None:
            raise Warning('The assinged dynamics is overriden by the dynamics of the'
                          ' first barrier on the barriers list')

        self._dynamics = dynamics
        return self

    def barrier(self, x):
        """
        Compute main barrier value at x. Main barrier value is the barrier which defines all the
         higher order cbfs involved in the composite barrier function expression.
         This method returns a horizontally stacked torch tensor of the value of barriers at x.
         Output: (batchsize, len(self._hocbf_funcs), 1)
        """
        return torch.stack([apply_and_batchize(barrier, x) for barrier in self._barrier_funcs]).transpose(1, 0)

    def hocbf(self, x):
        """
        Compute the highest-order barrier function hocbf(x) of self._hocbf_funcs for a given trajs x.
        This method returns a horizontally stacked torch tensor of the value of barriers at x.
        Output: (batchsize, len(self._hocbf_funcs), 1)
        """
        return torch.stack([apply_and_batchize(hocbf, x) for hocbf in self._hocbf_funcs]).transpose(1, 0)

    def Lf_hocbf(self, x):
        """
        Compute the Lie derivative of the highest-order barrier function with respect to the system dynamics f.
        Output: (batchsize, len(self._hocbf_funcs), f dimension)
        """
        return torch.stack([lie_deriv(x, hocbf, self._dynamics.f) for hocbf in self._hocbf_funcs]).transpose(1, 0).detach()

    def Lg_hocbf(self, x):
        """
        Compute the Lie derivative of the highest-order barrier function with respect to the system dynamics g.
        Output: (batchsize, len(self._hocbf_funcs), g.shape)
        """
        return torch.stack([lie_deriv(x, hocbf, self._dynamics.g) for hocbf in self._hocbf_funcs]).transpose(1, 0).detach()

    def min_barrier(self, x):
        """
        Calculate the minimum value among all the barrier values computed at point x.
        """
        return torch.min(self.barrier(x), dim=-2).values

    def get_hocbf_and_lie_derivs(self, x):
        return self.get_hocbf_and_lie_derivs_v2(x)

    @property
    def barriers_flatten(self):
        return [b for barrier in self._barriers for b in barrier]
