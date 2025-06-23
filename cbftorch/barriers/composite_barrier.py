from typing import List
from cbftorch.utils.utils import *
from cbftorch.barriers.barrier import Barrier


class CompositionBarrier(Barrier):
    """
    CompositionBarrier class, inherits from Barrier.
    This class represents a barrier formed by composing multiple barriers with a specific rule.
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self._barrier_funcs = None
        self._composition_rule = None
        self._barriers_raw = None

    def assign(self, barrier_func, rel_deg=1, alphas=None):
        """
        Override the assign method to raise an error.
        Assignment should be made through the assign_barriers_and_rule method.
        """
        raise 'For the CompositionBarrier class, assignment should be made through the assign_barriers_and_rule method'

    def assign_dynamics(self, dynamics):
        """
        Assign dynamics
        """
        # Handle the case where the composition barrier and all other barrier is already made, and a new dynamics is assigned
        self._dynamics = dynamics
        if self._composition_rule is not None:
            # barriers should be created once again using
            self.assign_barriers_and_rule(barriers=self._barriers_raw, rule=self._composition_rule,
                                          infer_dynamics=False)
        return self

    # @torch.jit.script
    def assign_barriers_and_rule(self, barriers: List[Barrier], rule: str, infer_dynamics: bool = False):
        """
        Assign multiple barriers and a composition rule to the CompositionBarrier object.

        Parameters:
            - barriers: List of Barrier objects to be composed.
            - rule: Composition rule to be applied. Choose from 'intersection', 'union', 'i' or 'u'

        Returns:
            - self: Updated CompositionBarrier object.
        """
        # For the case where a new dynamics is assigned save these arguments. Check assgin_dynamics for more details
        self._composition_rule = rule
        self._barriers_raw = barriers

        # Infer dynamics from the barriers
        if infer_dynamics:
            if self._dynamics is not None:
                raise Warning('The assinged dynamics is overriden by the dynamics of the'
                              ' first barrier on the barriers list')
            self._dynamics = barriers[0].dynamics
        elif self._dynamics is None:
            raise ('Dynamics should be assigned. Use infer_dynamics=True to take the dynamics of'
                   ' the first barrier as the dynamics of the composition barrier')

        self._rel_deg = 1

        # Define barrier functions and higher-order barrier function as compositions of individual barrier functions
        self._barrier_funcs = lambda x: torch.hstack([barrier.barrier(x) for barrier in barriers])
        self._hocbf_func = lambda x: self.compose(rule)(torch.hstack([barrier.hocbf(x) for barrier in barriers]))

        # Concatenate barrier.barriers for all barrier in barriers. This makes a list of lists.
        self._barriers = [barrier.barriers for barrier in barriers]

        # Append the higher-order composite barrier function to the list of barriers
        self._barriers.append([self._hocbf_func])

        return self

    def raise_rel_deg(self, x, raise_rel_deg_by=1, alphas=None):
        """
        This method takes the current hocbf and make a new hocbf with the relative degree raised
        by `raise_rel_deg_by`. The new hocbf has the relative degree of old rel_deg + raise_rel_deg_by
        """

        alphas = self._handle_alphas(alphas=alphas, rel_deg=raise_rel_deg_by)
        self._alphas.append(alphas)
        self._rel_deg += raise_rel_deg_by
        new_barriers = self._make_hocbf_series(barrier=self._hocbf_func,
                                               rel_deg=raise_rel_deg_by,
                                               alphas=alphas)

        self._hocbf_func = new_barriers[-1]
        self._barriers.append(new_barriers)

    def barrier(self, x):
        """
        Compute main barrier value at x. Main barrier value is the barrier which defines all the
         higher order cbfs involved in the composite barrier function expression.
         This method returns a horizontally stacked torch tensor of the value of barriers at x.
        """
        return apply_and_batchize(self._barrier_funcs, x)

    def compose(self, c_key: str):
        """
       Select the appropriate composition rule based on the provided key.

       Parameters:
           - c_key: Composition rule key.
        """
        if c_key == 'union' or c_key == 'u':
            return self.union_rule
        if c_key == 'intersection' or c_key == 'i':
            return self.intersection_rule

        return "composition rule is not valid"

    def union_rule(self, x):
        """
        Union composition rule placeholder.
        """
        raise NotImplementedError

    def intersection_rule(self, x):
        """
        Intersection composition rule placeholder.
        """
        raise NotImplementedError

    @property
    def barriers_flatten(self):
        return [b for barrier in self._barriers for b in barrier]


class SoftCompositionBarrier(CompositionBarrier):
    """
    SoftCompositionBarrier class, inherits from CompositionBarrier.
    This class represents a soft composition of multiple barriers with specific soft composition rules.
    """

    def union_rule(self, x):
        return apply_and_match_dim(lambda y: softmax(y, rho=self.cfg.softmax_rho, dim=-1), x)

    def intersection_rule(self, x):
        return apply_and_match_dim(lambda y: softmin(y, rho=self.cfg.softmin_rho, dim=-1), x)


class NonSmoothCompositionBarrier(CompositionBarrier):
    """
    NonSmoothCompositionBarrier class, inherits from CompositionBarrier.
    This class represents a non-smooth composition of multiple barriers with specific non-smooth composition rules.
    """

    def union_rule(self, x):
        return apply_and_match_dim(lambda y: torch.max(y, dim=-1).values, x)

    def intersection_rule(self, x):
        return apply_and_match_dim(lambda y: torch.min(y, dim=-1).values, x)
