from cbftorch.utils.utils import *
import torch
from torch.nn.functional import softplus, relu
from box import Box as AD
from cbftorch.utils.dynamics import AffineInControlDynamics
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier
from cbftorch.safe_controls.base_safe_control import BaseSafeControl, BaseMinIntervSafeControl


class CFSafeControl(BaseSafeControl):
    """
    Closed-Form-Safe Control
    """

    def __init__(self, action_dim, alpha=None, params=None):
        super().__init__(action_dim, alpha, params)
        self._action_dim = action_dim
        update_dict_no_overwrite(self._params, AD(slack_gain=1e24,
                                                        use_softplus=False,
                                                        softplus_gain=2.0))

    def assign_state_barrier(self, barrier):
        self._barrier = barrier
        return self

    def assign_dynamics(self, dynamics):
        self._dynamics = dynamics
        return self

    def safe_optimal_control(self, x, ret_info=False):
        x = tensify(x).to(torch.float64)
        Q = self._Q(x)
        c = self._c(x)
        Q_inv = torch.inverse(Q)

        assert Q.shape == (
            x.shape[0], self._action_dim, self._action_dim), 'Q should be of shape (batch_size, action_dim, action_dim)'
        assert c.shape == (x.shape[0], self._action_dim), 'c should be of shape (batch_size, action_dim)'

        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        hocbf -= self._params.buffer

        omega = Lf_hocbf - torch.einsum('bi,bij,bj->b', Lg_hocbf, Q_inv, c).unsqueeze(-1) + self._alpha(hocbf)
        den = torch.einsum('bi,bij,bj->b', Lg_hocbf, Q_inv, Lg_hocbf).unsqueeze(-1) + (
                1 / self._params.slack_gain) * hocbf ** 2
        num = softplus(-omega, beta=self._params.softplus_gain) if self._params.use_softplus else relu(-omega)
        lam = num / den

        u = - torch.einsum('bij,bi->bj', Q_inv, c - Lg_hocbf * lam)

        return u if not ret_info else self._add_optimal_control_info(u, hocbf, Lf_hocbf, Lg_hocbf, lam)

    def _add_optimal_control_info(self, u, hocbf, Lf_hocbf, Lg_hocbf, lam):
        info = {}
        info['slack_vars'] = hocbf * lam / self._params.slack_gain
        info['constraint_at_u'] = (Lf_hocbf + torch.einsum('bi,bi->b', Lg_hocbf, u).unsqueeze(-1) +
                                   self._alpha(hocbf) + info['slack_vars'] * hocbf)
        return u, info

    def eval_barrier(self, x):
        return self._barrier.hocbf(x)


class MinIntervCFSafeControl(BaseMinIntervSafeControl, CFSafeControl):
    """
    Minimum-Intervention-Closed-Form-Safe Control: This class inherits from CFSafeControl and introduces
    a minimum intervention cost approach. Rather than explicitly assigning cost matrices Q and c,
     this class directly works with the assinged desired control.
    """

    def safe_optimal_control(self, x, ret_info=False):
        x = tensify(x).to(torch.float64)

        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
        hocbf -= self._params.buffer

        u_d = self._desired_control(x)
        omega = Lf_hocbf + torch.einsum('bi,bi->b', Lg_hocbf, u_d).unsqueeze(-1) + self._alpha(hocbf)
        den = torch.einsum('bi,bi->b', Lg_hocbf, Lg_hocbf).unsqueeze(-1) + (
                1 / self._params.slack_gain) * hocbf ** 2
        num = softplus(-omega, beta=self._params.softplus_gain) if self._params.use_softplus else relu(-omega)
        lam = num / den

        u = u_d + Lg_hocbf * lam

        return u if not ret_info else self._add_optimal_control_info(u, hocbf, Lf_hocbf, Lg_hocbf, lam)


class InputConstCFSafeControl(CFSafeControl):
    """
    Input-Constrained-Closed-Form-Safe Control
    """

    def __init__(self, action_dim, alpha=None, params=None):
        super().__init__(action_dim, alpha, params)
        self._state_barrier = None
        self._ac_barrier = None

    def assign_dynamics(self, dynamics):
        raise "Use 'assign_state_action_dynamics' method to assign state and action dynamics"

    def assign_state_action_dynamics(self, state_dynamics,
                                     action_dynamics,
                                     action_output_function=lambda x: x):
        self._state_dyn = state_dynamics
        self._ac_dyn = action_dynamics
        self._ac_out_func = action_output_function
        self._make_augmented_dynamics()
        return self

    def assign_state_barrier(self, barrier):
        self._state_barrier = barrier
        return self

    def assign_action_barrier(self, action_barrier, rel_deg):
        self._ac_barrier = action_barrier
        self._ac_rel_deg = rel_deg
        return self

    def make(self):
        # make augmented dynamics
        self._make_augmented_dynamics()

        # TODO: Add options for composition rule
        # make composed barrier function
        self._make_composed_barrier()

        # make auxiliary desired action
        self._make_aux_desired_action()
        return self

    def safe_optimal_control(self, x, ret_info=False):
        x = tensify(x).to(torch.float64)

        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)

        hocbf -= self._params.buffer

        u_d = self.aux_desired_action(x)
        omega = Lf_hocbf + torch.einsum('bi,bi->b', Lg_hocbf, u_d).unsqueeze(-1) + self._alpha(hocbf)
        den = (torch.einsum('bi,bi->b', Lg_hocbf, Lg_hocbf).unsqueeze(-1) +
               (1 / self._params.slack_gain) * hocbf ** 2)
        num = softplus(-omega, beta=self._params.softplus_gain) if self._params.use_softplus else relu(-omega)
        lam = num / den

        u = u_d + Lg_hocbf * lam

        return u if not ret_info else self._add_optimal_control_info(u, hocbf, Lf_hocbf, Lg_hocbf, lam)

    # Helper functions
    def _make_composed_barrier(self):
        # Remake state barriers with the augmented dynamics
        self._state_barrier = [barrier.assign_dynamics(self._dynamics) for barrier in self._state_barrier]
        # Remake action barriers witht the augmented dynamics
        self._ac_barrier = [barrier.assign_dynamics(self._dynamics) for barrier in self._ac_barrier]
        self._barrier = (SoftCompositionBarrier(
            cfg=AD(softmin_rho=self._params.softmin_rho,
                         softmax_rho=self._params.softmax_rho)).assign_dynamics(
            self._dynamics).assign_barriers_and_rule(
            barriers=[*self._state_barrier, *self._ac_barrier], rule='i'))

    def _make_aux_desired_action(self):
        assert len(self._params.sigma) == self._ac_rel_deg + 1, "sigma must be of length 1 + action relative degree"

        # make desired control
        self._make_desired_control()

        ac_out_func = lambda x: self._ac_out_func(x[:, self._state_dyn.state_dim:])

        desired_control_lie_derivs = make_higher_order_lie_deriv_series(func=self._desired_control,
                                                                        field=self._dynamics.f,
                                                                        deg=self._ac_rel_deg)

        ac_out_func_lie_derivs = make_higher_order_lie_deriv_series(func=ac_out_func,
                                                                    field=self._dynamics.f,
                                                                    deg=self._ac_rel_deg)

        ac_out_Lg = lambda x: torch.inverse(lie_deriv(x, func=ac_out_func_lie_derivs[-2], field=self._dynamics.g))

        self.aux_desired_action = lambda x: torch.einsum('bmn, bm->bn', ac_out_Lg(x),
                                                         torch.stack([sigma * (dc(x) - of(x)) for dc, of, sigma in
                                                                      zip(desired_control_lie_derivs,
                                                                          ac_out_func_lie_derivs,
                                                                          self._params.sigma)]).sum(0)).detach()

    def _make_desired_control(self):
        self._desired_control = lambda x: -torch.einsum('bij,bi->bj',
                                                        torch.inverse(self._Q(x[:, :self._state_dyn.state_dim])),
                                                        self._c(x[:, :self._state_dyn.state_dim]))

    def _make_augmented_dynamics(self):
        assert (self._state_dyn.action_dim == self._ac_dyn.action_dim), \
            ('Dimension mismatch')

        aug_state_dim = self._state_dyn.state_dim + self._ac_dyn.state_dim
        aug_action_dim = self._state_dyn.action_dim

        self._dynamics = AffineInControlDynamics(state_dim=aug_state_dim,
                                                 action_dim=aug_action_dim)
        aug_f = lambda x: torch.hstack((self._state_dyn.rhs(x=x[:, :self._state_dyn.state_dim],
                                                            action=self._ac_out_func(x[:, self._state_dyn.state_dim:])),
                                        self._ac_dyn.f(x[:, self._state_dyn.state_dim:])))
        self._dynamics.set_f(aug_f)

        aug_g = lambda x: torch.cat((torch.zeros(x.shape[0],
                                                 self._state_dyn.state_dim,
                                                 self._state_dyn.action_dim, dtype=torch.float64),
                                     self._ac_dyn.g(x[:, self._state_dyn.state_dim:])), dim=1)

        self._dynamics.set_g(aug_g)


class MinIntervInputConstCFSafeControl(BaseMinIntervSafeControl, InputConstCFSafeControl):
    def assign_desired_control(self, desired_control):
        self._desired_control = desired_control
        self.make()
        return self

    def _make_desired_control(self):
        pass


class MinIntervInputConstCFSafeControlRaw(InputConstCFSafeControl):
    """
    This class considers the desired control as the auxiliary desired control.
    """

    def assign_desired_control(self, desired_control):
        self._desired_control = desired_control
        self.aux_desired_action = desired_control
        self.make()
        return self

    def _make_desired_control(self):
        pass

    def _make_aux_desired_action(self):
        pass
