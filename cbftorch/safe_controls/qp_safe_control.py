import torch
from qpth.qp import QPFunction
from cbftorch.safe_controls.base_safe_control import BaseSafeControl, BaseMinIntervSafeControl
from cbftorch.utils.utils import *
from cbftorch.safe_controls.closed_form_safe_control import InputConstCFSafeControl
from torch.nn.functional import pad

class QPSafeControl(BaseSafeControl):
    def __init__(self, action_dim, alpha=None, params=None):
        super().__init__(action_dim, alpha, params)

    def assign_state_barrier(self, barrier):
        self._barrier = barrier
        return self

    def assign_dynamics(self, dynamics):
        self._dynamics = dynamics
        return self

    def safe_optimal_control(self, x, ret_info=False):
        if self._params.slacked:
            return self.safe_optimal_control_slacked(x, ret_info)

        x = tensify(x).to(torch.float64)
        Q, c = self._make_objective(x)
        G, h = self._make_ineq_const(x)
        A, b = self._make_eq_const(x, Q.shape)

        u = QPFunction()(Q, c, G, h, A, b)
        if not ret_info:
            return u

        info = {}
        constraint_at_u = torch.einsum("brm,bm->br", -G, u) + h
        info['constraint_at_u'] = constraint_at_u

        return u, info


    def safe_optimal_control_slacked(self, x, ret_info=False):
        x = tensify(x).to(torch.float64)

        G, h = self._make_ineq_const_slacked(x)
        num_constraints = h.shape[-1]
        Q, c = self._make_objective_slacked(x, num_constraints)
        A, b = self._make_eq_const(x, Q.shape)

        res = QPFunction()(Q, c, G, h, A, b)
        u = res[:, :self._action_dim]

        if not ret_info:
            return u

        info = {}
        constraint_at_u = torch.einsum("brm,bm->br", -G, res) + h
        slack_vars = res[:, self._action_dim:]

        info['constraint_at_u'] = constraint_at_u
        info['slack_vars'] = slack_vars
        return u, info


    def _make_objective(self, x):
        Q = self._Q(x)
        c = self._c(x)

        assert Q.shape == (
            x.shape[0], self._action_dim, self._action_dim), 'Q should be of shape (batch_size, action_dim, action_dim)'
        assert c.shape == (x.shape[0], self._action_dim), 'c should be of shape (batch_size, action_dim)'
        return Q, c

    def _make_objective_slacked(self, x, num_constraints):
        Q, c = self._make_objective(x)
        # trick to do block diag on each batch dimension
        batch_size = x.shape[0]
        slack_quad_gains = self._params.slack_gain * 0.5 * torch.eye(num_constraints).repeat(batch_size, 1, 1)
        slack_linear_gains = torch.zeros(batch_size, num_constraints)
        Q = (pad(Q, (0, num_constraints, 0, num_constraints)) +
             pad(slack_quad_gains, (self._action_dim, 0, self._action_dim, 0)))
        c = torch.cat((c, slack_linear_gains), dim=-1)
        return Q, c

    def _make_ineq_const(self, x):
        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
        G = -Lg_hocbf
        h = (Lf_hocbf + self._alpha(hocbf)).squeeze(-1)
        return G, h

    def _make_ineq_const_slacked(self, x):
        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
        slack_weights = torch.diag_embed(hocbf.squeeze(-1), 0)
        G = torch.cat((-Lg_hocbf, -slack_weights), dim=-1)
        h = (Lf_hocbf + self._alpha(hocbf)).squeeze(-1)
        return G, h


    def _make_eq_const(self, x, Q_shape):
        A = torch.empty(Q_shape[0], 0, Q_shape[1])
        b = torch.empty(Q_shape[0], 0)
        return A, b

class MinIntervQPSafeControl(BaseMinIntervSafeControl, QPSafeControl):
    def assign_desired_control(self, desired_control):
        self._Q = lambda x: 2 * torch.eye(self._action_dim, dtype=torch.float64).repeat(x.shape[0], 1, 1)
        self._c = lambda x: -2 * desired_control(x)
        self._desired_control = desired_control



class InputConstQPSafeControl(QPSafeControl):
    def assign_control_bounds(self, low: list, high: list):
        assert len(low) == len(high), 'low and high should have the same length'
        assert len(low) == self._action_dim, 'low and high length should match action dimension'

        ac_low_G = lambda x: -torch.eye(self._action_dim, dtype=torch.float64).repeat(x.shape[0], 1, 1)
        ac_low_h = lambda x: -torch.tensor(low).to(torch.float64).repeat(x.shape[0], 1)

        ac_high_G = lambda x: -ac_low_G(x)
        ac_high_h = lambda x: torch.tensor(high).to(torch.float64).repeat(x.shape[0], 1)

        self._ac_G = lambda x: torch.cat([ac_low_G(x), ac_high_G(x)], dim=1)
        self._ac_h = lambda x: torch.cat([ac_low_h(x), ac_high_h(x)], dim=1)

        return self

    def safe_optimal_control(self, x, ret_info=False):
        if self._params.slacked:
            return self.safe_optimal_control_slacked(x, ret_info)

        x = tensify(x).to(torch.float64)
        Q, c = self._make_objective(x)

        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
        ac_G = self._ac_G(x)
        ac_h = self._ac_h(x)
        G = torch.cat([-Lg_hocbf, ac_G], dim=1)
        h = torch.cat([(Lf_hocbf + self._alpha(hocbf)).squeeze(-1), ac_h], dim=1)

        A, b = self._make_eq_const(x, Q.shape)

        u = QPFunction()(Q, c, G, h, A, b)

        if not ret_info:
            return u

        info = {}
        constraint_at_u = torch.einsum("brm,bm->br", -G, u) + h
        info['constraint_at_u'] = constraint_at_u

        return u, info


    def safe_optimal_control_slacked(self, x, ret_info=False):
        x = tensify(x).to(torch.float64)

        G, h = self._make_ineq_const_slacked(x)
        num_constraints = h.shape[-1]
        Q, c = self._make_objective_slacked(x, num_constraints)

        ac_G = pad(self._ac_G(x), (0, num_constraints))
        G = torch.cat((G, ac_G), dim=-2)
        h = torch.cat((h, self._ac_h(x)), dim=-1)

        A, b = self._make_eq_const(x, Q.shape)

        res = QPFunction()(Q, c, G, h, A, b)
        u = res[:, :self._action_dim]
        if not ret_info:
            return u

        info = {}
        constraint_at_u = torch.einsum("brm,bm->br", -G, res) + h
        slack_vars = res[:, self._action_dim:]

        info['constraint_at_u'] = constraint_at_u
        info['slack_vars'] = slack_vars
        return u, info


class MinIntervInputConstQPSafeControl(MinIntervQPSafeControl,
                                       InputConstQPSafeControl):
    pass
