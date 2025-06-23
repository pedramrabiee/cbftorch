import torch
from cbftorch.safe_controls.qp_safe_control import *

from cbftorch.utils.utils import *

class BackupSafeControl(InputConstQPSafeControl):

    def assign_state_barrier(self, barrier):
        self._barrier = barrier
        self.barrier_cfg=barrier.cfg
        return self

    def safe_optimal_control(self, x, ret_info=False):

        x = tensify(x).to(torch.float64)

        hocbf, Lf_hocbf, Lg_hocbf = self._barrier.get_hocbf_and_lie_derivs(x)
        feas_fact = self._get_feasibility_factor(x, Lf_hocbf, Lg_hocbf, hocbf)

        h_star_vals = self._barrier.get_h_stars(x)
        if len(self._barrier.backup_policies) > 1:
            ub_vals = torch.stack([self._barrier.backup_policies[i](x) for i in range(len(self._barrier.backup_policies))])
            ub_blend = self._get_backup_blend(h_star_vals, ub_vals)
            max_val, max_ind = torch.max(h_star_vals, dim=0)  # (201, 1)
            ub_select = torch.where(max_val <= self.barrier_cfg.epsilon,
                                    torch.gather(ub_vals, dim=0, index=max_ind.unsqueeze(0)).squeeze(0),
                                    ub_blend)
        else:
            ub_select = self._barrier.backup_policies[0](x)

        gamma = torch.min((hocbf - self.barrier_cfg.epsilon) / self.barrier_cfg.h_scale, feas_fact / self.barrier_cfg.feas_scale)
        gamma_pos_ind = (gamma>= 0).flatten()
        gamma_neg_ind = (gamma < 0).flatten()

        u_star = torch.zeros_like(ub_select)
        u_star[gamma_neg_ind, ...] = ub_select[gamma_neg_ind, ...]

        if torch.any(gamma_pos_ind):
            masked_x = x[gamma_pos_ind, ...]
            Q, c = self._make_objective(x)
            Q = Q[gamma_pos_ind, ...]
            c = c[gamma_pos_ind, ...]
            ac_G = self._ac_G(masked_x)
            ac_h = self._ac_h(masked_x)
            G = torch.cat([-Lg_hocbf[gamma_pos_ind, ...].unsqueeze(1), ac_G], dim=1)
            h = torch.cat([(Lf_hocbf[gamma_pos_ind, ...] + self._alpha(hocbf[gamma_pos_ind, ...] - self.barrier_cfg.epsilon)), ac_h], dim=1)
            A, b = self._make_eq_const(masked_x, Q.shape)
            u_star[gamma_pos_ind, ...] = QPFunction()(Q, c, G, h, A, b)

        beta = torch.where(gamma > 0,
                           torch.where(gamma >= 1, 1, gamma),
                           0)

        u = (1 - beta) * ub_select + beta * u_star

        if not ret_info:
            return u

        info={}
        constraint_val = torch.einsum('bi,bi->b', Lg_hocbf, u).unsqueeze(-1) + (Lf_hocbf + self._alpha(hocbf - self.barrier_cfg.epsilon))
        info['constraint_val'] = constraint_val
        info['u_star'] = u_star
        info['ub_select'] = ub_select
        info['feas_fact'] = feas_fact
        info['beta'] = beta
        return u, info


    def safe_optimal_control_slacked(self, x, ret_info=False):
        raise NotImplementedError


    def _get_backup_blend(self, h_star_vals, ub_vals):
        h_star_ind = torch.where(h_star_vals >= self.barrier_cfg.epsilon)[0]
        valid_ub = ub_vals[h_star_ind, ...]
        num = torch.sum((h_star_vals[h_star_ind,...] - self.barrier_cfg.epsilon) * valid_ub, dim=0)
        den = torch.sum((h_star_vals[h_star_ind,...] - self.barrier_cfg.epsilon), dim=0)
        return num / den

    def _get_feasibility_factor(self, x, Lf_hocbf, Lg_hocbf, hocbf):
        lp_sol = lp_solver(-Lg_hocbf, self._ac_G(x), self._ac_h(x))
        return Lf_hocbf + self._alpha(hocbf - self.barrier_cfg.epsilon) +  torch.einsum('bi,bi->b', Lg_hocbf, lp_sol).unsqueeze(-1)

class MinIntervBackupSafeControl(MinIntervQPSafeControl,BackupSafeControl):
    pass