import torch
from cbftorch.barriers.barrier import Barrier
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier
from cbftorch.utils.utils import *


class BackupBarrier(Barrier):

    def __init__(self, cfg):
        super(BackupBarrier, self).__init__()
        self.cfg = cfg
        self._rel_deg = 1

        self._backup_barriers = None
        self._backup_policies = None
        self._state_barrier = None
        self.action_num = None

        self.h_star = None
        self.h_stars = None

        self._last_x = None
        self._last_h_list = None


    def assign(self):
        raise "Use assign_state_barrier and assign_backup_barrier"


    def assign_state_barrier(self, state_barrier):

        assert isinstance(state_barrier, list), 'state_barrier must be List'
        self._state_barrier = SoftCompositionBarrier(
            cfg=self.cfg).assign_barriers_and_rule(barriers=[*state_barrier],
                                                   rule='i',
                                                   infer_dynamics=True)
        return self



    def assign_backup_policies(self, backup_policies):
        assert isinstance(backup_policies, list), 'backup_policies must be list'
        assert len(backup_policies) > 0, 'backup_policies must have at least one item'
        assert all(callable(f) for f in backup_policies), "all backup policies must be callable"

        self._backup_policies = backup_policies
        return self



    def assign_backup_barrier(self, backup_barriers):
        assert isinstance(backup_barriers, list), 'backup_barrier must be List'
        assert len(backup_barriers) > 0, 'backup_barrier must have at least one item'
        assert all(isinstance(f, Barrier) for f in backup_barriers), "all backup barriers must be Barrier"

        self._backup_barriers = backup_barriers
        return self



    def assign_dynamics(self, dynamics):
        self._dynamics = dynamics
        return self


    def _backup_barrier_func(self, x, ret_info=False):
        trajs = self.get_backup_traj(x).chunk(self.action_num, dim=1)
        b, m, n = trajs[0].shape
        self._state_barrier.hocbf(trajs[0].reshape(-1, n))
        h_list = [
            torch.cat((self._state_barrier.hocbf(traj.reshape(-1, n)).reshape(b, m, -1),
                       backup_barrier.hocbf(traj[-1, ...]).unsqueeze(0)))
            for traj, backup_barrier in zip(trajs, self._backup_barriers)]
        h_values = torch.stack([softmin(hh, self.cfg.softmin_rho, dim=0) for hh in h_list])
        final_h_val = softmax(h_values, self.cfg.softmax_rho, dim=0)
        ## Cache the results
        self._last_x = x
        self._last_h_list = h_list
        if not ret_info:
            return final_h_val
        else:
            self.h_stars = torch.stack([torch.amin(hh, dim=0) for hh in h_list])
            self.h_star =  torch.amax(self.h_stars, dim=0)
            return dict(h_star=self.h_star, h_stars=self.h_stars, h=final_h_val)



    def make(self):

        assert self.state_barrier is not None, \
            "State Barrier must be assigned. Use the assign_state_barrier method"

        assert self._backup_barriers is not None, \
            "Backup Barrier must be assigned. Use the assign_backup_barrier method"

        assert self._backup_policies is not None, \
            "Backup policies must be assigned. Use the assign_backup_policies method"

        assert self.dynamics is not None, "Dynamics must be assigned. Use the assign_dynamics method"

        assert len(self._backup_policies) == len(
            self._backup_barriers), "Backup policies number must match backup barrier number"

        self.action_num = len(self._backup_policies)

        self._barrier_func = self._backup_barrier_func

        # make higher-order barrier function
        self._barriers = self._make_hocbf_series(barrier=self.barrier, rel_deg=self._rel_deg, alphas=[])
        self._hocbf_func = self._barriers[-1]
        return self


    def get_backup_traj(self, x):
        traj = get_trajs_from_batched_action_func(x, self.dynamics, self._backup_policies,
                                                      self.cfg.time_steps, self.cfg.horizon,
                                                      self.cfg.method)
        return traj



    def raise_rel_deg(self):
        raise NotImplementedError


    def get_h_stars(self, x):
        if torch.all(torch.isclose(x, self._last_x)):
            self.h_stars = torch.stack([torch.amin(hh, dim=0) for hh in self._last_h_list])
            return self.h_stars
        else:
            self.h_stars = self._backup_barrier_func(x, ret_info=True)('h_stars')
            return self.h_stars

    def get_h_star(self, x):
        if torch.all(torch.isclose(x, self._last_x)):
            self.h_star = torch.amax(self.h_stars, dim=0)
            return self.h_star
        else:
            self.h_star = self._backup_barrier_func(x, ret_info=True)('h_star')
            return self.h_star

    @property
    def backup_policies(self):
        return self._backup_policies

    @property
    def backup_barriers(self):
        return self._backup_barriers

    @property
    def state_barrier(self):
        return self._state_barrier