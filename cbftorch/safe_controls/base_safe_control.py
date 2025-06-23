from cbftorch.utils.utils import *
from box import Box as AD


class BaseSafeControl:
    def __init__(self, action_dim, alpha=None, params=None):
        self._action_dim = action_dim
        self._alpha = alpha if alpha is not None else lambda x: x
        self._params = params
        self._dynamics = None
        self._barrier = None
        self._Q = None
        self._c = None

        if self._params is None:
            self._params = AD(buffer=0.0)
        if 'buffer' not in self._params:
            self._params['buffer'] = 0.0

    def assign_state_barrier(self, barrier):
        raise NotImplementedError

    def assign_dynamics(self, dynamics):
        raise NotImplementedError

    def assign_cost(self, Q, c):
        self._Q = Q
        self._c = c
        return self

    def safe_optimal_control(self, x, ret_info=False):
        raise NotImplementedError

    def get_safe_optimal_trajs(self, x0, timestep=0.001, sim_time=4.0, intermediate_steps=None, method='dopri5'):
        return get_trajs_from_action_func(x0=x0, dynamics=self._dynamics,
                                          action_func=self.safe_optimal_control,
                                          timestep=timestep, sim_time=sim_time,
                                          method=method)

    def get_safe_optimal_trajs_zoh(self, x0, timestep=0.001, sim_time=4.0, intermediate_steps=2, method='dopri5'):
        return get_trajs_from_action_func_zoh(x0=x0, dynamics=self._dynamics,
                                              action_func=self.safe_optimal_control,
                                              timestep=timestep, sim_time=sim_time,
                                              intermediate_steps=intermediate_steps,
                                              method=method)

    @property
    def barrier(self):
        return self._barrier


class BaseMinIntervSafeControl(BaseSafeControl):
    def assign_desired_control(self, desired_control):
        self._desired_control = desired_control
        return self

    def assign_cost(self, Q, c):
        raise ('Use assign_desired_control to assign desired control.'
               ' The min intervention cost is automatically assigned.')


    def get_desired_control_trajs(self, x0, timestep=0.001, sim_time=4.0, method='dopri5'):
        return get_trajs_from_action_func(x0=x0, dynamics=self._dynamics,
                                          action_func=self._desired_control,
                                          timestep=timestep, sim_time=sim_time,
                                          method=method)

    def get_desired_control_trajs_zoh(self, x0, timestep=0.001, sim_time=4.0, method='dopri5'):
        return get_trajs_from_action_func_zoh(x0=x0, dynamics=self._dynamics,
                                              action_func=self._desired_control,
                                              timestep=timestep, sim_time=sim_time,
                                              method=method)
