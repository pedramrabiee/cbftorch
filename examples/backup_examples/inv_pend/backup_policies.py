
import torch


class PendulumBackupControl:
    def __init__(self, gain, center, ac_lim):
        assert len(gain) == len(center), 'number of gain an center must be equal'
        assert isinstance(gain, torch.Tensor), 'gain must be a tensor'
        assert isinstance(center, torch.Tensor), 'center must be a tensor'


        control_num = gain.shape[0]
        self.gain = gain
        self.center = center
        self.ac_lim = torch.max(ac_lim)

        self.u_eq = self.get_u_equilibrium().unsqueeze(-1)

        # Create and store the list of functions during initialization
        self.control_functions = [
            lambda x, g=self.gain[i, :], u_eq=self.u_eq[i, :]: self.ac_lim * torch.tanh((u_eq + torch.einsum('bn,n->b', x, g).unsqueeze(-1)) / self.ac_lim)
            for i in range(control_num)]

    def __call__(self):
        return self.control_functions

    def get_u_equilibrium(self):
        return self.ac_lim * torch.atanh(
            (-1/self.ac_lim * torch.sin(self.center[:, 0])) / self.ac_lim) - self.gain[:,
            0] * self.center[:, 0]