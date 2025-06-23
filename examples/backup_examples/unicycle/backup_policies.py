import torch


class UnicycleBackupControl:
    def __init__(self, gain, ac_lim):
        assert isinstance(gain, torch.Tensor), 'gain must be a tensor'


        control_num = gain.shape[0]
        self.gain = gain
        self.ac_lim= torch.amax(ac_lim, dim=1)

        # Create and store the list of functions during initialization
        self.control_functions = [
            lambda x, g=self.gain[i, :]: torch.stack((self.ac_lim[0] * torch.tanh(g[0] * x[:,2]), torch.zeros(x.shape[0], dtype=torch.float64)), dim=-1)
            for i in range(control_num)]

    def __call__(self):
        return self.control_functions