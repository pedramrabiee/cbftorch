import torch
from math import pi



# Desired control definition

def desired_control(x, goal_pos, k1=2.0):
    ud1 = -k1 * (x[:, 0:1] - goal_pos[:, 0:1])
    ud2 = -k1 * (x[:, 1:2] - goal_pos[:, 1:2])
    return torch.hstack([ud1, ud2])