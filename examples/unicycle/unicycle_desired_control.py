import torch
from math import pi



# Desired control definition

def desired_control(x, goal_pos, k1=0.2, k2=1.0, k3=2.0):
    dist_to_goal = torch.norm(x[:, :2] - goal_pos[:, :2], dim=-1)
    q_x, q_y, v, theta = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    psi = torch.atan2(q_y - goal_pos[:, 1], q_x - goal_pos[:, 0]) - theta + pi
    ud1 = (-(k1 + k3) * v + (1 + k1 * k3) * dist_to_goal * torch.cos(psi) +
           k1 * (k2 * dist_to_goal + v) * torch.sin(psi) ** 2)

    ud2 = torch.where(dist_to_goal > 0.1, (k2 + v / dist_to_goal) * torch.sin(psi), 0.0)
    return torch.hstack([ud1.unsqueeze(-1), ud2.unsqueeze(-1)])