import torch



# Desired control definition

def desired_control(x, goal_pos, dyn_params, k1=0.8, k2=0.8):

    max_ac_lim = torch.amax(dyn_params.control_bounds, dim=1)
    s, c = torch.sin(x[:,-1]), torch.cos(x[:,-1])
    rot_mat = torch.stack([c, s, -s, c], dim=1).view(-1, 2, 2)
    dist_to_goal = x[:, :2] - goal_pos[:, :2]
    e = torch.einsum('bij,bj ->bi', rot_mat, dist_to_goal)

    q_x, q_y, v, theta = x[:, 0], x[:, 1], x[:, 2], x[:, 3]

    vd = -(k1 + k2)* v - (1+ k1*k2) * e[:,0] + torch.pow(e[:,1] * k1, 2) / dyn_params.d
    wd = -k1 / dyn_params.d * e[:,1]
    ud1 = max_ac_lim[0] * torch.tanh(vd)
    ud2 = torch.where(torch.norm(dist_to_goal) > 0.1, max_ac_lim[1] * torch.tanh(wd), 0.0)
    return torch.hstack([ud1.unsqueeze(-1), ud2.unsqueeze(-1)])