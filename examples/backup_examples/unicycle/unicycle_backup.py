from box import Box as AD
import matplotlib as mpl

from cbftorch.dynamics import UnicycleReducedOrderDynamics
from map_config import map_config
from cbftorch.barriers.backup_barrier import BackupBarrier
from cbftorch.utils.make_map import Map
from backup_policies import UnicycleBackupControl
from cbftorch.barriers.barrier import Barrier
from cbftorch.safe_controls.backup_safe_control import MinIntervBackupSafeControl
from unicycle_desired_control import desired_control
from cbftorch.utils.utils import *

from time import time
import datetime
import matplotlib.pyplot as plt


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'


# Backup configs
cfg = AD(softmax_rho=50,
         softmin_rho=50,
         pos_barrier_rel_deg=1,
         boundary_alpha=[],
         obstacle_alpha=[],
         velocity_alpha=[],
         horizon=1.0,
         time_steps=0.02,
         method='dopri5',
         epsilon=0.0,
         h_scale=0.012,
         feas_scale=0.05,
         )

# Map configs
map_cfg = AD(
         softmin_rho=20,
         boundary_alpha=[],
         obstacle_alpha=[],
         velocity_alpha=[],
         pos_barrier_rel_deg=1,
         )

torch.set_default_dtype(torch.float64)
control_bounds = torch.tensor([[-4.0, 4.0], [-1.0, 1.0]], dtype=torch.float64)
ub_gain = torch.tensor([[-15.0, 0.0]], dtype=torch.float64)
dynamics_param = AD(d=1, control_bounds=control_bounds)

control_gains = dict(k1=1.0, k2=0.8)


# # Goal positions
goal_pos = torch.tensor([
    [2.0, 4.5],
    [-1.0, 0.0],
    [-4.5, 8.0],
])


# goal_pos = torch.tensor([2.0, 4.5], dtype=torch.float64).unsqueeze(0)

# Initial Conditions
x0 = torch.tensor([[-3.0, -8.5, 0.0, 0.0]], dtype=torch.float64).repeat(goal_pos.shape[0], 1)






# Instantiate dynamics
dynamics = UnicycleReducedOrderDynamics(params=dynamics_param)


# Make barrier from map_
map_ = Map(dynamics=dynamics, cfg=map_cfg, barriers_info=map_config)

state_barrier= map_.barrier

backup_controls = UnicycleBackupControl(ub_gain, control_bounds)()

backup_barrier_functional = [lambda x: state_barrier.hocbf(x) - (100* torch.pow(x[:,2], 2)).unsqueeze(-1) / control_bounds.amax(dim=1)[0]]

backup_barriers = [Barrier().assign(barrier_func=func,
                                 rel_deg=1,
                                 alphas=[]).assign_dynamics(dynamics) for func in backup_barrier_functional]

fwd_barrier = (BackupBarrier(cfg).assign_state_barrier([state_barrier]).assign_backup_policies(backup_controls)
           .assign_backup_barrier(backup_barriers).assign_dynamics(dynamics).make())

fwd_barrier.barrier(x0)

safety_filter = (MinIntervBackupSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 1.0 * x,
    params=AD(slack_gain=1e24,
              slacked=False,
              use_softplus=False,
              softplus_gain=2.0)
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=fwd_barrier)
                 .assign_control_bounds(low=control_bounds[:, 0].tolist(), high=control_bounds[:, 1].tolist()))





timestep = 0.02
sim_time = 20.0

# assign desired control based on the goal positions
safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos, dyn_params=dynamics_param ,**control_gains)(x))
)

# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs_zoh(x0=x0, sim_time=sim_time, timestep=timestep, method='dopri5')
print(time() - start_time)

# Rearrange trajs
trajs = [torch.vstack(t.split(dynamics.state_dim)) for t in torch.hstack([tt for tt in trajs])]





# Get actions values along the trajs
actions = []
h_vals = []
h_s = []
feas_factor = []
constraints_val =[]
u_star = []
ub_select = []
ud = []
beta=[]
h_star =[]
des_ctrls = []
for i, traj in enumerate(trajs):
    des_ctrl = lambda x: vectorize_tensors(
        partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1), dyn_params=dynamics_param, **control_gains)(x))
    safety_filter.assign_desired_control(desired_control=des_ctrl)
    action, info = safety_filter.safe_optimal_control(traj, ret_info=True)
    hocbf, Lf_hocbf, Lg_hocbf = safety_filter.barrier.get_hocbf_and_lie_derivs(traj.to(torch.float64))
    actions.append(action)
    h_vals.append(safety_filter.barrier.hocbf(traj).detach())
    h_s.append(state_barrier.hocbf(traj).detach())
    h_star.append(safety_filter.barrier.get_h_star(traj).detach())
    des_ctrls.append(des_ctrl(traj))

    feas_factor.append(info['feas_fact'])
    constraints_val.append(info['constraint_val'])
    u_star.append(info['u_star'])
    ub_select.append(info['ub_select'])
    beta.append(info['beta'])
############
#  Plots   #
############

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

x = np.linspace(-10 - 0.5, 10 +0.5, 500)
y = np.linspace(-10 - 0.5, 10 +0.5, 500)
X, Y = np.meshgrid(x, y)
points = np.column_stack((X.flatten(), Y.flatten()))
points = torch.tensor(points, dtype=torch.float64)
points = torch.cat((points, torch.zeros(points.shape[0], 1, dtype=torch.float64)), dim=-1)
Z = map_.barrier.min_barrier(points)
Z = Z.reshape(X.shape)


fig, ax = plt.subplots(figsize=(6, 6))

contour_plot = ax.contour(X, Y, Z, levels=[0], colors='red')



ax.set_xlabel(r'$\theta$', fontsize=14)
ax.set_ylabel(r'$\dot{\theta}$', fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal', adjustable='box')  # Maintain equal aspect ratio
ax.tick_params(axis='x', labelsize=12)  # Font size for x-axis ticks
ax.tick_params(axis='y', labelsize=12)  # Font size for y-axis ticks



for i in range(len(trajs)):
    ax.plot(trajs[i][0, 0], trajs[0][0, 1], 'x', color='blue', markersize=8, label=r'$x_0$' if i == 0 else None)
    ax.plot(trajs[i][-1, 0], trajs[i][-1, 1], '+', color='blue', markersize=8, label=r'$x_f$' if i == 0 else None)
    ax.plot(trajs[i][:, 0], trajs[i][:, 1], label='Trajectories' if i == 0 else None, color='black')
    ax.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='limegreen', label='Goal' if i == 0 else None)

ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)
plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/Trajectories_QP_Backup_Safe_Control_{current_time}.png', dpi=600)
plt.show()

# Calculate time array based on the number of data points and timestep
num_points = trajs[0].shape[0]  # Assuming trajs has the same length for all elements
time = np.linspace(0.0, (num_points - 1) * timestep, num_points)

# Create subplot for trajs and action variables
fig, axs = plt.subplots(6, 1, figsize=(8, 8))

# Plot trajs variables
axs[0].plot(time, trajs[0][:, 0], label=r'$r_x$', color='black')
axs[0].set_ylabel(r'$r_x$', fontsize=14)

axs[1].plot(time, trajs[0][:, 1], label=r'$r_y}$', color='black')
axs[1].set_ylabel(r'$r_y$', fontsize=14)

axs[2].plot(time, trajs[0][:, 2], label=r'$v$', color='black')
axs[2].set_ylabel(r'$v$', fontsize=14)

axs[3].plot(time, trajs[0][:, 3], label=r'$\theta$', color='black')
axs[3].set_ylabel(r'$\theta$', fontsize=14)

axs[4].plot(time, actions[0][:, 0], label='u', color='black')
axs[4].plot(time, ub_select[0][:, 0], label=r'$u_b$', color='green')
axs[4].plot(time, u_star[0][:, 0], label=r'$u_*$', color='blue')
axs[4].plot(time, des_ctrls[0][:, 0], label=r'$u_d$', color='red', linestyle='--')
axs[4].legend(fontsize=14, loc='upper right', frameon=False, ncol=4)
axs[4].set_ylabel(r'$u_1$', fontsize=14)

axs[5].plot(time, actions[0][:, 1], label='u', color='black')
axs[5].plot(time, ub_select[0][:, 1], label=r'$u_b$', color='green')
axs[5].plot(time, u_star[0][:, 1], label=r'$u_*$', color='blue')
axs[5].plot(time, des_ctrls[0][:, 1], label=r'$u_d$', color='red', linestyle='--')
axs[5].legend(fontsize=14, loc='upper right', frameon=False, ncol=4)
axs[5].set_ylabel(r'$u_2$', fontsize=14)

axs[5].set_xlabel(r'$t~(\rm {s})$', fontsize=14)
ax.legend()
plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/Controls_QP_Backup_Safe_Control_{current_time}.png', dpi=600)
plt.show()



# Create subplot for barrier variables
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# Plot barrier variables
axs[0].plot(time, h_vals[0][:, 0], label=r'$h$', color='blue')
axs[0].plot(time, h_star[0][:, 0], label=r'$\bar h_*$', color='orange', linestyle='--')
axs[0].plot(time, h_s[0][:, 0], label=r'$h_s$', color='red', linestyle='--')
axs[0].plot(time, torch.zeros(time.shape[0], 1), color='green', linestyle='dotted')
axs[0].set_ylabel(r'$h$', fontsize=14)
axs[0].legend(fontsize=14, loc='best', frameon=False)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
axs[0].set_yscale('log')

axs[1].plot(time, (h_vals[0][:, 0] - cfg.epsilon)/cfg.h_scale, label=r'$\frac{h - \epsilon}{\kappa_h}$', color='blue')
axs[1].plot(time, (feas_factor[0][:, 0])/cfg.feas_scale, label=r'$\frac{\beta}{\kappa_\beta}$', color='red')
axs[1].plot(time, torch.zeros(time.shape[0], 1), color='green')
axs[1].set_ylabel(r'$\frac{h - \epsilon}{\kappa_h}, \frac{\beta}{\kappa_\beta}$', fontsize=14)
axs[1].legend(fontsize=14, loc='best', frameon=False)
axs[1].set_yscale('log')


axs[2].plot(time, beta[0][:, 0], label=r'$\sigma$', color='blue')

axs[2].set_ylabel(r'$\sigma$', fontsize=14)
axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=14)
ax.legend()
plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/Barriers_QP_Backup_Safe_Control_{current_time}.png', dpi=600)
plt.show()

