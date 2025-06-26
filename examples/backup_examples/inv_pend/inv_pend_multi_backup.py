

from box import Box as AD
import matplotlib as mpl
from math import pi


from cbftorch.dynamics import InvertPendDynamics
from map_config import map_config2
from cbftorch.barriers.backup_barrier import BackupBarrier
from cbftorch.utils.make_map import Map
from backup_policies import PendulumBackupControl
from cbftorch.barriers.barrier import Barrier
from map_config import multibackup_barrier_functional
from cbftorch.safe_controls.backup_safe_control import MinIntervBackupSafeControl

from time import time
import datetime
import matplotlib.pyplot as plt
from cbftorch.utils.utils import *


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'


# Backup configs
cfg = AD(softmax_rho=50,
         softmin_rho=100,
         pos_barrier_rel_deg=1,
         boundary_alpha=[],
         horizon=5,
         time_steps=0.1,
         method='dopri5',
         epsilon=0.0,
         h_scale=0.05,
         feas_scale=0.05,
         )

torch.set_default_dtype(torch.float64)


control_bounds = torch.tensor([-1.5, 1.5]).to(torch.float64)


ub_gain = torch.tensor([[-3, -3],
           [-3, -3],
           [-3, -3]], dtype=torch.float64)

ub_center = torch.tensor([[0, 0],
             [pi/2, 0],
             [-pi/2, 0]]).to(torch.float64)



# Initial Conditions
x0 = torch.tensor([[0.5, 0.0],
                   [-2.7, 0.0]])




# Instantiate dynamics
dynamics = InvertPendDynamics()


# Make barrier from map_
map_ = Map(barriers_info=map_config2, dynamics=dynamics, cfg=cfg)

state_barrier = map_.barrier

backup_controls = PendulumBackupControl(ub_gain, ub_center, control_bounds)()
backup_barriers = [Barrier().assign(barrier_func=func,
                                 rel_deg=1,
                                 alphas=[]).assign_dynamics(dynamics) for func in multibackup_barrier_functional]

fwd_barrier = (BackupBarrier(cfg).assign_state_barrier(state_barrier).assign_backup_policies(backup_controls)
           .assign_backup_barrier(backup_barriers).assign_dynamics(dynamics).make())

safety_filter = (MinIntervBackupSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 1.0 * x,
    params=AD(slack_gain=1e24,
              slacked=False,
              use_softplus=False,
              softplus_gain=2.0)
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=fwd_barrier)
                 .assign_control_bounds(low=[control_bounds[0]], high=[control_bounds[1]]))



timestep = 0.1
sim_time = 20.0

# assign desired control based on the goal positions
safety_filter.assign_desired_control(
    desired_control=lambda x: torch.zeros(x.shape[0], 1, dtype=torch.float64, requires_grad=x.requires_grad))

# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs_zoh(x0=x0, sim_time=sim_time, intermediate_steps=5, timestep=timestep, method='dopri5')
print(time() - start_time)


# Rearrange trajs
trajs = [torch.vstack(t.split(dynamics.state_dim)) for t in torch.hstack([tt for tt in trajs])]





# Get actions values along the trajs
actions = []
h_vals = []
h_star =[]
h_s = []
feas_factor = []
constraints_val =[]
u_star = []
ub_select = []
ud = []
beta=[]
for i, traj in enumerate(trajs):
    action, info = safety_filter.safe_optimal_control(traj, ret_info=True)
    hocbf, Lf_hocbf, Lg_hocbf = safety_filter.barrier.get_hocbf_and_lie_derivs(traj.to(torch.float64))
    actions.append(action.detach())
    h_vals.append(safety_filter.barrier.hocbf(traj).detach())
    h_s.append(state_barrier.barrier(traj).detach())
    h_star.append(safety_filter.barrier.get_h_star(traj).detach())

    feas_factor.append(info['feas_fact'])
    constraints_val.append(info['constraint_val'])
    u_star.append(info['u_star'].detach())
    ub_select.append(info['ub_select'].detach())
    beta.append(info['beta'])
############
#  Plots   #
############

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

x = np.linspace(-pi - 0.5, pi +0.5, 500)
y = np.linspace(-pi - 0.5, pi +0.5, 500)
X, Y = np.meshgrid(x, y, )
points = np.column_stack((X.flatten(), Y.flatten()))
points = torch.tensor(points, dtype=torch.float64)
Z = map_.barrier.min_barrier(points)
Z = Z.reshape(X.shape)

Z_backup = backup_barriers[0].min_barrier(points).reshape(X.shape)
Z_backup = [backup_barriers[i].min_barrier(points).reshape(X.shape) for i in range(len(backup_barriers))]

Z_fwd = fwd_barrier.min_barrier(points)
Z_fwd = Z_fwd.reshape(X.shape)

fig, ax = plt.subplots(figsize=(6, 6))

map_contour_plot = ax.contour(X, Y, Z, levels=[0], colors='red')

backup_contour_plots = [ax.contour(X, Y, Z_backup[i], levels=[0], colors='blue')
                 for i in range(len(Z_backup))]

fwd_contour_plot = ax.contour(X, Y, Z_fwd, levels=[0], colors='green')



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


ax.legend()
plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/Trajectories_QP_Backup_Safe_Control_{current_time}.png', dpi=600)
plt.show()

# Calculate time array based on the number of data points and timestep
num_points = trajs[0].shape[0]  # Assuming trajs has the same length for all elements
time = np.linspace(0.0, (num_points - 1) * timestep, num_points)

# Create subplot for trajs and action variables
fig, axs = plt.subplots(3, 1, figsize=(8, 8))

# Plot trajs variables
axs[0].plot(time, trajs[0][:, 0], label=r'$\theta$', color='black')
axs[0].set_ylabel(r'$\theta$', fontsize=14)
# axs[0].legend(fontsize=14, loc='best', frameon=False)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

axs[1].plot(time, trajs[0][:, 1], label=r'$\dot{\theta}$', color='black')
axs[1].set_ylabel(r'$\dot{\theta}$', fontsize=14)

axs[2].plot(time, actions[0][:, 0], label='u', color='black')
axs[2].plot(time, ub_select[0][:, 0], label=r'$u_b$', color='green')
axs[2].plot(time, u_star[0][:, 0], label=r'$u_*$', color='blue')
axs[2].legend(fontsize=14, loc='best', frameon=False)

axs[2].set_ylabel(r'$u$', fontsize=14)
axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=14)
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

axs[2].plot(time, beta[0][:, 0], label=r'$\beta$', color='blue')

axs[2].set_ylabel(r'$\sigma$', fontsize=14)
axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=14)
ax.legend()
plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/Barriers_QP_Backup_Safe_Control_{current_time}.png', dpi=600)
plt.show()



