from box import Box as AD
import matplotlib as mpl
from math import pi
from cbftorch.dynamics import DIDynamics
from cbftorch.utils.make_map import Map
from cbftorch.safe_controls.closed_form_safe_control import MinIntervCFSafeControl
from map_config import map_config
from double_integrator_desired_control import desired_control
from time import time
import datetime
import matplotlib.pyplot as plt
from cbftorch.utils.utils import *


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# Control gains
control_gains = dict(k1=2.0, k2=2.0)

# Barrier configs
cfg = AD(softmax_rho=20,
         softmin_rho=20,
         pos_barrier_rel_deg=2,
         vel_barrier_rel_deg=1,
         obstacle_alpha=[10.0],
         boundary_alpha=[10.0],
         velocity_alpha=[],
         )

# Instantiate dynamics
dynamics = DIDynamics(state_dim=4, action_dim=2)

# Make barrier from map_
map_ = Map(dynamics=dynamics, cfg=cfg, barriers_info=map_config)

# Make safety filter and assign dynamics and barrier
safety_filter = MinIntervCFSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 0.5 * x,
    params=AD(slack_gain=1e24,
              use_softplus=False,
              softplus_gain=2.0)
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=map_.barrier)

# Goal positions
goal_pos = torch.tensor([
    [3.0, 4.5],
    [-7.0, 9.0],
    # [-7.0, 0.0],
    # [7.0, 1.5],
    [-1.0, 7.0],
    [5.0, 9.5],
])

# Initial Conditions
x0 = torch.tensor([-1.0, -8.5, 0.0, 0.0]).repeat(goal_pos.shape[0], 1)
timestep = 0.01
sim_time = 12.0

# assign desired control based on the goal positions
safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos, **control_gains)(x))
)

# Simulate trajectories
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')

# Rearrange trajs
trajs = [torch.vstack(t.split(dynamics.state_dim)) for t in torch.hstack([tt for tt in trajs])]

# Get actions values along the trajs
actions = []
des_ctrls = []
h_vals = []
min_barriers = []
min_constraint = []
for i, traj in enumerate(trajs):
    des_ctrl = lambda x: vectorize_tensors(
            partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1), **control_gains)(x))
    safety_filter.assign_desired_control(
        desired_control=des_ctrl
    )
    actions.append(safety_filter.safe_optimal_control(traj))
    des_ctrls.append(des_ctrl(traj))
    h_vals.append(map_.barrier.hocbf(traj))
    min_barriers.append(map_.barrier.get_min_barrier_at(traj))
    min_constraint.append(map_.barrier.min_barrier(traj))

############
#  Plots   #
############

current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

x = np.linspace(-10.5, 10.5, 500)
y = np.linspace(-10.5, 10.5, 500)
X, Y = np.meshgrid(x, y, )
points = np.column_stack((X.flatten(), Y.flatten()))
points = np.column_stack((points, np.zeros(points.shape)))
points = torch.tensor(points, dtype=torch.float32)
# print(barriers[0].barrier(points))
Z = map_.barrier.min_barrier(points)
Z = Z.reshape(X.shape)

fig, ax = plt.subplots(figsize=(6, 6))

contour_plot = ax.contour(X, Y, Z, levels=[0], colors='red')
# Adding a custom legend handle for the contour
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='red', lw=1.5)]

ax.set_xlabel(r'$q_{\rm x}$', fontsize=16)
ax.set_ylabel(r'$q_{\rm y}$', fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_aspect('equal', adjustable='box')  # Maintain equal aspect ratio
ax.tick_params(axis='x', labelsize=16)  # Font size for x-axis ticks
ax.tick_params(axis='y', labelsize=16)  # Font size for y-axis ticks
ax.set_xticks([-10, -5, 0, 5, 10])
ax.set_yticks([-10, -5, 0, 5, 10])

ax.plot(trajs[0][0, 0], trajs[0][0, 1], 'x', color='blue', markersize=8, label=r'$x_0$')

for i in range(4):
    ax.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='limegreen', label='Goal' if i == 0 else None)
    ax.plot(trajs[i][-1, 0], trajs[i][-1, 1], '+', color='blue', markersize=8, label=r'$x_f$' if i == 0 else None)
    ax.plot(trajs[i][:, 0], trajs[i][:, 1], label='Trajectories' if i == 0 else None, color='black')

# Creating the legend
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\mathcal{S}_{\rm s}$')

ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)

custom_order = [r'$\mathcal{S}_{\rm s}$', 'Goal', 'Trajectories', r'$x_0$', r'$x_f$']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in custom_order]
ordered_labels = custom_order

plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/Trajectories_CF_Safe_Control_{current_time}.png', dpi=600)
plt.show()

# Calculate time array based on the number of data points and timestep
num_points = trajs[0].shape[0]  # Assuming trajs has the same length for all elements
time = np.linspace(0.0, (num_points - 1) * timestep, num_points)

# Create subplot for trajs and action variables
fig, axs = plt.subplots(5, 1, figsize=(8, 8))

# Plot trajs variables
axs[0].plot(time, trajs[0][:, 0], label=r'$q_{\rm x}$', color='red')
axs[0].plot(time, trajs[0][:, 1], label=r'$q_{\rm y}$', color='blue')
axs[0].plot(time, torch.ones(time.shape) * goal_pos[0, 0], label=r'$q_{\rm d, x}$', color='red', linestyle=':')
axs[0].plot(time, torch.ones(time.shape) * goal_pos[0, 1], label=r'$q_{\rm d, y}$', color='blue', linestyle=':')
axs[0].legend(loc='lower center', ncol=4, frameon=False, fontsize=14)

axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$', fontsize=16)
axs[0].legend(fontsize=14, loc='lower center', ncol=4, frameon=False)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

axs[1].plot(time, trajs[0][:, 2], label='v_x', color='black')
axs[1].set_ylabel(r'$v_x$', fontsize=16)

axs[2].plot(time, trajs[0][:, 3], label='v_y', color='black')
axs[2].set_ylabel(r'$v_y$', fontsize=16)

# Plot actions
axs[3].plot(time, actions[0][:, 0], label=r'$u_1$', color='black')
axs[3].plot(time, des_ctrls[0][:, 0], color='red', linestyle='--', label=r'$u_{{\rm d}_1}$')
axs[3].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)
axs[3].set_ylabel(r'$u_1$', fontsize=16)

axs[4].plot(time, actions[0][:, 1], label=r'$u_2$', color='black')
axs[4].plot(time, des_ctrls[0][:, 1],  color='red', linestyle='--', label=r'$u_{{\rm d}_2}$')
axs[4].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)
axs[4].set_ylabel(r'$u_2$', fontsize=16)

axs[4].set_xlabel(r'$t~(\rm {s})$', fontsize=16)



for i in range(4):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Multiply x-axis labels by timestep value (0.001)
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time[0], time[-1])

# Adjust layout and save the combined plot
plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()

plt.savefig(f'figs/States_CF_Safe_Control_{current_time}.png', dpi=600)

# Show the plots
plt.show()


fig, axs = plt.subplots(3, 1, figsize=(8, 4.5))

# Plot barrier values
axs[0].plot(time, h_vals[0], color='black')
axs[0].set_ylabel(r'$h$', fontsize=16)

# Plot barrier values
axs[1].plot(time, min_barriers[0], color='black')
axs[1].set_ylabel(r'$\min b_{j, i}$', fontsize=16)

axs[2].plot(time, min_constraint[0], color='black')
axs[2].set_ylabel(r'$\min h_j$', fontsize=16)

axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=16)


for i in range(2):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for i, ax in enumerate(axs):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time[0], time[-1])
    if i != 0:
        ax.set_yticks([0, 0.5])

plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()

plt.savefig(f'figs/Barriers_CF_Safe_Control_{current_time}.png', dpi=600)

# Show the plots
plt.show()