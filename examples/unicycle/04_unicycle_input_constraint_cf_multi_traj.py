from unicycle_desired_control import desired_control
from map_config import map_config
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.dynamics import LowPassFilterDynamics
from cbftorch.barriers.barrier import Barrier
from cbftorch.safe_controls.closed_form_safe_control import MinIntervInputConstCFSafeControl
from box import Box as AD
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import pi
from cbftorch.utils.make_map import Map
from time import time
import datetime
from cbftorch.utils.utils import *

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

torch.set_default_dtype(torch.float64)

# Control gains
control_gains = dict(k1=0.2, k2=1.0, k3=2.0)

# Barrier configs
cfg = AD(softmax_rho=20,
         softmin_rho=10,
         pos_barrier_rel_deg=3,
         vel_barrier_rel_deg=2,
         obstacle_alpha=[1.0, 2.5],
         boundary_alpha=[6.0, 1.0],
         velocity_alpha=[10.0],
         )

# Instantiate dynamics
state_dynamics = UnicycleDynamics()

# make position and velocity barrer
map_ = Map(dynamics=state_dynamics, cfg=cfg, barriers_info=map_config)
pos_barrier, vel_barrier = map_.get_barriers()

# make action dynamics
ac_dyn = LowPassFilterDynamics(state_dim=2, action_dim=2, params=dict(gains=[1.0, 0.6]))

# make action output function
ac_out_func = lambda x: x

# make action barriers
ac_barriers_funcs = [*make_box_barrier_functionals(bounds=[-4.0, 4.0], idx=4),
                     *make_box_barrier_functionals(bounds=[-1.0, 1.0], idx=5)]
ac_barriers = [Barrier().assign(barrier_func=ac_barrier, rel_deg=1).assign_dynamics(ac_dyn) for ac_barrier in
               ac_barriers_funcs]

# Make input constrained safety filter, assign state and action dynamics, assign state and action barriers
safety_filter = MinIntervInputConstCFSafeControl(
    action_dim=state_dynamics.action_dim,
    alpha=lambda x: 1.0 * x,
    params=AD(slack_gain=100,
              use_softplus=False,
              softplus_gain=2.0,
              softmin_rho=10,
              softmax_rho=10,
              sigma=[1.0, 1.0])
).assign_state_action_dynamics(state_dynamics=state_dynamics,
                               action_dynamics=ac_dyn,
                               action_output_function=ac_out_func).assign_state_barrier(
    barrier=[*pos_barrier, *vel_barrier]).assign_action_barrier(action_barrier=ac_barriers, rel_deg=1)


# Goal positions
x = np.linspace(-10, 10, 60)
y = np.linspace(-10, 10, 60)
X, Y = np.meshgrid(x, y, )
points = np.column_stack((X.flatten(), Y.flatten()))
points = np.column_stack((points, np.zeros(points.shape)))
points = torch.tensor(points, dtype=torch.float32)
Z = map_.barrier.min_barrier(points)
goals = points[(Z >= 0).squeeze()]
print("num trajs: ", len(goals))
goal_pos = goals[:, :2]

# Initial Conditions
timestep = 0.001
sim_time = 20.0

safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos, **control_gains)(x)))

safety_filter.make()
x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2, 0.0, 0.0]).repeat(goal_pos.shape[0], 1)

# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')
print(time() - start_time)

# Rearrange trajs
trajs = [torch.vstack(t.split(6)) for t in torch.hstack([tt for tt in trajs])]


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


for i in range(len(trajs)):
    ax.plot(trajs[i][:, 0], trajs[i][:, 1], color='deepskyblue', alpha=0.05)

# Creating a custom line for 'Trajectories' with alpha 0.8 for the legend
custom_lines.append(Line2D([0], [0], color='deepskyblue', alpha=0.6, label='Trajectories'))

# Goal positions
goal_pos = torch.tensor([
    [3.5, 4.5],
    [-8.0, 9.0],
    [-0.9, 7.0],
    [6.0, 9.5],
])

timestep = 0.01
sim_time = 20.0

safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos)(x)))

safety_filter.make()
x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2, 0.0, 0.0]).repeat(goal_pos.shape[0], 1)

# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')
print(time() - start_time)

# Rearrange trajs
trajs = [torch.vstack(t.split(6)) for t in torch.hstack([tt for tt in trajs])]

# Get actions values along the trajs
actions = []
des_ctrls = []
h_vals = []
min_barriers = []
min_constraint = []
aux_des_ctrls = []
for i, traj in enumerate(trajs):
    des_ctrl = lambda x: vectorize_tensors(
        partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1), **control_gains)(x))
    safety_filter.assign_desired_control(
        desired_control=des_ctrl).make()
    actions.append(safety_filter.safe_optimal_control(traj))
    aux_des_ctrls.append(safety_filter.aux_desired_action(traj))
    des_ctrls.append(des_ctrl(traj))
    h_vals.append(safety_filter.barrier.hocbf(traj).detach())
    min_barriers.append(safety_filter.barrier.get_min_barrier_at(traj))
    min_constraint.append(safety_filter.barrier.min_barrier(traj))

ax.plot(trajs[0][0, 0], trajs[0][0, 1], 'x', color='blue', markersize=8, label=r'$x_0$')

for i in range(4):
    ax.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='limegreen', label='Goal' if i == 0 else None)
    ax.plot(trajs[i][-1, 0], trajs[i][-1, 1], '+', color='blue', markersize=8, label=r'$x_f$' if i == 0 else None)
    ax.plot(trajs[i][:, 0], trajs[i][:, 1], label='Selected Trajectories' if i == 0 else None, color='black')


# Creating the legend
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\mathcal{S}_{\rm s}$')
handles.insert(3, custom_lines[1])  # Add the custom line for Trajectories with higher alpha
labels.insert(3, 'Trajectories')  # Add the corresponding label

# ax.legend(handles, labels)

custom_order = [r'$\mathcal{S}_{\rm s}$', r'$x_0$', 'Goal', 'Trajectories', 'Selected Trajectories']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in custom_order]
ax.legend(ordered_handles, custom_order, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)


plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/04_Trajectories_Input_Constrained_CF_Safe_Control_Multi_Trajs_{current_time}_200dpi.png', dpi=200)

plt.show()


# Calculate time array based on the number of data points and timestep
num_points = trajs[0].shape[0]  # Assuming trajs has the same length for all elements
time = np.linspace(0.0, (num_points - 1) * timestep, num_points)

# Create subplot for trajs and action variables
fig, axs = plt.subplots(7, 1, figsize=(8, 10))

# Plot trajs variables
axs[0].plot(time, trajs[0][:, 0], label=r'$q_{\rm x}$', color='red')
axs[0].plot(time, trajs[0][:, 1], label=r'$q_{\rm y}$', color='blue')
axs[0].plot(time, torch.ones(time.shape) * goal_pos[0, 0], label=r'$q_{\rm d, x}$', color='red', linestyle=':')
axs[0].plot(time, torch.ones(time.shape) * goal_pos[0, 1], label=r'$q_{\rm d, y}$', color='blue', linestyle=':')
axs[0].legend(loc='lower center', ncol=4, frameon=False, fontsize=14)

axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$', fontsize=16)
axs[0].legend(fontsize=14, loc='lower center', ncol=4, frameon=False)
axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

axs[1].plot(time, trajs[0][:, 2], label='v', color='black')
axs[1].set_ylabel(r'$v$', fontsize=16)

axs[2].plot(time, trajs[0][:, 3], label='theta', color='black')
axs[2].set_ylabel(r'$\theta$', fontsize=16)

# Plot actions
axs[3].plot(time, trajs[0][:, 4], label=r'$u_1$', color='black')
axs[3].plot(time, des_ctrls[0][:, 0], color='red', linestyle='--', label=r'$u_{{\rm d}_1}$')
axs[3].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)
axs[3].set_ylabel(r'$u_1$', fontsize=16)

axs[4].plot(time, trajs[0][:, 5], label=r'$u_2$', color='black')
axs[4].plot(time, des_ctrls[0][:, 1],  color='red', linestyle='--', label=r'$u_{{\rm d}_2}$')
axs[4].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)
axs[4].set_ylabel(r'$u_2$', fontsize=16)

# Plot actions
axs[5].plot(time, actions[0][:, 0], label=r'$\hat u_1$', color='black')
axs[5].plot(time, aux_des_ctrls[0][:, 0], color='red', linestyle='--', label=r'$\hat u_{{\rm d}_1}$')
axs[5].set_ylabel(r'$\hat u_1$', fontsize=16)
axs[5].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)

axs[6].plot(time, actions[0][:, 1], label='$\hat u_2$', color='black')
axs[6].plot(time, aux_des_ctrls[0][:, 1], color='red', linestyle='--', label=r'$\hat u_{{\rm d}_2}$')
axs[6].set_ylabel(r'$\hat u_2$', fontsize=16)
axs[6].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)
axs[6].set_xlabel(r'$t~(\rm {s})$', fontsize=16)

for i in range(6):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

# Multiply x-axis labels by timestep value (0.001)
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(time[0], time[-1])


# Adjust layout and save the combined plot
plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()

plt.savefig(f'figs/04_States_Input_Constrained_CF_Safe_Control_Multi_Trajs_{current_time}.png')

# Show the plots
plt.show()


fig, axs = plt.subplots(3, 1, figsize=(8, 4.5))

# Plot barrier values
axs[0].plot(time, h_vals[0], color='black')
axs[0].set_ylabel(r'$\hat h$', fontsize=16)

# Plot barrier values
axs[1].plot(time, min_barriers[0], color='black')
axs[1].set_ylabel(r'$\min \hat b_{j, i}$', fontsize=16)

axs[2].plot(time, min_constraint[0], color='black')
axs[2].set_ylabel(r'$\min \hat h_j$', fontsize=16)

axs[2].set_xlabel(r'$t~(\rm {s})$', fontsize=16)


for i in range(2):
    axs[i].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=16)
    ax.set_xlim(time[0], time[-1])
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_yticks([0, 0.5])


plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()


plt.savefig(f'figs/04_Barriers_Input_Constrained_CF_Safe_Control_Multi_Trajs_{current_time}.png', dpi=200)

# Show the plots
plt.show()