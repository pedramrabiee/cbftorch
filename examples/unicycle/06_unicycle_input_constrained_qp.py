from box import Box as AD
import matplotlib as mpl
from math import pi
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.make_map import Map
from map_config import map_config
from unicycle_desired_control import desired_control
from cbftorch.barriers.multi_barrier import MultiBarriers
from time import time
import datetime
import matplotlib.pyplot as plt
from cbftorch.safe_controls.qp_safe_control import MinIntervInputConstQPSafeControl
from cbftorch.utils.utils import *

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# Control gains
control_gains = dict(k1=0.2, k2=1.0, k3=2.0)

# Barrier configs
cfg = AD(softmax_rho=20,
         softmin_rho=10,
         pos_barrier_rel_deg=2,
         vel_barrier_rel_deg=1,
         obstacle_alpha=[2.5],
         boundary_alpha=[1.0],
         velocity_alpha=[],
         )

# Instantiate dynamics
dynamics = UnicycleDynamics()

# Make barrier from map_
map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg)

pos_barrier, vel_barrier = map_.get_barriers()

barrier = MultiBarriers()
barrier.add_barriers([*pos_barrier, *vel_barrier], infer_dynamics=True)


safety_filter = MinIntervInputConstQPSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 1.0 * x,
    params=AD(slack_gain=200,
              slacked=True,
              use_softplus=False,
              softplus_gain=2.0)
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=barrier
                                                          ).assign_control_bounds(low=[-4.0, -1.0], high=[4.0, 1.0])

# Goal positions
goal_pos = torch.tensor([
    [3.0, 4.5],
    [-7.0, 0.0],
    [7.0, 1.5],
    [-1.0, 7.0]
])

# Initial Conditions
x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2]).repeat(goal_pos.shape[0], 1)
timestep = 0.01
sim_time = 20.0

# assign desired control based on the goal positions
safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos)(x))
)

# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')
print(time() - start_time)

# Rearrange trajs
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
        desired_control=des_ctrl)
    start_time = time()
    actions.append(safety_filter.safe_optimal_control(traj))
    print("totol_time", time() - start_time)

    for j in range(10):
        start_time = time()
        safety_filter.safe_optimal_control(traj[j, :].unsqueeze(0))
        print(time() - start_time)

    des_ctrls.append(des_ctrl(traj))
    h_vals.append(safety_filter.barrier.hocbf(traj).detach())
    min_barriers.append(safety_filter.barrier.get_min_barrier_at(traj))
    min_constraint.append(safety_filter.barrier.min_barrier(traj))

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
for i in range(len(trajs)):
    ax.plot(goal_pos[i][0], goal_pos[i][1], '*', markersize=10, color='limegreen', label='Goal' if i == 0 else None)
    ax.plot(trajs[i][-1, 0], trajs[i][-1, 1], '+', color='blue', markersize=8, label=r'$x_f$' if i == 0 else None)
    ax.plot(trajs[i][:, 0], trajs[i][:, 1], label='Trajectories' if i == 0 else None, color='black')


# Creating the legend
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\mathcal{S}_{\rm s}$')
# ax.legend(handles, labels)
ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)

custom_order = [r'$\mathcal{S}_{\rm s}$', 'Goal', 'Trajectories', r'$x_0$', r'$x_f$']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in custom_order]
ordered_labels = custom_order

plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/06_Trajectories_Input_Constrained_QP_Safe_Control_{current_time}.png', dpi=600)
plt.show()

# Calculate time array based on the number of data points and timestep
num_points = trajs[0].shape[0]  # Assuming trajs has the same length for all elements
time = np.linspace(0.0, (num_points - 1) * timestep, num_points)

# Create subplot for trajs and action variables
fig, axs = plt.subplots(5, 1, figsize=(8, 10))

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
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlim(time[0], time[-1])


# Adjust layout and save the combined plot
plt.subplots_adjust(wspace=0, hspace=0.2)
plt.tight_layout()

plt.savefig(f'figs/06_States_Input_Constrained_QP_Safe_Control_{current_time}.png')

# Show the plots
plt.show()


fig, axs = plt.subplots(3, 1, figsize=(8, 4.5))

# Plot barrier values
axs[0].plot(time, h_vals[0].min(dim=1).values.squeeze(), color='black')
axs[0].set_ylabel(r'$\min b_{d_{{\rm c} -1}, i}$', fontsize=16)

# Plot barrier values
axs[1].plot(time, min_barriers[0], color='black')
axs[1].set_ylabel(r'$\min b_{j, i}$', fontsize=16)

axs[2].plot(time, min_constraint[0], color='black')
axs[2].set_ylabel(r'$\min h_j$', fontsize=16)

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

plt.savefig(f'figs/06_Barriers_Input_Constrained_QP_Safe_Control_{current_time}.png', dpi=600)

# Show the plots
plt.show()