from unicycle_desired_control import desired_control
from map_config import map_config
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.dynamics import LowPassFilterDynamics
from cbftorch.barriers.barrier import Barrier
from cbftorch.safe_controls.closed_form_safe_control import MinIntervInputConstCFSafeControl
from box import Box as AD
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import pi
from cbftorch.utils.make_map import Map
import datetime
from cbftorch.utils.utils import *

mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

torch.set_default_dtype(torch.float64)

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
map_ = Map(barriers_info=map_config, dynamics=state_dynamics, cfg=cfg)
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
              sigma=[1.0, 1.0],
              buffer=0.0)
).assign_state_action_dynamics(state_dynamics=state_dynamics,
                               action_dynamics=ac_dyn,
                               action_output_function=ac_out_func).assign_state_barrier(
    barrier=[*pos_barrier, *vel_barrier]).assign_action_barrier(action_barrier=ac_barriers, rel_deg=1)


goal_pos = torch.tensor([
    [3.5, 4.5],
    [-8.0, 9.0],
    [-0.9, 7.0],
    [6.0, 9.5],
])

# Initial Conditions
timestep = 0.01
sim_time = 20.0

safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos, k1=0.2, k2=1.0, k3=2.0)(x)))

safety_filter.make()
x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2, 0.0, 0.0]).repeat(goal_pos.shape[0], 1)

# Simulate trajectories
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')

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
        partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1))(x))
    safety_filter.assign_desired_control(
        desired_control=des_ctrl).make()
    actions.append(safety_filter.safe_optimal_control(traj))

    for j in range(10):
        safety_filter.safe_optimal_control(traj[j, :].unsqueeze(0))

    aux_des_ctrls.append(safety_filter.aux_desired_action(traj))
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
plt.savefig(f'figs/05_Trajectories_Input_Constrained_CF_Safe_Control_{current_time}.png', dpi=300)
plt.show()

# Calculate time array based on the number of data points and timestep
num_points = trajs[0].shape[0]  # Assuming trajs has the same length for all elements
time = np.linspace(0.0, (num_points - 1) * timestep, num_points)
for id, traj in enumerate(trajs):
    # Create subplot for trajs and action variables
    fig, axs = plt.subplots(7, 1, figsize=(8, 10))

    # Plot trajs variables
    axs[0].plot(time, traj[:, 0], label=r'$q_{\rm x}$', color='red')
    axs[0].plot(time, traj[:, 1], label=r'$q_{\rm y}$', color='blue')
    axs[0].plot(time, torch.ones(time.shape) * goal_pos[0, 0], label=r'$q_{\rm d, x}$', color='red', linestyle=':')
    axs[0].plot(time, torch.ones(time.shape) * goal_pos[0, 1], label=r'$q_{\rm d, y}$', color='blue', linestyle=':')
    axs[0].legend(loc='lower center', ncol=4, frameon=False, fontsize=14)

    axs[0].set_ylabel(r'$q_{\rm x}, q_{\rm y}$', fontsize=16)
    axs[0].legend(fontsize=14, loc='lower center', ncol=4, frameon=False)
    axs[0].tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    axs[1].plot(time, traj[:, 2], label='v', color='black')
    axs[1].set_ylabel(r'$v$', fontsize=16)

    axs[2].plot(time, traj[:, 3], label='theta', color='black')
    axs[2].set_ylabel(r'$\theta$', fontsize=16)

    # Plot actions
    axs[3].plot(time, traj[:, 4], label=r'$u_1$', color='black')
    axs[3].plot(time, des_ctrls[id][:, 0], color='red', linestyle='--', label=r'$u_{{\rm d}_1}$')
    axs[3].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)
    axs[3].set_ylabel(r'$u_1$', fontsize=16)

    axs[4].plot(time, traj[:, 5], label=r'$u_2$', color='black')
    axs[4].plot(time, des_ctrls[id][:, 1],  color='red', linestyle='--', label=r'$u_{{\rm d}_2}$')
    axs[4].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)
    axs[4].set_ylabel(r'$u_2$', fontsize=16)

    # Plot actions
    axs[5].plot(time, actions[id][:, 0], label=r'$\hat u_1$', color='black')
    axs[5].plot(time, aux_des_ctrls[id][:, 0], color='red', linestyle='--', label=r'$\hat u_{{\rm d}_1}$')
    axs[5].set_ylabel(r'$\hat u_1$', fontsize=16)
    axs[5].legend(loc='upper right', ncol=2, frameon=False, fontsize=14)

    axs[6].plot(time, actions[id][:, 1], label='$\hat u_2$', color='black')
    axs[6].plot(time, aux_des_ctrls[id][:, 1], color='red', linestyle='--', label=r'$\hat u_{{\rm d}_2}$')
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

    plt.savefig(f'figs/05_States_Input_Constrained_CF_Safe_Control_{current_time}_{id}.png')

    # Show the plots
    plt.show()


    fig, axs = plt.subplots(3, 1, figsize=(8, 4.5))

    # Plot barrier values
    axs[0].plot(time, h_vals[id], color='black')
    axs[0].set_ylabel(r'$\hat h$', fontsize=16)

    # Plot barrier values
    axs[1].plot(time, min_barriers[id], color='black')
    axs[1].set_ylabel(r'$\min \hat b_{\hat \jmath, i}$', fontsize=16)

    axs[2].plot(time, min_constraint[id], color='black')
    axs[2].set_ylabel(r'$\min \hat h_{\hat \jmath}$', fontsize=16)

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



    plt.savefig(f'figs/05_Barriers_Input_Constrained_CF_Safe_Control_{current_time}_{id}.png', dpi=300)

    # Show the plots
    plt.show()