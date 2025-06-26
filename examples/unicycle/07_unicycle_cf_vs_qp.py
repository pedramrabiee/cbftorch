from box import Box as AD
import matplotlib as mpl
from math import pi
from cbftorch.dynamics import UnicycleDynamics
from cbftorch.utils.make_map import Map
from cbftorch.safe_controls.closed_form_safe_control import MinIntervCFSafeControl
from map_config import map_config
from unicycle_desired_control import desired_control
from cbftorch.barriers.multi_barrier import MultiBarriers
from cbftorch.safe_controls.qp_safe_control import MinIntervQPSafeControl

from time import time
import datetime
import matplotlib.pyplot as plt
from cbftorch.utils.utils import *


mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

# Control gains

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
dynamics = UnicycleDynamics()

# Make barrier from map_
map_ = Map(barriers_info=map_config, dynamics=dynamics, cfg=cfg)

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
    # [7.0, 1.5],
    [-1.0, 7.0],
    [5.0, 9.5],
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
trajs = [torch.vstack(t.split(dynamics.state_dim)) for t in torch.hstack([tt for tt in trajs])]

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
    ax.plot(trajs[i][:, 0], trajs[i][:, 1], label='Soft-minimum R-CBF' if i == 0 else None, color='black')


pos_barrier, vel_barrier = map_.get_barriers()

barrier = MultiBarriers()
barrier.add_barriers([*pos_barrier, *vel_barrier], infer_dynamics=True)


safety_filter = MinIntervQPSafeControl(
    action_dim=dynamics.action_dim,
    alpha=lambda x: 0.5 * x,
    params=AD(slack_gain=1e24,
              slacked=False,
              use_softplus=False,
              softplus_gain=2.0)
).assign_dynamics(dynamics=dynamics).assign_state_barrier(barrier=barrier)

# Goal positions
goal_pos = torch.tensor([
    [3.0, 4.5],
    [-8.0, 9.0],
    # [-7.0, 0.0],
    # [7.0, 1.5],
    [-1.0, 7.0],
    [6.0, 9.5],
])

# Initial Conditions
x0 = torch.tensor([-1.0, -8.5, 0.0, pi / 2]).repeat(goal_pos.shape[0], 1)
timestep = 0.01
sim_time = 25.0

# assign desired control based on the goal positions
safety_filter.assign_desired_control(
    desired_control=lambda x: vectorize_tensors(partial(desired_control, goal_pos=goal_pos)(x))
)

# Simulate trajectories
start_time = time()
trajs = safety_filter.get_safe_optimal_trajs(x0=x0, sim_time=sim_time, timestep=timestep, method='euler')
print(time() - start_time)

# Rearrange trajs
trajs = [torch.vstack(t.split(dynamics.state_dim)) for t in torch.hstack([tt for tt in trajs])]


for i, traj in enumerate(trajs):
    des_ctrl = lambda x: vectorize_tensors(
            partial(desired_control, goal_pos=goal_pos[i].repeat(x.shape[0], 1))(x))
    safety_filter.assign_desired_control(
        desired_control=des_ctrl
    )
    action = safety_filter.safe_optimal_control(traj)
    hocbf, Lf_hocbf, Lg_hocbf = safety_filter.barrier.get_hocbf_and_lie_derivs(traj.to(torch.float64))
    constraint = Lf_hocbf.squeeze(-1) + torch.einsum('ijk,ik->ij', Lg_hocbf, action) + 0.5 * hocbf.squeeze(-1)
    cutoff = (torch.min(constraint, dim=-1).values < -1e-4).nonzero(as_tuple=True)[0]

    if cutoff.numel() > 0:
        cutoff = cutoff[0].item()
        trajs[i] = traj[:cutoff]  # Cut the trajectory short



for i in range(4):
    ax.plot(trajs[i][-1, 0], trajs[i][-1, 1], '+', color='blue', markersize=8)
    ax.plot(trajs[i][:, 0], trajs[i][:, 1], label='Multiple HOCBF' if i == 0 else None, color='blueviolet')

# Creating the legend
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, custom_lines[0])
labels.insert(0, r'$\mathcal{S}_{\rm s}$')

custom_order = [r'$\mathcal{S}_{\rm s}$', r'$x_0$', 'Soft-minimum R-CBF', 'Multiple HOCBF', 'Goal']
handle_dict = dict(zip(labels, handles))
ordered_handles = [handle_dict[label] for label in custom_order]
ordered_labels = custom_order

ax.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False, fontsize=12)



plt.tight_layout()

# Save the contour plot
plt.savefig(f'figs/07_Trajectories_CF_QP_Comparison{current_time}.png', dpi=200)
plt.show()