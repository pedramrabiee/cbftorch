# CBFTorch: Control Barrier Functions in PyTorch

[![PyPI version](https://badge.fury.io/py/cbftorch.svg)](https://badge.fury.io/py/cbftorch)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  <img src="https://raw.githubusercontent.com/pedramrabiee/cbftorch/main/examples/unicycle/figs/01_Trajectories_CF_Safe_Control_Multi_Trajs_300dpi.png" alt="Trajectory Plots" width="600">
</p>

CBFTorch is a PyTorch-based library for implementing Control Barrier Functions (CBFs) and higher-order CBFs for safe control of dynamical systems. It provides a flexible and efficient framework for constructing safety filters using barrier functions with closed-form solutions.

## Installation

### Quick Install

```bash
pip install cbftorch
```

### Prerequisites

CBFTorch requires PyTorch. Since PyTorch installation depends on your system's CUDA version, you may need to install it separately:

**For CUDA 12.4 (latest):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install cbftorch
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install cbftorch
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install cbftorch
```

### Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/pedramrabiee/cbftorch.git
cd cbftorch
pip install -e .
```

### Conda Environment

For a complete environment setup, use the provided `environment.yml`:

```bash
git clone https://github.com/pedramrabiee/cbftorch.git
cd cbftorch
conda env create -f environment.yml
conda activate cbftorch
```

### System Dependencies

For full functionality (including LaTeX rendering in plots), install:

**Ubuntu/Debian:**
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra dvipng
```

**macOS:**
```bash
brew install --cask mactex
```

## Features

- **Control Barrier Functions**: Implementation of CBFs with automatic differentiation
- **Higher-Order CBFs**: Support for systems with high relative degree
- **Barrier Composition**: Smooth and non-smooth composition of multiple barriers
- **Closed-Form Solutions**: Efficient closed-form safe control synthesis
- **PyTorch Integration**: GPU acceleration and automatic differentiation
- **Input Constraints**: Handle control input limits through augmented dynamics

### Barrier Class

The `Barrier` class represents a basic barrier function with methods for assigning barrier functions, dynamics, and computing barrier values.

- `assign(barrier_func, rel_deg=1, alphas=None)`: Assigns a barrier function to the Barrier object.
- `assign_dynamics(dynamics)`: Assigns dynamics to the Barrier object and generates higher-order barrier functions.
- `raise_rel_deg(x, raise_rel_deg_by=1, alphas=None)`: Raises the relative degree of the barrier function.
- `barrier(x)`: Computes the barrier function value at a given state.
- `hocbf(x)`: Computes the highest-order barrier function value at a given state.
- `Lf_hocbf(x)`: Computes the Lie derivative of the highest-order barrier function with respect to the system dynamics.
- `Lg_hocbf(x)`: Computes the Lie derivative of the highest-order barrier function with respect to the system dynamics.
- `compute_barriers_at(x)`: Computes barrier values at a given state.
- `get_min_barrier_at(x)`: Gets the minimum barrier value at a given state.

### CompositionBarrier Class

The `CompositionBarrier` class extends the `Barrier` class and represents a barrier formed by composing multiple barriers with a specific rule.

- `assign_barriers_and_rule(barriers, rule)`: Assigns multiple barriers and a composition rule to the CompositionBarrier object.
- `min_barrier(x)`: Calculates the minimum value among all the barrier values computed at a point.
- `compose(c_key)`: Selects the appropriate composition rule based on the provided key.

### SoftCompositionBarrier Class

The `SoftCompositionBarrier` class extends the `CompositionBarrier` class and represents a soft composition of multiple barriers with specific soft composition rules.

### NonSmoothCompositionBarrier Class

The `NonSmoothCompositionBarrier` class extends the `CompositionBarrier` class and represents a non-smooth composition of multiple barriers with specific non-smooth composition rules.


### CFSafeControl Class

The `CFSafeControl` class implements a safety filter for closed-form optimal control. It allows for safe control of dynamical systems by incorporating barrier functions and Lie derivatives.

- `assign_state_barrier(barrier)`: Assigns a state barrier to the safety filter.
- `assign_dynamics(dynamics)`: Assigns system dynamics to the safety filter.
- `assign_cost(Q, c)`: Assigns the cost function parameters for optimal control.
- `safe_optimal_control(x)`: Computes safe optimal control actions based on the current state.
- `get_safe_optimal_trajs(x0, timestep, sim_time, method)`: Simulates the system trajectory under safe optimal control.
- `eval_barrier(x)`: Evaluates the barrier function at a given state.

### MinIntervCFSafeControl Class

The `MinIntervCFSafeControl` class extends the `CFSafeControl` and automatically assigns the cost function for minimizing intervention during control. It provides methods for assigning desired control actions.

- `assign_desired_control(desired_control)`: Assigns the desired control action for the system.

### InputConstCFSafeControl Class

The `InputConstCFSafeControl` class extends the `CFSafeControl` and incorporates input constraints into the safety filtering process. It provides methods for assigning state and action dynamics, as well as action barriers.

- `assign_state_action_dynamics(state_dynamics, action_dynamics, action_output_function)`: Assigns both state and action dynamics along with an action output function.
- `assign_state_barrier(barrier)`: Assigns a state barrier to the safety filter.
- `assign_action_barrier(action_barrier, rel_deg)`: Assigns an action barrier and its relative degree.

### MinIntervInputConstCFSafeControl Class

The `MinIntervInputConstCFSafeControl` class extends the `InputConstCFSafeControl` and automatically assigns the desired control action while considering input constraints.

- `assign_desired_control(desired_control)`: Assigns the desired control action for the system.


### Usage Example

```python

# Create a Barrier object
barrier = Barrier().assign(barrier_func=h, rel_deg=2, alphas=[alpha1, alpha2])
barrier.assign_dynamics(dynamics)

# Compute the barrier function, HOCBF, and their Lie derivatives
barrier_value = barrier.barrier(x)
hocbf_value = barrier.hocbf(x)
Lf_hocbf = barrier.Lf_hocbf(x)
Lg_hocbf = barrier.Lg_hocbf(x)

# Create a CompositionBarrier object
barriers = [barrier1, barrier2, barrier3]
composition_barrier = SoftCompositionBarrier().assign_barriers_and_rule(barriers, 'union')

# Compute the composed barrier function and HOCBF
composed_barrier_value = composition_barrier.barrier(x)
composed_hocbf_value = composition_barrier.hocbf(x)
```



### Unicycle Example

The Unicycle example demonstrates the use of higher-order composition barrier functions for safe control of a unicycle robot navigating through obstacles to reach predefined goal locations.


Example code snippet:

```python
from cbftorch import Barrier, SoftCompositionBarrier, MinIntervCFSafeControl
from cbftorch.utils.dynamics import AffineInControlDynamics

# Instantiate dynamics
dynamics = UnicycleDynamics(state_dim=4, action_dim=2)

# Define barrier function configurations and dynamics parameters and create a Map object
map = Map(barriers_info=barriers_info, dynamics=dynamics, cfg=cfg)

# Simulate the unicycle's trajectory
```

If you use this code or find it helpful in your research, please consider citing our paper:

```bibtex
@article{rabiee2024closed,
  title={A Closed-Form Control for Safety Under Input Constraints Using a Composition of Control Barrier Functions},
  author={Rabiee, Pedram and Hoagg, Jesse B},
  journal={arXiv preprint arXiv:2406.16874},
  year={2024}
}
