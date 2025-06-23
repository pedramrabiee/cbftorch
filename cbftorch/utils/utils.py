import torch
import numpy as np
from torch.autograd import grad
import math
from functools import partial
from torchdiffeq import odeint
from cvxopt import solvers,matrix



def make_circle_barrier_functional(center, radius):
    center = vectorize_tensors(center).to(torch.float64)
    return lambda x: torch.norm((vectorize_tensors(x)[..., :2] - center.to(x.device)), p=2,
                                dim=-1) / radius - 1


def make_norm_rectangular_barrier_functional(center, size, rotation=0.0, p=20):
    size, center = vectorize_tensors(size).to(torch.float64), vectorize_tensors(center).to(torch.float64)
    return lambda x: torch.norm(
        (rotate_tensors(points=vectorize_tensors(x), center=center.to(x.device),
                        angle_rad=-rotation) - center.to(x.device)) / size.to(x.device),
        p=p,
        dim=-1) - 1


def make_affine_rectangular_barrier_functional(center, size, rotation=0.0, smooth=False, rho=40):
    size, center = vectorize_tensors(size).to(torch.float64), vectorize_tensors(center).to(torch.float64)

    # Define the normals for the axis-aligned rectangle (in local coordinate system)
    A = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.float64)
    # Rotate the normals
    b = torch.tensor([
        center[0] + size[0],
        -center[0] + size[0],
        center[1] + size[1],
        -center[1] + size[1]
    ], dtype=torch.float64).unsqueeze(0)

    def affine_rectangle(x):
        rotate_x = rotate_tensors(points=vectorize_tensors(x[..., :2]), center=center.to(x.device), angle_rad=-rotation)
        ans = torch.einsum('mn,bn->bm', A, rotate_x) - b
        if smooth:
            return softmax(x=ans, rho=rho, dim=-1)
        return torch.max(ans , dim=-1).values


    # Return the affine constraint function
    return affine_rectangle

def make_norm_rectangular_boundary_functional(center, size, rotation=0.0, p=20):
    return lambda x: -make_norm_rectangular_barrier_functional(center, size, rotation, p)(x)

def make_affine_rectangular_boundary_functional(center, size, rotation=0.0, smooth=False, rho=40):
    return lambda x: -make_affine_rectangular_barrier_functional(center, size, rotation, smooth, rho)(x)


def make_box_barrier_functionals(bounds, idx):
    lb, ub = bounds
    # TODO: test dimensions
    return [lambda x: vectorize_tensors(x)[..., idx] - torch.tensor(lb), lambda x: torch.tensor(ub) -
                                                                                   vectorize_tensors(x)[..., idx]]


def make_ellipse_barrier_functional(center, A):

    center = vectorize_tensors(tensify(center)).to(torch.float64)
    A = tensify(A).to(torch.float64)
    return lambda x: 1 - torch.einsum('bi,ij,bj->b', vectorize_tensors(x) - center, A, vectorize_tensors(x) - center).unsqueeze(-1)


def make_linear_alpha_function_form_list_of_coef(coef_list):
    return [(lambda x, c=c: c * x) for c in coef_list]


def make_cubic_alpha_function_form_list_of_coef(coef_list):
    return [(lambda x, c=c: c * x ** 3) for c in coef_list]


def make_tanh_alpha_function_form_list_of_coef(coef_list):
    return [(lambda x, c=c: c * torch.tanh(x)) for c in coef_list]


def vectorize_tensors(arr):
    if isinstance(arr, torch.Tensor):
        return arr.unsqueeze_(0) if arr.ndim == 1 else arr
    return tensify(arr)


def tensify(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, (list, tuple)):
        return torch.tensor(obj)
    raise ValueError("Unsupported object type for conversion to tensor")


def apply_and_batchize(func, x):
    res = func(vectorize_tensors(x))
    if x.ndim == 2 and res.ndim == 1:
        return res.view(-1, 1)
    return res


def softmin(x, rho, conservative=False, dim=0):
    return softmax(x=x, rho=-rho, conservative=conservative, dim=dim)


def softmax(x, rho, conservative=True, dim=0):
    res = 1 / rho * torch.logsumexp(rho * x, dim=dim)
    return res - np.log(x.size(dim)) / rho if conservative else res


def lie_deriv(x, func, field):
    x = vectorize_tensors(x)
    func_deriv = get_func_deriv(x, func)
    field_val = field(x)
    return lie_deriv_from_values(func_deriv, field_val)


def get_func_deriv(x, func):
    x = vectorize_tensors(x)
    grad_req = x.requires_grad
    x.requires_grad_()
    func_val = func(x).sum(0)
    func_deriv = get_func_deriv_from_func_vals(x, func_val)
    x.requires_grad_(requires_grad=grad_req)
    return func_deriv


def get_func_deriv_from_func_vals(x, func_val):
    func_deriv = [grad(fval, x, create_graph=True)[0] for fval in func_val]
    return func_deriv


def lie_deriv_from_values(func_deriv, field_val):
    assert field_val.ndim in {2, 3}, 'Field dimension is not accepted'

    if field_val.ndim == 2:
        return torch.cat([torch.einsum('bn,bn->b', fderiv, field_val).unsqueeze(1) for fderiv in func_deriv], dim=1)

    if field_val.ndim == 3:
        return torch.einsum('rbn,bnm->brm', torch.stack(func_deriv), field_val).squeeze(1)


def make_higher_order_lie_deriv_series(func, field, deg):
    """
          Generate a series of higher-order lie derivative.

          Parameters:

      """
    ans = [func]
    for i in range(deg):
        holie_i = lambda x, holie=ans[i], f=field: lie_deriv(x, holie, f)
        ans.append(holie_i)
    return ans


def rotate_tensors(points, center, angle_rad):
    center_size = center.shape[0]
    rotation_matrix = torch.tensor([[math.cos(angle_rad), -math.sin(angle_rad)],
                                    [math.sin(angle_rad), math.cos(angle_rad)]], dtype=torch.float64).to(points.device)
    rotated_xy = torch.matmul(points[..., :2] - center[..., :2], rotation_matrix.t()) + center[..., :2]
    return torch.cat([rotated_xy, points[..., 2:center_size]], dim=-1)


def apply_and_match_dim(func, x):
    # Apply the function to the input tensor
    res = func(x)

    if x.ndim == 1 and res.ndim == 1:
        return res
    # Check if dimensions need to be matched
    if x.ndim == 1 and res.ndim == 2:
        return res.squeeze_(0)
    if x.ndim == 2 and res.ndim == 1:
        return res.view(-1, 1)

    return res


def get_trajs_from_action_func(x0, dynamics, action_func, timestep, sim_time, method='euler'):
    return odeint(func=lambda t, y: dynamics.rhs(y, action_func(y)),
                  y0=x0,
                  t=torch.linspace(0.0, sim_time, int(sim_time / timestep) + 1),
                  method=method).detach()


# def get_trajs_from_action_func_zoh(x0, dynamics, action_func, timestep, sim_time, method='euler'):
#     # dynamics = dynamics.reset_zoh_time()
#     return odeint(func=lambda t, y: partial(dynamics.rhs_zoh, action_func=action_func, timestep=timestep)(t, y),
#                   y0=x0,
#                   t=torch.linspace(0.0, sim_time, int(sim_time / timestep) + 1),
#                   method=method).detach()

def get_trajs_from_action_func_zoh(x0, dynamics, action_func, timestep, sim_time, intermediate_steps,
                                           method='dopri5'):
    # Get dimensions
    batch_size = x0.shape[0]
    state_dim = x0.shape[1]
    num_steps = int(sim_time / timestep) + 1

    # Pre-allocate trajectories
    trajs = torch.zeros((num_steps, batch_size, state_dim), device=x0.device, dtype=torch.float64)
    trajs[0] = x0

    # Create intermediate time points for better integration
    t_local = torch.linspace(0, timestep, intermediate_steps, device=x0.device, dtype=torch.float64)

    # Simulate system one timestep at a time
    for i in range(num_steps - 1):
        # Compute control for current batch of states
        current_controls = action_func(trajs[i])

        # Integrate each state in the batch with its corresponding control
        # Only keep the final state
        next_states = odeint(
            lambda t, x: dynamics.rhs(x, current_controls),
            trajs[i],
            t_local,
            method=method
        )[-1]

        trajs[i + 1] = next_states

    return trajs.detach()


def update_dict_no_overwrite(original_dict, new_dict):
    for key, value in new_dict.items():
        if key not in original_dict:
            original_dict[key] = value




def get_trajs_from_batched_action_func(x0, dynamics, action_funcs, timestep, sim_time, method='euler'):
    action_num = len(action_funcs)
    return odeint(
            func=lambda t, y: torch.cat([dynamics.rhs(yy.squeeze(0), action(yy.squeeze(0)))
                                         for yy, action in zip(y.chunk(action_num, dim=1), action_funcs)],
                                        dim=0),
            y0=x0.unsqueeze(0).repeat(1, action_num, 1),
            t=torch.linspace(0.0, sim_time, int(sim_time / timestep)),
            method=method
        ).squeeze(1)


def lp_solver(c, G, h):

    batch_size, n = c.shape

    # Prepare empty list to store the solutions
    solutions = []

    # Solve each LP problem individually
    for i in range(batch_size):
        # Extract the c, G, h for the i-th problem
        c_np = c[i].numpy()  # Convert to NumPy
        G_np = G[i].numpy()  # Convert to NumPy
        h_np = h[i].numpy()  # Convert to NumPy


        # Scale the problem to improve numerical stability
        scale_c = np.max(np.abs(c_np)) if np.max(np.abs(c_np)) > 0 else 1.0
        scale_G = np.max(np.abs(G_np)) if np.max(np.abs(G_np)) > 0 else 1.0
        scale_h = np.max(np.abs(h_np)) if np.max(np.abs(h_np)) > 0 else 1.0

        c_scaled = matrix((c_np / scale_c).astype(np.float64))
        G_scaled = matrix((G_np / scale_G).astype(np.float64))
        h_scaled = matrix((h_np / scale_h).astype(np.float64))




        # Solve the LP using CVXOPT's linprog function
        solvers.options['show_progress'] = False
        sol = solvers.lp(c_scaled, G_scaled, h_scaled, msg=False)


        # Extract the solution and append it to the list
        x = np.array(sol['x']).flatten() * (scale_h / scale_G)
        solutions.append(x)

    # Convert solutions to a PyTorch tensor and return
    return torch.tensor(np.array(solutions), dtype=c.dtype, device=c.device)



class SVM:
    def __init__(self, cfg):
        self.cfg = cfg
        self.support_vectors = None
        self.support_vector_labels = None
        self.omega = None

        # Create the kernel function using the factory
        self.kernel_fn = self._kernel_factory(cfg.kernel_type,
                                              sigma=cfg.sigma if cfg.kernel_type == 'rbf' and hasattr(
                                                  cfg, 'sigma') else None,
                                              degree=cfg.degree if cfg.kernel_type == 'polynomial' and hasattr(
                                                  cfg, 'degree') else None,
                                              coef=cfg.coef if cfg.kernel_type == 'polynomial' and hasattr(
                                                  cfg, 'coef') else None)


    def fit(self, x, y):
        raise NotImplementedError

    def predict(self, x):
        if self.support_vectors is None:
            raise ValueError("Model not trained yet")
        kernel_matrix = self.kernel_fn(x, self.support_vectors)
        return torch.sign(torch.mm(kernel_matrix, self.omega * self.support_vector_labels) + self.b)


    def _kernel_factory(self, func_name, **kwargs):
        def linear(x1, x2, sigma=None, degree=None, coef=None):
            return torch.einsum('ik,jk->ij', x1, x2)

        def rbf(x1, x2, sigma=1.0, degree=None, coef=None):

            x1_norm_squared = torch.einsum('ij,ij->i', x1, x1)
            x2_norm_squared = torch.einsum('ij,ij->i', x2, x2)

            pairwise_sq_dists = x1_norm_squared.unsqueeze(1) + x2_norm_squared.unsqueeze(0) - 2 * torch.einsum(
                'ik,jk->ij', x1, x2)

            return torch.exp(-pairwise_sq_dists / (2 * sigma ** 2))


        def polynomial(x1, x2, sigma=None, degree=3.0, coef=1.0):
            coef = tensify(coef).to(torch.float64)
            return (torch.einsum('ik,jk->ij', x1, x2) + coef) ** degree

        kernel_functions = {
            'linear': linear,
            'polynomial': polynomial,
            'rbf': rbf,
        }

        assert func_name in kernel_functions, "Kernel function method not implemented"

        return partial(kernel_functions[func_name], **kwargs)

    def boundary_func(self, X, y, sv, sv_y, lambdas, b):

        return lambda x: (torch.einsum('ik,ik,ni->nk', lambdas, sv_y, self.kernel_fn(x, sv)).squeeze(-1) + b).unsqueeze(-1)

    def fit(self, X, y, lambdas=None):
        n_samples, n_features = X.shape
        kernel_dot = self.kernel_fn(X, X)

        Q = torch.einsum('ik,jk->ij', y, y) * kernel_dot
        c = -torch.ones(n_samples, dtype=torch.float64)

        G_lower = -torch.eye(n_samples, dtype=torch.float64)
        h_lower = torch.zeros(n_samples, dtype=torch.float64)

        G_upper = torch.eye(n_samples, dtype=torch.float64)
        asymmetric_ind = (y == 1).squeeze(-1)
        G_upper_masked = G_upper[asymmetric_ind]

        h_upper = torch.ones(G_upper_masked.shape[0]) * self.cfg.safe_slack

        G = torch.cat([G_upper_masked, G_lower], dim=0)
        h = torch.cat([h_upper, h_lower], dim=0)

        A = y.t()
        b = torch.zeros(1, dtype=torch.float64)

        solvers.options['show_progress'] = False
        Q_np = Q.numpy()
        c_np = c.numpy()
        G_np = G.numpy()
        h_np = h.numpy()
        A_np = A.numpy()
        b_np = b.numpy()


        solution = solvers.qp(matrix(Q_np), matrix(c_np),
                              matrix(G_np), matrix(h_np),
                              matrix(A_np), matrix(b_np))
        lambdas = (torch.tensor(np.array(solution['x']), dtype=torch.float64))


        # Find support vectors
        sol_indx = (lambdas > 1e-5).flatten()
        lambdas = lambdas[sol_indx]
        sv_y = y[sol_indx]
        sv = X[sol_indx]

        b = torch.sum(sv_y - torch.sum(self.kernel_fn(sv, sv) * lambdas * sv_y, dim=-1).unsqueeze(-1), dim=0) / sv.shape[0]

        return self.boundary_func(X, y, sv, sv_y, lambdas, b)