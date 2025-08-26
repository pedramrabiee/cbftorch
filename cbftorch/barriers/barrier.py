from typing import List
from cbftorch.utils.utils import *
from cbftorch.utils.tensor_manager import ensure_batched, tensor_input, ensure_dtype


class Barrier:
    def __init__(self, cfg=None):
        """
        Initialize Barrier class.

        Parameters:
        - cfg (optional): Configuration parameters.
        """
        self._dynamics = None
        self.cfg = cfg
        self._barrier_func = None
        self._barriers = None
        self._hocbf = None
        self._rel_deg = None
        self._hocbf_func = None
        self._alphas = None
        
        # Gradient caching system
        self._gradient_cache = {}
        self._cache_enabled = True

    def assign(self, barrier_func, rel_deg=1, alphas=None):
        """
        Assign barrier function to the Barrier object.

        Parameters:
        - barrier_func: Barrier function to be assigned.
        - rel_deg: Relative degree of the barrier function.
        - alphas: List of class-K functions for higher-order barriers.

        Returns:
        - self: Updated Barrier object.
        """
        assert callable(barrier_func), "barrier_func must be a callable function"
        alphas = self._handle_alphas(alphas=alphas, rel_deg=rel_deg)

        # Assign barrier function definition: self._barrier_func is the main constraint that the
        # higher-order cbf is defined based on
        self._barrier_func = barrier_func
        self._rel_deg = rel_deg
        self._alphas = alphas
        return self

    def assign_dynamics(self, dynamics):
        """
         Assign dynamics to the Barrier object and generate higher-order barrier functions.

         Returns:
         - self: Updated Barrier object.
         """
        assert self._barrier_func is not None, \
            "Barrier functions must be assigned first. Use the assign method"

        self._dynamics = dynamics
        # make higher-order barrier function
        self._barriers = self._make_hocbf_series(barrier=self.barrier, rel_deg=self._rel_deg, alphas=self._alphas)
        self._hocbf_func = self._barriers[-1]
        return self

    def raise_rel_deg(self, x, raise_rel_deg_by=1, alphas=None):
        """
        This method takes the current hocbf and make a new hocbf with the relative degree raised
        by `raise_rel_deg_by`. The new hocbf has the relative degree of old rel_deg + raise_rel_deg_by
        """

        alphas = self._handle_alphas(alphas=alphas, rel_deg=raise_rel_deg_by)
        self._alphas.append(alphas)
        self._rel_deg += raise_rel_deg_by

        self._barriers.append(*self._make_hocbf_series(barrier=self._hocbf_func,
                                                       rel_deg=raise_rel_deg_by,
                                                       alphas=alphas))
        self._hocbf_func = self._barriers[-1]

    @tensor_input(ensure_batch=True, input_arg_index=1)
    @ensure_dtype(input_arg_index=1)
    def barrier(self, x):
        """
        Compute the barrier function barrier(x) for a given trajs x.
        """
        return apply_and_batchize(self._barrier_func, x)

    @tensor_input(ensure_batch=True, input_arg_index=1)
    @ensure_dtype(input_arg_index=1)
    def hocbf(self, x):
        """
        Compute the highest-order barrier function hocbf(x) for a given trajs x.
        """
        return apply_and_batchize(self._hocbf_func, x)

    @tensor_input(ensure_batch=True, input_arg_index=1)
    @ensure_dtype(input_arg_index=1)
    def get_hocbf_and_lie_derivs(self, x):
        # x is already standardized by decorators
        # Use cached gradient computation for efficiency
        hocbf, _, Lf_hocbf, Lg_hocbf = self._compute_cached_gradients(x)
        return hocbf, Lf_hocbf, Lg_hocbf


    @tensor_input(ensure_batch=True, input_arg_index=1)
    @ensure_dtype(input_arg_index=1)
    def Lf_hocbf(self, x):
        """
        Compute the Lie derivative of the highest-order barrier function with respect to the system dynamics f.
        """
        # Use cached gradients for efficiency
        _, _, Lf_hocbf, _ = self._compute_cached_gradients(x)
        return Lf_hocbf

    @tensor_input(ensure_batch=True, input_arg_index=1)
    @ensure_dtype(input_arg_index=1)
    def Lg_hocbf(self, x):
        """
        Compute the Lie derivative of the highest-order barrier function with respect to the system dynamics g.
        """
        # Use cached gradients for efficiency
        _, _, _, Lg_hocbf = self._compute_cached_gradients(x)
        return Lg_hocbf
        
    def _get_cache_key(self, x: torch.Tensor) -> str:
        """Generate cache key for gradient caching."""
        # Use a simple hash based on tensor properties and first few values
        # This balances performance with cache effectiveness
        sample_values = x.detach().flatten()[:min(16, x.numel())]  # Use first 16 values max
        sample_hash = hash(tuple(sample_values.cpu().numpy().round(decimals=8)))
        return f"grad_{x.shape}_{x.dtype}_{sample_hash}"
    
    def _compute_cached_gradients(self, x: torch.Tensor) -> tuple:
        """
        Compute and cache gradients for efficient reuse.
        
        Args:
            x: Input tensor (already standardized)
            
        Returns:
            Tuple of (hocbf, hocbf_deriv, Lf_hocbf, Lg_hocbf)
        """
        cache_key = self._get_cache_key(x)
        
        # Check cache first
        if self._cache_enabled and cache_key in self._gradient_cache:
            return self._gradient_cache[cache_key]
        
        # Compute gradients
        grad_req = x.requires_grad
        x.requires_grad_()
        
        # Compute HOCBF directly to avoid circular dependency
        # Note: we need gradients enabled for hocbf to compute hocbf_deriv
        hocbf = self._hocbf_func(x)
        if hocbf.ndim == 1:
            hocbf = hocbf.unsqueeze(-1)
        
        # Compute gradients
        hocbf_deriv = [grad(fval, x, create_graph=True)[0] for fval in hocbf.sum(0)]
        
        # Compute Lie derivatives using cached gradients
        Lf_hocbf = lie_deriv_from_values(hocbf_deriv, self._dynamics.f(x))
        Lg_hocbf = lie_deriv_from_values(hocbf_deriv, self._dynamics.g(x))
        
        # Restore gradient requirement
        x.requires_grad_(requires_grad=grad_req)
        
        # Cache results
        result = (hocbf.detach(), hocbf_deriv, Lf_hocbf.detach(), Lg_hocbf.detach())
        if self._cache_enabled:
            self._gradient_cache[cache_key] = result
        
        return result
    
    def clear_gradient_cache(self):
        """Clear the gradient cache."""
        self._gradient_cache.clear()
    
    def enable_gradient_caching(self, enabled: bool = True):
        """Enable or disable gradient caching."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_gradient_cache()

    def compute_barriers_at(self, x):
        """
        Compute barrier values at a given trajs x.
        """
        return [apply_and_batchize(func=barrier, x=x).detach() for barrier in self.barriers_flatten]

    def get_min_barrier_at(self, x):
        """
        Get the minimum barrier value at a given trajs x.
        """
        return torch.min(torch.hstack(self.compute_barriers_at(x)), dim=-1).values.unsqueeze(-1)

    def min_barrier(self, x):
        """
        Calculate the minimum value among all the barrier values computed at point x.
        """
        return torch.min(self.barrier(x), dim=-1).values.unsqueeze(-1)

    # Getters
    @property
    def rel_deg(self):
        """
        Get the relative degree of the barrier.
        """
        return self._rel_deg

    @property
    def barriers(self):
        """
         Get the list of barrier functions of all relative degrees upto self.rel_deg
        """
        return self._barriers

    @property
    def barriers_flatten(self):
        """
             Get the flatten list of barrier functions of all relative degrees. This method has application mainly
             in the composite barrier function class
        """
        return self.barriers

    @property
    def dynamics(self):
        """
        Get the dynamics associated with the system.
        """
        return self._dynamics

    @property
    def num_barriers(self):
        return len(self.barriers_flatten)

    # Helper methods

    def _make_hocbf_series(self, barrier, rel_deg, alphas):
        """
              Generate a series of higher-order barrier functions.

              Parameters:
              - barrier: Initial barrier function.
              - rel_deg: Relative degree of the barrier.
              - alphas: List of class-K functions.

          """
        ans = [barrier]
        for i in range(rel_deg - 1):
            # Create a proper class-based barrier function instead of lambda
            hocbf_i = self._create_hocbf_function(ans[i], alphas[i], i)
            ans.append(hocbf_i)
        return ans
    
    def _create_hocbf_function(self, prev_barrier, alpha, level):
        """
        Create an efficient HOCBF function without lambda closures.
        
        Args:
            prev_barrier: Previous barrier function in the series
            alpha: Class-K function for this level
            level: Level in the HOCBF series
            
        Returns:
            Optimized HOCBF function
        """
        class HOCBFFunction:
            def __init__(self, barrier_instance, prev_barrier, alpha, level):
                self.barrier_instance = barrier_instance
                self.prev_barrier = prev_barrier
                self.alpha = alpha
                self.level = level
                
            def __call__(self, x):
                # Use standardized tensor input
                from ..config import DEFAULT_DTYPE
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x, dtype=DEFAULT_DTYPE)
                
                # Compute previous barrier value
                prev_val = apply_and_batchize(self.prev_barrier, x)
                
                # Compute Lie derivative
                lie_deriv_val = lie_deriv(x, self.prev_barrier, self.barrier_instance._dynamics.f)
                
                # Compute alpha function
                alpha_val = apply_and_batchize(func=self.alpha, x=prev_val)
                
                return lie_deriv_val + alpha_val
        
        return HOCBFFunction(self, prev_barrier, alpha, level)
    
    def _compute_hocbf_vectorized(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vectorized computation of HOCBF for better performance.
        
        Args:
            x: Input tensor (already standardized)
            
        Returns:
            HOCBF values
        """
        # Use the highest-order barrier function directly
        return apply_and_batchize(self._hocbf_func, x)

    def _handle_alphas(self, alphas, rel_deg):
        if rel_deg > 1:
            if alphas is None:
                alphas = [(lambda x: x) for _ in range(rel_deg - 1)]
            assert isinstance(alphas, list) and len(alphas) == rel_deg - 1 and callable(alphas[0]), \
                "alphas must be a list with length equal to (rel_deg - 1) of callable functions "
        return alphas
