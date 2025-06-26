from abc import ABC, abstractmethod
from box import Box as AD
from cbftorch.barriers.barrier import Barrier
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier, NonSmoothCompositionBarrier
from cbftorch.utils.utils import *
import cv2


class GeometryProvider(ABC):
    """Abstract base class for providing geometry data to the map."""

    @abstractmethod
    def get_geometries(self):
        """
        Return list of (geom_type, geom_info) tuples.

        geom_type can be: 'cylinder', 'box', 'norm_box', 'boundary', 'norm_boundary'
        geom_info is a dict with parameters specific to each geometry type
        """
        pass

    def get_velocity_constraints(self):
        """
        Return velocity constraints as (idx, bounds) tuple or None.

        idx: indices of velocity components to constrain
        bounds: list of (min, max) tuples for each component
        """
        return None


class StaticGeometryProvider(GeometryProvider):
    """Provides pre-defined static geometry from a configuration dict."""

    def __init__(self, geoms_config):
        """
        Args:
            geoms_config: List of (geom_type, geom_info) tuples or dict with 'geoms' key
        """
        if isinstance(geoms_config, dict):
            self.geoms = geoms_config.get('geoms', [])
            self.velocity = geoms_config.get('velocity', None)
        else:
            self.geoms = geoms_config
            self.velocity = None

    def get_geometries(self):
        return self.geoms

    def get_velocity_constraints(self):
        return self.velocity


class ImageGeometryProvider(GeometryProvider):
    """Provides geometry by processing an image file."""

    def __init__(self, image_path, synthesis_cfg):
        self.image_path = image_path
        self.synthesis_cfg = AD(synthesis_cfg) if synthesis_cfg else AD()
        self._barrier_func = None
        self._process_image()

    def get_geometries(self):
        # Return empty list as we'll handle this specially in Map
        return []

    def get_barrier_function(self, dynamics):
        """Get the barrier function created from image processing."""
        if self._barrier_func is None:
            raise RuntimeError("Image processing failed")
        return lambda x: self._barrier_func(dynamics.get_pos(x))

    def _process_image(self):
        """Process image and create SVM-based barrier function."""
        params = {
            'safety_margin': getattr(self.synthesis_cfg, 'safety_margin', 0.5),
            'pixels_per_meter': getattr(self.synthesis_cfg, 'pixels_per_meter', 250),
            'downsample_rate': getattr(self.synthesis_cfg, 'downsample_rate', 5)
        }

        bnd_points, safe_points = self._sample_from_image(**params)
        points_cat = torch.cat([bnd_points, safe_points])
        labels = torch.cat([
            -torch.ones(bnd_points.shape[0], dtype=torch.float64),
            torch.ones(safe_points.shape[0], dtype=torch.float64)
        ]).unsqueeze(-1)

        self._barrier_func = SVM(self.synthesis_cfg).fit(points_cat, labels)

    def _sample_from_image(self, safety_margin=0.5, pixels_per_meter=250, downsample_rate=5):
        """Sample obstacle and safe points from image."""
        safety_margin_pixels = int(safety_margin * pixels_per_meter)

        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        height = img.shape[0]
        obstacle_points = []
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] >= 0:
                # Downsample obstacle points
                for j in range(0, len(contour), downsample_rate):
                    point = contour[j]
                    x = point[0][0] / pixels_per_meter
                    y = (height - point[0][1]) / pixels_per_meter
                    obstacle_points.append([x, y])

        kernel = np.ones((safety_margin_pixels * 2, safety_margin_pixels * 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)

        safe_contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        safe_points = []
        for i, contour in enumerate(safe_contours):
            if hierarchy[0][i][3] >= 0:
                # Downsample safe points
                for j in range(0, len(contour), downsample_rate):
                    point = contour[j]
                    x = point[0][0] / pixels_per_meter
                    y = (height - point[0][1]) / pixels_per_meter
                    safe_points.append([x, y])

        return torch.tensor(obstacle_points, dtype=torch.float64), torch.tensor(safe_points, dtype=torch.float64)


class Map:
    """
    Main Map class that creates barriers from geometry providers.

    This class can be initialized in multiple ways:
    1. With a GeometryProvider instance
    2. With barriers_info dict (creates StaticGeometryProvider)
    3. With image path (creates ImageGeometryProvider)
    """

    def __init__(self, dynamics, cfg, geometry_provider=None, barriers_info=None,
                 image_path=None, synthesis_cfg=None):
        """
        Initialize Map with one of the following:
        - geometry_provider: Custom GeometryProvider instance
        - barriers_info: Dict with 'geoms' and optionally 'velocity' keys
        - image_path: Path to image file for image-based barriers

        Args:
            dynamics: System dynamics
            cfg: Configuration dict with barrier parameters
            geometry_provider: Optional custom geometry provider
            barriers_info: Optional dict with geometry information
            image_path: Optional path to image file
            synthesis_cfg: Optional config for image synthesis (required if image_path provided)
        """
        self.dynamics = dynamics
        self.cfg = AD(cfg)

        # Initialize geometry provider based on inputs
        if geometry_provider is not None:
            self.geometry_provider = geometry_provider
        elif barriers_info is not None:
            # Handle legacy dict format with potential 'image' key
            if isinstance(barriers_info, dict) and 'image' in barriers_info:
                self.geometry_provider = ImageGeometryProvider(
                    barriers_info['image'],
                    synthesis_cfg or cfg.get('synthesis_cfg')
                )
            else:
                self.geometry_provider = StaticGeometryProvider(barriers_info)
        elif image_path is not None:
            self.geometry_provider = ImageGeometryProvider(image_path, synthesis_cfg)
        else:
            raise ValueError("Must provide one of: geometry_provider, barriers_info, or image_path")

        self.pos_barriers = None
        self.vel_barriers = None
        self._create_barriers()

    def _create_barriers(self):
        """Create position and velocity barriers from geometry provider."""
        # Create position barriers
        if isinstance(self.geometry_provider, ImageGeometryProvider):
            self.pos_barriers = self._create_image_barriers()
        else:
            self.pos_barriers = self._create_geometric_barriers()

        # Create velocity barriers
        velocity_constraints = self.geometry_provider.get_velocity_constraints()
        if velocity_constraints:
            self.vel_barriers = self._create_velocity_barriers(velocity_constraints)
        else:
            self.vel_barriers = []

        # Combine all barriers
        all_barriers = self.pos_barriers + self.vel_barriers

        # Create composite barriers
        self.barrier = SoftCompositionBarrier(
            cfg=self.cfg
        ).assign_barriers_and_rule(
            barriers=all_barriers,
            rule='i',
            infer_dynamics=True
        )

        self.map_barrier = NonSmoothCompositionBarrier(
            cfg=self.cfg
        ).assign_barriers_and_rule(
            barriers=self.pos_barriers,
            rule='i',
            infer_dynamics=True
        )

    def _create_geometric_barriers(self):
        """Create barriers from geometric primitives."""
        barriers = []
        geoms = self.geometry_provider.get_geometries()

        for geom_type, geom_info in geoms:
            barrier_func, alpha_key = self._get_barrier_config(geom_type)
            alphas = make_linear_alpha_function_form_list_of_coef(
                getattr(self.cfg, alpha_key)
            )

            barriers.append(
                Barrier().assign(
                    barrier_func=barrier_func(**geom_info),
                    rel_deg=self.cfg.pos_barrier_rel_deg,
                    alphas=alphas
                ).assign_dynamics(self.dynamics)
            )

        return barriers

    def _create_image_barriers(self):
        """Create barriers from image-based geometry provider."""
        barrier_func = self.geometry_provider.get_barrier_function(self.dynamics)
        alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)

        barrier = Barrier().assign(
            barrier_func=barrier_func,
            rel_deg=self.cfg.pos_barrier_rel_deg,
            alphas=alphas
        ).assign_dynamics(self.dynamics)

        return [barrier]

    def _create_velocity_barriers(self, velocity_constraints):
        """Create velocity barriers from constraints."""
        idx, bounds = velocity_constraints
        alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.velocity_alpha)
        vel_barriers = make_box_barrier_functionals(bounds=bounds, idx=idx)

        barriers = [
            Barrier().assign(
                barrier_func=vel_barrier,
                rel_deg=self.cfg.vel_barrier_rel_deg,
                alphas=alphas
            ).assign_dynamics(self.dynamics)
            for vel_barrier in vel_barriers
        ]

        return barriers

    def _get_barrier_config(self, geom_type):
        """Get barrier function and alpha configuration key for geometry type."""
        mapping = {
            'cylinder': (make_circle_barrier_functional, 'obstacle_alpha'),
            'box': (make_affine_rectangular_barrier_functional, 'obstacle_alpha'),
            'norm_box': (make_norm_rectangular_barrier_functional, 'obstacle_alpha'),
            'boundary': (make_affine_rectangular_boundary_functional, 'boundary_alpha'),
            'norm_boundary': (make_norm_rectangular_boundary_functional, 'boundary_alpha'),
        }

        if geom_type not in mapping:
            raise NotImplementedError(f"Geometry type '{geom_type}' not supported")

        return mapping[geom_type]

    def get_barriers(self):
        """Get position and velocity barriers separately."""
        return self.pos_barriers, self.vel_barriers


# Convenience functions for common use cases
def make_map_from_geoms(geoms, dynamics, cfg, velocity_constraints=None):
    """
    Create a map from a list of geometry specifications.

    Args:
        geoms: List of (geom_type, geom_info) tuples
        dynamics: System dynamics
        cfg: Barrier configuration
        velocity_constraints: Optional (idx, bounds) tuple

    Returns:
        Map instance
    """
    barriers_info = {'geoms': geoms}
    if velocity_constraints:
        barriers_info['velocity'] = velocity_constraints

    return Map(dynamics=dynamics, cfg=cfg, barriers_info=barriers_info)


def make_map_from_image(image_path, dynamics, cfg, synthesis_cfg=None):
    """
    Create a map from an image file.

    Args:
        image_path: Path to image file
        dynamics: System dynamics
        cfg: Barrier configuration
        synthesis_cfg: Optional synthesis configuration for image processing

    Returns:
        Map instance
    """
    return Map(
        dynamics=dynamics,
        cfg=cfg,
        image_path=image_path,
        synthesis_cfg=synthesis_cfg or cfg.get('synthesis_cfg')
    )