from box import Box as AD
from cbftorch.barriers.barrier import Barrier
from cbftorch.barriers.composite_barrier import SoftCompositionBarrier, NonSmoothCompositionBarrier
from cbftorch.utils.utils import *
import cv2


class Map:
    def __init__(self, barriers_info, dynamics, cfg):
        self.barriers_info = AD(barriers_info)
        self.dynamics = dynamics
        self.cfg = AD(cfg)
        self.pos_barriers = None
        self.vel_barriers = None
        self.make_barrier_from_map()

    def make_barrier_from_map(self):
        if hasattr(self.barriers_info,'geoms'):
            self.pos_barriers = self.make_position_barrier_from_map()
        elif hasattr(self.barriers_info,'image'):
            self.pos_barriers = self.make_position_barrier_from_image()
        if hasattr(self.barriers_info, 'velocity'):
            self.vel_barriers = self.make_velocity_barrier()
            self.barrier = SoftCompositionBarrier(
                cfg=self.cfg).assign_barriers_and_rule(barriers=[*self.pos_barriers, *self.vel_barriers],
                                                       rule='i',
                                                       infer_dynamics=True)
        else:
            self.barrier = SoftCompositionBarrier(
                cfg=self.cfg).assign_barriers_and_rule(barriers=[*self.pos_barriers],
                                                       rule='i',
                                                       infer_dynamics=True)

        self.map_barrier = NonSmoothCompositionBarrier(
            cfg=self.cfg).assign_barriers_and_rule(barriers=[*self.pos_barriers],
                                                   rule='i',
                                                   infer_dynamics=True)

    def get_barriers(self):
        return self.pos_barriers, self.vel_barriers

    def make_position_barrier_from_map(self):
        geoms = self.barriers_info.geoms
        barriers = []

        for geom_type, geom_info in geoms:

            if geom_type == 'cylinder':
                barrier_func = make_circle_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
            elif geom_type == 'box':
                barrier_func = make_affine_rectangular_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
            elif geom_type == 'norm_box':
                barrier_func = make_norm_rectangular_barrier_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
            elif geom_type == 'boundary':
                barrier_func = make_affine_rectangular_boundary_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.boundary_alpha)
            elif geom_type == 'norm_boundary':
                barrier_func = make_norm_rectangular_boundary_functional
                alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.boundary_alpha)
            else:
                raise NotImplementedError
            barriers.append(
                Barrier().assign(barrier_func=barrier_func(**geom_info),
                                 rel_deg=self.cfg.pos_barrier_rel_deg,
                                 alphas=alphas).assign_dynamics(self.dynamics))
        return barriers


    def make_position_barrier_from_image(self):
        self.image_path = self.barriers_info.image
        self.synthesis_cfg = self.cfg.synthesis_cfg if hasattr(self.cfg, 'synthesis_cfg') else None

        params = {k: v for k, v in {
            'safety_margin': self.synthesis_cfg.safety_margin,
            'pixels_per_meter': self.synthesis_cfg.pixels_per_meter,
            'downsample_rate': self.synthesis_cfg.downsample_rate
        }.items() if v is not None}

        bnd_points, safe_points = self.map_sampler(**params)
        points_cat = torch.cat([bnd_points, safe_points])
        labels = torch.cat([-torch.ones(bnd_points.shape[0], dtype=torch.float64),
                            torch.ones(safe_points.shape[0], dtype=torch.float64)]).unsqueeze(-1)

        pos_func = SVM(self.synthesis_cfg).fit(points_cat, labels)
        barrier_func = lambda x: pos_func(self.dynamics.get_pos(x))
        alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.obstacle_alpha)
        barriers = Barrier().assign(barrier_func=barrier_func,
                                    rel_deg=self.cfg.pos_barrier_rel_deg,
                                    alphas=alphas).assign_dynamics(self.dynamics)
        return [barriers]



    def make_velocity_barrier(self):
        alphas = make_linear_alpha_function_form_list_of_coef(self.cfg.velocity_alpha)
        idx, bounds = self.barriers_info.velocity
        vel_barriers = make_box_barrier_functionals(bounds=bounds, idx=idx)
        barriers = [Barrier().assign(
            barrier_func=vel_barrier,
            rel_deg=self.cfg.vel_barrier_rel_deg,
            alphas=alphas).assign_dynamics(self.dynamics) for vel_barrier in vel_barriers]

        return barriers


    def map_sampler(self, safety_margin=0.5, pixels_per_meter=250, downsample_rate=5):
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








