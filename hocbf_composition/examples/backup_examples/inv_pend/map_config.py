from box import Box as AD
from math import pi
from hocbf_composition.utils.utils import *


map_config = dict(
    geoms=(
        ('norm_boundary', AD(center=[0.0, 0.0], size=[pi, pi], p=100)),
    ))



map_config2 = dict(
    geoms=(
        ('norm_boundary', AD(center=[0.0, 0.0], size=[pi, 1.0], p=100)),
    ))



backup_barrier_functional = [make_ellipse_barrier_functional(center=[0, 0], A = [[17.85, 3.57], [3.57, 3.57]])]

# backup_barrier_functional = [make_ellipse_barrier_functional(center=[pi/2, 0], A = [[24, 4.0], [4.0, 5.6]])]

multibackup_barrier_functional = [make_ellipse_barrier_functional(center=[0, 0], A = [[17.85, 3.57], [3.57, 3.57]]),
                             make_ellipse_barrier_functional(center=[pi/2, 0], A = [[24, 4.0], [4.0, 5.6]]),
                             make_ellipse_barrier_functional(center=[-pi/2, 0], A = [[24, 4.0], [4.0, 5.6]])]