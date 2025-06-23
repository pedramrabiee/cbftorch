from box import Box as AD

# Make map_ configuration
map_config = dict(
    geoms=(
        ('norm_box', AD(center=[2.5, 1.0, 4.0], size=[3.0, 2.5, 5.0], p=20)),
        ('norm_box', AD(center=[-2.5, 2.5, 4.0], size=[1.25, 1.25, 5.0], p=20)),
        ('norm_box', AD(center=[-5.0, -5.0, 4.0], size=[1.875, 1.875, 5.0], p=20)),
        ('norm_box', AD(center=[5.0, -6.0, 4.0], size=[3.0, 2.0, 5.0], p=20)),
        ('norm_box', AD(center=[-7.0, 5.0, 4.0], size=[2.0, 3.0, 5.0], p=20)),
        ('norm_box', AD(center=[6.0, 7.0, 4.0], size=[3.0, 1.0, 5.0], p=20)),
        ('norm_boundary', AD(center=[0.0, 0.0, 4.0], size=[10.0, 10.0, 5.0], p=20)),
    ))
