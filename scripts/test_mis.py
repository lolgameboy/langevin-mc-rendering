import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# Uncomment this line to use my own mitsuba build
# (Also, own build requires python 3.12.9 instead of 3.11.9)
sys.path.insert(1, '../../mitsuba3/build/python')



import mitsuba as mi
import drjit as dr
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=12)



mi.set_variant("llvm_ad_rgb_double")
from pss_sampler import PssSampler
from trace_path import calculate_sample_contribution, calculate_sample_contribution_bidir

scene = mi.load_file("../scenes/scene_metal.xml")
scene_name = "cornell_box"
sensor = scene.sensors()[0]
rng = dr.rng(seed=mi.UInt32(0))

cam_transform = sensor.m_to_world
film = sensor.film()
resolution = film.crop_size()
x_fov = mi.traverse(sensor)["x_fov"]
plane_width = 2 * dr.tan(dr.deg2rad(x_fov / 2))
plane_size = mi.Vector2f(plane_width, plane_width * resolution.y / resolution.x)
for i in range(10):
    sample = rng.random(mi.ArrayXf, (50,1))
    luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, max_depth=8)


