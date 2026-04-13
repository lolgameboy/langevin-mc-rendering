import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# Uncomment this line to use my own mitsuba build
# (Also, own build requires python 3.12.9 instead of 3.11.9)
sys.path.insert(1, '../../mitsuba3/build/python')

import mitsuba as mi
import drjit as dr
import numpy as np
np.set_printoptions(precision=12)

mi.set_variant("llvm_ad_rgb_double")
from pss_sampler import PssSampler
from lmc_integrator import LMC
from utils import render_mc
from trace_path import calculate_sample_contribution



scene = mi.load_file("../scenes/scene_directional.xml")
scene_name = "cornell_box_directional"

sensor = scene.sensors()[0]
film = sensor.film()
resolution = film.crop_size()

x_fov = mi.traverse(sensor)["x_fov"]
plane_width = 2 * dr.tan(dr.deg2rad(x_fov / 2))
plane_size = mi.Vector2f(plane_width, plane_width * resolution.y / resolution.x)

rng = dr.rng(seed=mi.UInt32(0))
sample = rng.random(mi.ArrayXf, (50,1))
calculate_sample_contribution(sample, scene, scene.sensors()[0].m_to_world, plane_size, resolution)

lmc = LMC()
render_mc(scene, scene.sensors()[0], False, 0.1 * 1000000)
lmc.render(scene, scene.sensors()[0], False, 0.1 * 1000000, 300000, True, 0.001, 0.005)