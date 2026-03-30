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

scene = mi.load_file("../scenes/scene.xml")
scene_name = "cornell_box"
sensor = scene.sensors()[0]
rng = dr.rng(seed=mi.UInt32(0))

cam_transform = sensor.m_to_world
film = sensor.film()
resolution = film.crop_size()
x_fov = mi.traverse(sensor)["x_fov"]
plane_width = 2 * dr.tan(dr.deg2rad(x_fov / 2))
plane_size = mi.Vector2f(plane_width, plane_width * resolution.y / resolution.x)
for i in range(3):
    sample = rng.random(mi.ArrayXf, (50,1))
    luminance, res, pixel_x, pixel_y = calculate_sample_contribution_bidir(sample, scene, cam_transform, plane_size, resolution, max_depth=4)
    print(res)


@dr.syntax
def render_mc(scene: mi.Scene, sensor: mi.Sensor, N : mi.Int, seed: mi.UInt = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> mi.TensorXf:
        film = sensor.film()
        # I now actually use the film to collect the samples,
        # it is fine like this, but 
        # this is not neccesary as long as I just use box filter in scene
        # Also, for PSS I need box filter anyway, so there i do it manually
        # Reason I need box filter is that otherwise block.put adds some 
        # filter weight making sample count division invalid (see pss code)
        film.prepare([])

        resolution = film.crop_size()
        image_block = film.create_block()

        cam_transform = sensor.m_to_world
        rng = dr.rng(seed=mi.UInt32(seed))

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        x_fov = mi.traverse(sensor)["x_fov"]
        plane_width = 2 * dr.tan(dr.deg2rad(x_fov / 2))
        plane_size = mi.Vector2f(plane_width, plane_width * resolution.y / resolution.x)

        i = mi.Int(0)

        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))

            luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, max_depth=4)
        
            value = dr.zeros(mi.ArrayXf, (image_block.channel_count(), 1))
            value[0] = res[0]
            value[1] = res[1]
            value[2] = res[2]
            value[3] = 1
            image_block.put(mi.Point2f(pixel_x, pixel_y), value)
            i += 1
        
        film.put_block(image_block)
        return film.develop()

img = render_mc(scene, sensor, 10000000)

plt.axis("off")
plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping TODO why this needed?
plt.show()