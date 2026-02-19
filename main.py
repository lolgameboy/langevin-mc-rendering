import drjit as dr
import mitsuba as mi

dr.set_flag(dr.JitFlag.Debug, True)
#dr.set_flag(dr.ADFlag.AllowNoGrad, True)

# Before importing own code!
mi.set_variant("llvm_ad_rgb")

import matplotlib.pyplot as plt
from pss_integrator import Pss
from pss_integrator import calculate_sample_contribution, calculate_sample_contribution_ref




from pss_sampler import PssSampler
scene = mi.load_file("scenes/scene.xml")
sample = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

pss_sampler = PssSampler(sample)
ray_origin_local = mi.Vector3f(0, 0, 0)
ray_direction_local = mi.Vector3f(0, 0, 1)
cam_transform = scene.sensors()[0].m_to_world
# Also determine affected pixel already
ray = mi.Ray3f(o=cam_transform.transform_affine(ray_origin_local), d=cam_transform.transform_affine(ray_direction_local))
diffray = mi.RayDifferential3f(ray)


pathtracer = mi.load_dict({
    'type': 'path',
    'max_depth': 8,
})


# Sample this ray
res = pathtracer.sample(scene, pss_sampler, diffray)
print(res)
print(res[0])


# Used for testing if my own path tracer is correct
@dr.syntax
def render_mc(scene: mi.Scene, sensor: mi.Sensor, seed: mi.UInt = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> mi.TensorXf:
        film = sensor.film()
        resolution = film.crop_size()

        image_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            3
        )

        sample_counts_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            1
        )

        cam_transform = sensor.m_to_world
        rng = dr.rng(seed=mi.UInt32(0))

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        # x_fov is a lie (or I am retarded somewhere else in my code)
        y_fov = mi.traverse(sensor)["x_fov"]
        plane_height = dr.tan(dr.deg2rad(y_fov)) # Originally divided y_fov by 2, but turns out it is already halved in spec?
        plane_size = mi.Vector2f(plane_height * resolution.x / resolution.y, plane_height)

        # TODO Maybe can make this faster by using a proper integrator and summing the final image
        N = mi.Int(10000000)
        i = mi.Int(0)

        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))

            dr.enable_grad(sample)
            luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng)
            dr.backward(luminance, dr.ADFlag.AllowNoGrad)
            dr.print(dr.grad(sample))
        
            image_block.put(mi.Point2f(pixel_x, pixel_y), res)
            sample_counts_block.put(mi.Point2f(pixel_x, pixel_y), [1])
            i += 1
        
        image = image_block.tensor() / sample_counts_block.tensor()
        #refimage = mi.render(scene, spp=100, integrator=mi.load_dict({'type':'path', 'max_depth':8}))
        #rmse = dr.sqrt(dr.mean(dr.square(refimage - image)))
        #dr.print(rmse)
        #return refimage
        return image



pss = Pss()

#img = pss.render(scene, scene.sensors()[0])
img = render_mc(scene, scene.sensors()[0])
# img = mi.render(scene, integrator=pathtracer, spp=20)

plt.axis("off")
plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping TODO why this needed?

plt.show()

