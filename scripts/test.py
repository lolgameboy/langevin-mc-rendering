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


# sample = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

# scene = mi.load_file("../scenes/scene.xml")

# pss_sampler = PssSampler(sample)
# ray_origin_local = mi.Vector3f(0, 0, 0)
# ray_direction_local = mi.Vector3f(0, 0, 1)
# cam_transform = scene.sensors()[0].m_to_world
# # Also determine affected pixel already
# ray = mi.Ray3f(o=cam_transform.transform_affine(ray_origin_local), d=cam_transform.transform_affine(ray_direction_local))
# diffray = mi.RayDifferential3f(ray)


# pathtracer = mi.load_dict({
#     'type': 'path',
#     'max_depth': 8,
# })


# # Sample this ray
# res = pathtracer.sample(scene, pss_sampler, diffray)
# #print(res)
# #print(res[0])





def calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, max_depth = 6):
    pss_sampler = PssSampler(sample)
    # Sample an initial ray through the image plane
    ray_origin_local = mi.Vector3f(0, 0, 0)
    rand_x = pss_sampler.next_1d()
    rand_y = pss_sampler.next_1d()
    x = plane_size.x / 2 - rand_x * plane_size.x
    y = plane_size.y / 2 - rand_y * plane_size.y
    ray_direction_local = dr.normalize(mi.Vector3f(x, y, 1))

    # Also determine affected pixel already
    pixel_x = mi.Float(rand_x * resolution.x)
    pixel_y = mi.Float(rand_y * resolution.y)

    ray = mi.Ray3f(o=cam_transform.translation() + ray_origin_local, d=cam_transform.transform_affine(ray_direction_local))

    active = mi.Bool(True)
    throughput = 1
    L = mi.Spectrum(0)

    # ------------------------------------------------------------------
    # Path tracing loop
    # ------------------------------------------------------------------
    for depth in range(max_depth):
        si = scene.ray_intersect(ray, active)
        active &= si.is_valid()

        if not dr.any(active):
            break
        bsdf = si.bsdf()
        ctx = mi.BSDFContext()

        # Add emission (only on depth 0, see comment below)
        if depth == 0:
            emitter = si.emitter(scene)
            L += throughput * emitter.eval(si, active)

        # Add light ray
        ds, weight = scene.sample_emitter_direction(si, pss_sampler.next_2d())

        # ds.d is unit direction from si point to sampled point on emitter
        wo = si.to_local(ds.d)

        bsdf_val = bsdf.eval(ctx, si, wo, True)
        #print(f'Em:{bsdf_val}, {weight}')
        print(f'through:{throughput}')

        L += throughput * weight * bsdf_val

        # BSDF sampling

        # Loop 1: sample 4, 5, 6
        u_bsdf = pss_sampler.next_2d()
        u_component = pss_sampler.next_1d()

        bsdf_sample, bsdf_weight = bsdf.sample(
            ctx,
            si,
            u_component,
            u_bsdf,
            active
        )
        print(bsdf_sample, bsdf_weight)
        
        throughput *= bsdf_weight

        # Spawn next ray
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        #if depth == 5:
            #print(ray.o, ray.d)
            #print(f"{L.numpy()}")
            #print(throughput)

    luminance = mi.luminance(L)
    return luminance, L, pixel_x, pixel_y





from utils import calculate_sample_contribution_ref

#scene = mi.load_file("../scenes/veach-bidir/scene.xml")
scene = mi.load_file("../scenes/empty_box.xml")
sensor = scene.sensors()[0]
cam_transform = sensor.m_to_world
film = sensor.film()
resolution = film.crop_size()
# Determine size of "physical" image plane from fov parameter
x_fov = mi.traverse(sensor)["x_fov"]
plane_width = 2 * dr.tan(dr.deg2rad(x_fov / 2))
plane_size = mi.Vector2f(plane_width, plane_width * resolution.y / resolution.x)
# interesting seeds: 2, 7, 12
rng = dr.rng(seed=mi.UInt32(0))
sample_size = 32

for i in range(0):
    sample = rng.random(mi.ArrayXf, (sample_size,1))
    dr.enable_grad(sample)
    luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution)
    dr.backward(dr.log(dr.maximum(luminance, 1e-8)), dr.ADFlag.AllowNoGrad)
    gradlog = dr.grad(sample)
    for i in range(sample_size):
        eps = 1e-1
        sample[i] += eps
        luminance_after, _, _, _ = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution)
        ll = dr.log(dr.maximum(luminance, 1e-8))
        g = gradlog[i] * eps
        lla = dr.log(dr.maximum(luminance_after, 1e-8))
        print(f'log luminance: {ll}, gradient {i}: {g}, log luminance after: {lla}, err:{(lla - ll) - g}')
        sample[i] -= eps


# same but with ref
for i in range(0):
    sample = rng.random(mi.ArrayXf, (sample_size,1))

    luminance, res, pixel_x, pixel_y = calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution, rng)
    # Approx gradient
    eps = 1e-8
    gradlog = dr.zeros(mi.ArrayXf, (sample_size, 1))
    for sam_i in range(sample_size):
        sample[sam_i] += eps
        lum2, _, _, _ = calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution, rng)
        gradlog[sam_i] = (- dr.log(dr.maximum(lum2, 1e-8)) + dr.log(dr.maximum(luminance, 1e-8))) / eps
        sample[sam_i] -= eps

    for i in range(sample_size):
        eps = 1e-1
        sample[i] += eps
        luminance_after, _, _, _ = calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution, rng)
        ll = dr.log(dr.maximum(luminance, 1e-8))
        g = gradlog[i] * eps
        lla = dr.log(dr.maximum(luminance_after, 1e-8))
        print(f'log luminance: {ll}, gradient {i}: {g}, log luminance after: {lla}, err:{(lla - ll) - g}')
        sample[i] -= eps

for i in range(0):
    sample = rng.random(mi.ArrayXf, (sample_size,1))
    dr.enable_grad(sample)
    print("BEFORE")
    luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution)
    dr.backward(dr.log(dr.maximum(luminance, 1e-8)), dr.ADFlag.AllowNoGrad)
    gradlog = dr.grad(sample)
    print("AFTER")
    for i in range(14, 30):
        print(f'Sample {i}:')
        eps = 1e-3
        sample[i] += eps
        luminance_after, _, _, _ = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution)
        ll = dr.log(dr.maximum(luminance, 1e-8))
        g = gradlog[i] * eps
        lla = dr.log(dr.maximum(luminance_after, 1e-8))
        #print(f'log luminance: {ll}, gradient {i}: {g}, log luminance after: {lla}, err:{(lla - ll) - g}')
        sample[i] -= eps