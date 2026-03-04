from pathlib import Path
import drjit as dr
import mitsuba as mi

dr.set_flag(dr.JitFlag.Debug, True)
#dr.set_flag(dr.ADFlag.AllowNoGrad, True)

# Before importing own code!
mi.set_variant("llvm_ad_rgb")

import matplotlib.pyplot as plt
from lmc_integrator import LMC
from trace_path import calculate_sample_contribution, calculate_sample_contribution_ref
from utils import render_ref, render_convergence, render_mc
from pss_sampler import PssSampler

scene = mi.load_file("../scenes/scene.xml")
scene_name = "cornell_box"



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
#print(res)
#print(res[0])

def render_scene(method, scene, N, use_cached = True, integrand_samples = 100000,
                stepsize = 0.01,
                large_mut_chance = 0.0001,
                precond = True,
                beta = 0.999,
                delta = 0.001,
                momentum = True,
                alpha = 0.9,
                dimin_adapt = True,
                dimin_adapt_coeff_M = 0.000000001,
                dimin_adapt_coeff_m = 0.00000001,
                seed = 0):
    if method == "lmc":
        fileName = f'lmc_{N}_{integrand_samples}_{stepsize}_{large_mut_chance}'
        if precond:
            fileName += f'_pre_{beta}_{delta}'
        if momentum:
            fileName += f'_mom_{alpha}'
        if dimin_adapt and (precond or momentum):
            fileName += f'_da'
            if precond:
                fileName += f'_{dimin_adapt_coeff_M}'
            if momentum:
                fileName += f'_{dimin_adapt_coeff_m}'

        pathStr = f'cache/{scene_name}/{fileName}.exr'
        pngPathStr = f'cache/{scene_name}/{fileName}.png'
        def render_function():
            lmc = LMC()
            return lmc.render(scene, scene.sensors()[0], N, integrand_samples, False, 
                              stepsize, large_mut_chance, 
                              precond, beta, delta, 
                              momentum, alpha, 
                              dimin_adapt, dimin_adapt_coeff_M, dimin_adapt_coeff_m, 
                              seed)

    if method == "pss":
        fileName = f'pss_{N}_{integrand_samples}_{stepsize}_{large_mut_chance}'
        pathStr = f'cache/{scene_name}/{fileName}.exr'
        pngPathStr = f'cache/{scene_name}/{fileName}.png'
        def render_function():
            lmc = LMC()
            return lmc.render(scene, scene.sensors()[0], N, True)

    if method == "mc":
        pathStr = f'cache/{scene_name}/mc_{N}.exr'
        pngPathStr = f'cache/{scene_name}/mc_{N}.png'
        def render_function():
            return render_mc(scene, scene.sensors()[0], N, True)
    
    path = Path(pathStr)
    if path.exists() and use_cached:
        bmp = mi.Bitmap(pathStr)
        image = mi.TensorXf(bmp)
    else:
        image = render_function()
        mi.Bitmap(image).write(pathStr)
        # TEMP: also save as png for viewing pleasure
        mi.Bitmap(image).convert(
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True
        ).write(pngPathStr)

    ref_image = render_convergence(scene, scene_name, 0.003, True)
    rmse = dr.sqrt(dr.mean(dr.square(ref_image - image)))
    print(f'{method}:\nN:{N}, rmse:{rmse}')
    diffimg = dr.abs(ref_image - image)
    return diffimg


# img = render_mc(scene, scene.sensors()[0], 500000)
# img = mi.render(scene, integrator=pathtracer, spp=20)
# img = render_convergence(scene, 0.0001)
# img = render_scene("lmc", scene, N=10 * 1000000, use_cached=True, integrand_samples=100000,
#                 stepsize=0.01, large_mut_chance=0.001, 
#                 precond=True, beta=0.999, delta=0.001, 
#                 momentum=False, alpha=0.99,
#                 dimin_adapt=False, dimin_adapt_coeff_M=1e-9, dimin_adapt_coeff_m=1e-8,
#                 seed=0)

img = render_scene("pss", scene, N=10 * 1000000, use_cached=True, integrand_samples=100000,
                 stepsize=0.1, large_mut_chance=0.1)

plt.axis("off")
plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping TODO why this needed?

plt.show()

