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

def render_scene(method, scene, N, use_cached = True):
    if method == "lmc":
        pathStr = f'cache/{scene_name}/lmc_{N}.exr'
        pngPathStr = f'cache/{scene_name}/lmc_{N}.png'
        def render_function():
            lmc = LMC()
            return lmc.render(scene, scene.sensors()[0], N, False)

    if method == "pss":
        pathStr = f'cache/{scene_name}/pss_{N}.exr'
        pngPathStr = f'cache/{scene_name}/pss_{N}.png'
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
img = render_scene("pss", scene, 3 * 1000000, False)

plt.axis("off")
plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping TODO why this needed?

plt.show()

