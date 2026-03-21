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

def render_scene(method, scene, scene_name, ref_rmsediff, 
                N, use_cached = True, integrand_samples = 100000,
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
    

        def render_function():
            lmc = LMC()
            return lmc.render(scene, scene.sensors()[0], N, integrand_samples, False, 
                              stepsize, large_mut_chance, 
                              precond, beta, delta, 
                              momentum, alpha, 
                              dimin_adapt, dimin_adapt_coeff_M, dimin_adapt_coeff_m, 
                              seed)
    
    if method == "lmc_ref":
        fileName = f'lmc_ref_{N}_{integrand_samples}_{stepsize}_{large_mut_chance}'
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
    

        def render_function():
            lmc = LMC()
            return lmc.render_ref(scene, scene.sensors()[0], N, integrand_samples, False, 
                              stepsize, large_mut_chance, 
                              precond, beta, delta, 
                              momentum, alpha, 
                              dimin_adapt, dimin_adapt_coeff_M, dimin_adapt_coeff_m, 
                              seed)

    if method == "pss":
        fileName = f'pss_{N}_{integrand_samples}_{stepsize}_{large_mut_chance}'
        def render_function():
            lmc = LMC()
            return lmc.render(scene, scene.sensors()[0], N, integrand_samples, True, stepsize, large_mut_chance)

    if method == "mc":
        fileName = f'mc_{N}'
        def render_function():
            return render_mc(scene, scene.sensors()[0], N, True), 0
    
    pathStr = f'cache/{scene_name}/{fileName}.exr'
    pngPathStr = f'cache/{scene_name}/{fileName}.png'
    acceptRatioPathStr = f'cache/{scene_name}/{fileName}.acceptratio'
    path = Path(pathStr)
    if path.exists() and use_cached:
        bmp = mi.Bitmap(pathStr)
        image = mi.TensorXf(bmp)
        with open(acceptRatioPathStr, "r") as f:
            acceptratio = f.read()
    else:
        image, acceptratio = render_function()
        mi.Bitmap(image).write(pathStr)
        # TEMP: also save as png for viewing pleasure
        mi.Bitmap(image).convert(
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=True
        ).write(pngPathStr)
        # Save accept ratio
        with open(acceptRatioPathStr, "w") as f:
            f.write(str(acceptratio))


    ref_image = render_convergence(scene, scene_name, ref_rmsediff, True)
    rmse = dr.sqrt(dr.mean(dr.square(ref_image - image)))
    print(f'{method}:\nN:{N}, rmse:{rmse}, accept ratio:{acceptratio}')
    diffimg = dr.abs(ref_image - image)
    return image, rmse, acceptratio, diffimg

def plot_convergence(methods, scene, scene_name, ref_rmsediff, 
                use_cached = True, 
                l_samples = [1, 3, 10],
                l_integrand_samples = [100000],
                l_stepsize = [0.01],
                l_large_mut_chance = [0.0001],
                l_precond = [True],
                l_beta = [0.999],
                l_delta = [0.001],
                l_momentum = [True],
                l_alpha = [0.9],
                l_dimin_adapt = [True],
                l_dimin_adapt_coeff_M = [0.000000001],
                l_dimin_adapt_coeff_m = [0.00000001],
                seed = 0):
    
    methodsHad = []
    for method in methods:
        for integrand_samples in l_integrand_samples:
            for stepsize in l_stepsize:
                for large_mut_chance in l_large_mut_chance:
                    for precond in l_precond:
                        for beta in l_beta:
                            for delta in l_delta:
                                for momentum in l_momentum:
                                    for alpha in l_alpha:
                                        for dimin_adapt in l_dimin_adapt:
                                            for dimin_adapt_coeff_M in l_dimin_adapt_coeff_M:
                                                for dimin_adapt_coeff_m in l_dimin_adapt_coeff_m:
                                                    if method == "lmc":
                                                        fullMethodName = f'lmc_{integrand_samples}_{stepsize}_{large_mut_chance}'
                                                        if precond:
                                                            fullMethodName += f'_pre_{beta}_{delta}'
                                                        if momentum:
                                                            fullMethodName += f'_mom_{alpha}'
                                                        if dimin_adapt and (precond or momentum):
                                                            fullMethodName += f'_da'
                                                            if precond:
                                                                fullMethodName += f'_{dimin_adapt_coeff_M}'
                                                            if momentum:
                                                                fullMethodName += f'_{dimin_adapt_coeff_m}'
                                                    elif method == "pss":
                                                        fullMethodName = f'pss_{integrand_samples}_{stepsize}_{large_mut_chance}'
                                                    elif method == "mc":
                                                        fullMethodName = f'mc'
                                                    if fullMethodName not in methodsHad:
                                                        rmses = []
                                                        for i in l_samples:
                                                            img, rmse, acceptratio, diffimage = render_scene(method, scene, scene_name, 
                                                                    ref_rmsediff, N=i * 1000000,
                                                                    use_cached=use_cached, integrand_samples=integrand_samples,
                                                                    stepsize=stepsize, large_mut_chance=large_mut_chance, 
                                                                    precond=precond, beta=beta, delta=delta, 
                                                                    momentum=momentum, alpha=alpha,
                                                                    dimin_adapt=dimin_adapt, dimin_adapt_coeff_M=dimin_adapt_coeff_M, 
                                                                    dimin_adapt_coeff_m=dimin_adapt_coeff_m,
                                                                    seed=seed)
                                                            rmses.append(float(rmse.array[0]))
                                                        methodsHad.append(fullMethodName)
                                                        plt.loglog(l_samples, rmses, label=fullMethodName)
    plt.legend()
    plt.show()






#scene = mi.load_file("../scenes/scene.xml")
#scene_name = "cornell_box"
scene = mi.load_file("../scenes/veach-ajar/scene.xml")
scene_name = "veach_door"
#scene = mi.load_file("../scenes/veach-bidir/scene.xml")
#scene_name = "veach_egg"

# plot_convergence(["mc", "lmc"], scene, scene_name, 0.5, use_cached=True,               
#                 l_samples = [1, 3, 10, 30, 100, 300, 1000],# , 3000, 10000],
#                 l_integrand_samples = [100000, 10000000],
#                 l_stepsize = [0.01],
#                 l_large_mut_chance = [0.0001],
#                 l_precond = [True],
#                 l_beta = [0.999],
#                 l_delta = [0.001],
#                 l_momentum = [False],
#                 l_alpha = [0.9],
#                 l_dimin_adapt = [False],
#                 l_dimin_adapt_coeff_M = [0.000000001],
#                 l_dimin_adapt_coeff_m = [0.00000001],
#                 seed = 0)

# img = render_mc(scene, scene.sensors()[0], 500000)
# img = mi.render(scene, integrator=pathtracer, spp=20)
# img = render_convergence(scene, 0.0001)
img, rmse, acceptratio, diffimg = render_scene("lmc_ref", scene, N=10 * 1000000, scene_name=scene_name, 
                ref_rmsediff=1, use_cached=False, integrand_samples=300000,
                stepsize=0.001, large_mut_chance=0.02, 
                precond=True, beta=0.999, delta=0.001, 
                momentum=False, alpha=0.9,
                dimin_adapt=False, dimin_adapt_coeff_M=1e-9, dimin_adapt_coeff_m=1e-8,
                seed=0)
#img = render_convergence(scene, scene_name, 0.5, True)

# _, rmse, img = render_scene("pss", scene, N=10 * 1000000, use_cached=True, integrand_samples=100000,
#                  stepsize=0.1, large_mut_chance=0.001)

plt.axis("off")
plt.imshow(img ** (1.0 / 2.2)); # approximate sRGB tonemapping TODO why this needed?

plt.figure()
plt.imshow(diffimg ** (1/2.2))
plt.show()

