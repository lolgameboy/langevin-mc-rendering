import drjit as dr
import mitsuba as mi

from pss_sampler import PssSampler
from trace_path import calculate_sample_contribution, calculate_sample_contribution_ref
from utils import log_gaussian_diag

class LMC(mi.Integrator):
    def __init__(self):
        pass
    #def render(self, scene: mi.Scene, sensor: mi.Sensor, seed: mi.UInt = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> mi.TensorXf:
    @dr.syntax
    def render(self,
                scene: mi.Scene,
                sensor: mi.Sensor,
                total_samples,
                integrand_samples = 100000,
                pss = False,
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
                seed = 0) -> mi.TensorXf:
        
        film = sensor.film()
        resolution = film.crop_size()

        luminance_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            1
        )

        color_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            3
        )

        cam_transform = sensor.m_to_world
        rng = dr.rng(seed=mi.UInt32(seed))

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        x_fov = mi.traverse(sensor)["x_fov"]
        plane_width = 2 * dr.tan(dr.deg2rad(x_fov / 2))
        plane_size = mi.Vector2f(plane_width, plane_width * resolution.y / resolution.x)

        # Preprocessing step: calculate total integrand of image
        N = mi.Int(integrand_samples)
        i = mi.Int(0)
        integrand = mi.Float(0)

        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))
            luminance, _, _, _ = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution)
            integrand += luminance / N
            i += 1

        sample = rng.random(mi.ArrayXf, (sample_size,1))

        dr.enable_grad(sample)
        luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution)
        dr.backward(dr.log(dr.maximum(luminance, 1e-8)), dr.ADFlag.AllowNoGrad)
        gradlog = dr.grad(sample)
        
        luminance_block.put(mi.Point2f(pixel_x, pixel_y), [luminance])
        color_block.put(mi.Point2f(pixel_x, pixel_y), res)

        N = mi.Int(total_samples)
        i = mi.Int(0)

        avgaccept = mi.Float(0)

        prev_G = dr.ones(mi.ArrayXf, (sample_size, 1))
        G = dr.ones(mi.ArrayXf, (sample_size, 1))
        M = dr.ones(mi.ArrayXf, (sample_size, 1))

        prev_d = dr.zeros(mi.ArrayXf, (sample_size, 1))
        d = dr.zeros(mi.ArrayXf, (sample_size, 1))
        m = dr.zeros(mi.ArrayXf, (sample_size, 1))
        
        while i < N:
            # Generate proposal mutation
            # Large or small step mutation?

            # TODO When large mutation is rejected, should I attempt another large mut or just any mut specifically?
            # TODO Good value? Should this depend on sample count?? Is this only important for convergence as
            # small bias for low sample count is not important??? Because outshadowed by error?
            diminAdaptCoeffM = mi.Float(dimin_adapt_coeff_M)
            diminAdaptCoeffm = mi.Float(dimin_adapt_coeff_m)
            if not dimin_adapt:
                diminAdaptCoeffM = mi.Float(0)
                diminAdaptCoeffm = mi.Float(0)
            # TODO What to do with M on large mutation? Reset? Keep? Check paper or do experiments?
            large_mut = rng.random(mi.Float, (1)) < large_mut_chance
            if large_mut:
                prop_sample = rng.random(mi.ArrayXf, (sample_size, 1))
            else:
                # Calculate preconditioning matrix (diagonal!)
                if precond:
                    adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffM)
                    G = beta * prev_G + (1 - beta) * (gradlog * gradlog)
                    M = 1 / (delta * dr.identity(mi.ArrayXf, (sample_size, 1)) + adapt_coeff * dr.sqrt(G))
                else:
                    M = dr.ones(mi.ArrayXf, (sample_size, 1))
                # Calculate momentum
                if momentum:
                    adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffm)
                    d = alpha * prev_d + (1 - alpha) * gradlog
                    m = gradlog + adapt_coeff * d
                else:
                    m = gradlog
                w = rng.normal(mi.ArrayXf, (sample_size,1), scale=dr.sqrt(stepsize))
                mutation = 0.5 * stepsize * M * m + dr.sqrt(M)*w
                prop_sample = sample + mutation

                # For PSS:
                if pss:
                    mutation = rng.normal(mi.ArrayXf, (sample_size,1), scale=dr.sqrt(stepsize))
                    prop_sample = sample + mutation

            # M-H chain lives in R^n, but sample evaluation happens in unit cube.
            # This is to fix acceptance chance incorrectness when wrapping chain.
            # prop_sample_eval is passed to the path tracer but prop_sample is used in acceptance etc
            #prop_sample = prop_sample - dr.floor(prop_sample)
            #dr.enable_grad(prop_sample)
            prop_sample_eval = prop_sample - dr.floor(prop_sample)
            dr.enable_grad(prop_sample_eval)
            # Calculate proposal luminance
            prop_luminance, prop_res, prop_pixel_x, prop_pixel_y = calculate_sample_contribution(prop_sample_eval, scene, cam_transform, plane_size, resolution)
            # dr.backward(dr.log(dr.maximum(prop_luminance, 1e-8)), dr.ADFlag.AllowNoGrad)
            # TODO Experiment: dr.log(prop_luminance + 1e-3) seems to perform slightly better than 
            # dr.log(dr.maximum(prop_luminance, 1e-8))
            dr.backward(dr.log(dr.maximum(prop_luminance, 1e-8)), dr.ADFlag.AllowNoGrad)
            prop_gradlog = dr.grad(prop_sample_eval)
            # Truncated gradients
            prop_gradlog = dr.minimum(prop_gradlog, 100)

            # calculate proposal preconditioning
            if precond:
                adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffM)
                G_prop = beta * prev_G + (1 - beta) * (prop_gradlog * prop_gradlog)
                M_prop = 1 / (delta * dr.identity(mi.ArrayXf, (sample_size, 1)) + adapt_coeff * dr.sqrt(G_prop))
            else:
                M_prop = dr.ones(mi.ArrayXf, (sample_size, 1))

            # calculate proposal momentum
            if momentum:
                adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffm)

                d_prop = alpha * prev_d + (1 - alpha) * prop_gradlog
                m_prop = prop_gradlog + adapt_coeff * d_prop
            else:
                m_prop = prop_gradlog


            # Calculate acceptance chance  
            # log to avoid underflow
            if large_mut:
                #a = mi.Float(1)
                a = dr.minimum(1, prop_luminance / luminance)
            else:
                covar = dr.full(mi.ArrayXf, stepsize, (sample_size, 1))
                # TODO acceptance chance formula for adaptation? Does it matter much? It is wrong anyway because not
                # time-homogeneous, and so because of diminishing adaptation M and m influence will vanish anyway
                # But there probably is a "best approximation". What is this? 
                # Currently extract d comp (actual momentum factor) from m for reverse probability and add prop_gradlog instead of gradlog
                log_alpha = dr.log(dr.maximum(prop_luminance, 1e-8)) \
                            + log_gaussian_diag(sample, prop_sample + 0.5 * stepsize * M_prop * m_prop, covar * M_prop) \
                            - dr.log(dr.maximum(luminance, 1e-8)) \
                            - log_gaussian_diag(prop_sample, sample + 0.5 * stepsize * M * m, covar * M)
                a = dr.exp(dr.minimum(0.0, log_alpha))
                # For PSS:
                if pss:
                    a = dr.minimum(1, prop_luminance / luminance)
            avgaccept += a

            # PSS weights via kelemen (NOT YET CORRECT!)
            # contrib = res / luminance / integrand
            # prop_contrib = prop_res / prop_luminance / integrand
            # prop_w1 = (a * prop_contrib) / (prop_contrib + large_mut_chance)
            # w1 = ((1-a) * contrib) / (contrib + large_mut_chance)
            # w2 = large_mut_chance / (contrib + large_mut_chance)
            # prop_w2 = large_mut_chance / (prop_contrib + large_mut_chance)
            # F = res / luminance
            # prop_F = prop_res / prop_luminance
            # if large_mut:
            #     color_block.put(mi.Point2f(pixel_x, pixel_y), w1 * F / contrib + w2 * F / contrib)
            #     color_block.put(mi.Point2f(prop_pixel_x, prop_pixel_y), prop_w1 * prop_F / prop_contrib + prop_w2 * prop_F / prop_contrib)
            # else:
            #     color_block.put(mi.Point2f(pixel_x, pixel_y), w1 * F / contrib)
            #     color_block.put(mi.Point2f(prop_pixel_x, prop_pixel_y), prop_w1 * prop_F / prop_contrib)
            
            # For now:
            color_block.put(mi.Point2f(pixel_x, pixel_y), (1-a) * res / (luminance) * integrand)
            color_block.put(mi.Point2f(prop_pixel_x, prop_pixel_y), a * prop_res / (prop_luminance) * integrand)

            if rng.uniform(mi.Float, 1) < a:
                luminance, res, pixel_x, pixel_y = prop_luminance, prop_res, prop_pixel_x, prop_pixel_y
                sample = prop_sample
                gradlog = prop_gradlog
                prev_d = d
                prev_G = G
                # TODO Experiment two options: reset and low global step chance, or 
                # no reset and higher global step chance
                # What with momentum? Reset makes intuitive sense, but dont think they reset in paper
                if large_mut:
                    #prev_G = dr.ones(mi.ArrayXf, (sample_size, 1))
                    prev_d = dr.zeros(mi.ArrayXf, (sample_size, 1))

            # Explanation: PSSMLT actually uses luminance (combination of RGB values) to explore domain 
            # (because it only takes a scalar output) to mutate the sample)
            # Then it intuitively counts pixel visits to determine luminance distribution over the whole image
            # BUT! We use RGB channels. Thus, we must keep track of how this luminance is "divided" amongst the different colors.
            # So instead of counting "1" to the pixel, we count the fractions of colors, and they average to 1.
            # Also multiply with total integrand because MLT gives *proportional* distribution
            # Update: on second inspection this still seems correct, although it might be better to 
            # let the fractions of color *sum* to 1 instead of average to 1 
            # I think this is connected to the total brightness, and it might influence the total brightness factor by 3
            # or possibly not i guess. check this.
            # Hmm, or maybe my reasoning here is completely flawed from the start, because (255, 0, 0) and (255, 255, 0)
            # can be equally bright but are simply different colors. Is that how it works?
            # or is light red more like 255 200 200? 
            # ALRIGHT: New strategy: Store luminance and color separately. 
            # Add color with correct "luminance weight" to other colors 
            # (rn just adding is fine because lum = max rgb so weigt is correct automatically)
            # Then at the end you have correct *relative* rgb values. So just rescale to correct luminance
            
            # splatting to block now done for both tentative and current sample, weighted (above)
            #luminance_block.put(mi.Point2f(pixel_x, pixel_y), [luminance])
            #color_block.put(mi.Point2f(pixel_x, pixel_y), res / (luminance) * integrand)
            i += 1
        # We have now simply counted all samples per bucket. 
        # Now we must appropriately devide the whole image to get a proper distribution that integrates to 1
        # We must devide by sample count and multiply by buckets (reasoning: 1D uniform integral, 2 buckets, 10 samples)
        luminance_tensor = luminance_block.tensor() / N * resolution.x * resolution.y

        # Rescale colors to correct luminance
        weight_tensor = dr.reshape(dr.max(color_block.tensor(), -1), (resolution.y, resolution.x, 1))
        # Mask pixels with 0 samples to result in black (0)
        result = dr.select(luminance_tensor != 0, luminance_tensor / weight_tensor, 0)
        # COOL TO KNOW: Just returning luminance tensor gives nice sample count map. Could be useful to illustrate in thesis?
        #return color_block.tensor() * result
        #TEMP
        return color_block.tensor() / N * resolution.x * resolution.y, avgaccept[0] / N
    












    @dr.syntax
    def render_ref(self,
                scene: mi.Scene,
                sensor: mi.Sensor,
                total_samples,
                integrand_samples = 100000,
                pss = False,
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
                seed = 0) -> mi.TensorXf:
        
        film = sensor.film()
        resolution = film.crop_size()

        luminance_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            1
        )

        color_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            3
        )

        cam_transform = sensor.m_to_world
        rng = dr.rng(seed=mi.UInt32(seed))

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        x_fov = mi.traverse(sensor)["x_fov"]
        plane_width = 2 * dr.tan(dr.deg2rad(x_fov / 2))
        plane_size = mi.Vector2f(plane_width, plane_width * resolution.y / resolution.x)

        # Preprocessing step: calculate total integrand of image
        N = mi.Int(integrand_samples)
        i = mi.Int(0)
        integrand = mi.Float(0)

        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))
            luminance, _, _, _ = calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution)
            integrand += luminance / N
            i += 1

        sample = rng.random(mi.ArrayXf, (sample_size,1))

        luminance, res, pixel_x, pixel_y = calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution)
        # Approx gradient
        eps = 1e-8
        gradlog = dr.zeros(mi.ArrayXf, (sample_size, 1))
        for sam_i in range(sample_size):
            sample[sam_i] += eps
            lum2, _, _, _ = calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution)
            gradlog[sam_i] = (- dr.log(dr.maximum(lum2, 1e-8)) + dr.log(dr.maximum(luminance, 1e-8))) / eps
            sample[sam_i] -= eps
        
        luminance_block.put(mi.Point2f(pixel_x, pixel_y), [luminance])
        color_block.put(mi.Point2f(pixel_x, pixel_y), res)

        N = mi.Int(total_samples)
        i = mi.Int(0)

        avgaccept = mi.Float(0)

        prev_G = dr.ones(mi.ArrayXf, (sample_size, 1))
        G = dr.ones(mi.ArrayXf, (sample_size, 1))
        M = dr.ones(mi.ArrayXf, (sample_size, 1))

        prev_d = dr.zeros(mi.ArrayXf, (sample_size, 1))
        d = dr.zeros(mi.ArrayXf, (sample_size, 1))
        m = dr.zeros(mi.ArrayXf, (sample_size, 1))

        prop_gradlog = dr.zeros(mi.ArrayXf, (sample_size, 1))
        
        while i < N:
            # Generate proposal mutation
            # Large or small step mutation?

            # TODO When large mutation is rejected, should I attempt another large mut or just any mut specifically?
            # TODO Good value? Should this depend on sample count?? Is this only important for convergence as
            # small bias for low sample count is not important??? Because outshadowed by error?
            diminAdaptCoeffM = mi.Float(dimin_adapt_coeff_M)
            diminAdaptCoeffm = mi.Float(dimin_adapt_coeff_m)
            if not dimin_adapt:
                diminAdaptCoeffM = mi.Float(0)
                diminAdaptCoeffm = mi.Float(0)
            # TODO What to do with M on large mutation? Reset? Keep? Check paper or do experiments?
            large_mut = rng.random(mi.Float, (1)) < large_mut_chance
            if large_mut:
                prop_sample = rng.random(mi.ArrayXf, (sample_size, 1))
            else:
                # Calculate preconditioning matrix (diagonal!)
                if precond:
                    adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffM)
                    G = beta * prev_G + (1 - beta) * (gradlog * gradlog)
                    M = 1 / (delta * dr.identity(mi.ArrayXf, (sample_size, 1)) + adapt_coeff * dr.sqrt(G))
                else:
                    M = dr.ones(mi.ArrayXf, (sample_size, 1))
                # Calculate momentum
                if momentum:
                    adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffm)
                    d = alpha * prev_d + (1 - alpha) * gradlog
                    m = gradlog + adapt_coeff * d
                else:
                    m = gradlog
                w = rng.normal(mi.ArrayXf, (sample_size,1), scale=dr.sqrt(stepsize))
                mutation = 0.5 * stepsize * M * m + dr.sqrt(M)*w
                prop_sample = sample + mutation

                # For PSS:
                if pss:
                    mutation = rng.normal(mi.ArrayXf, (sample_size,1), scale=dr.sqrt(stepsize))
                    prop_sample = sample + mutation

            # M-H chain lives in R^n, but sample evaluation happens in unit cube.
            # This is to fix acceptance chance incorrectness when wrapping chain.
            # prop_sample_eval is passed to the path tracer but prop_sample is used in acceptance etc
            prop_sample_eval = prop_sample - dr.floor(prop_sample)
            # Calculate proposal luminance
            prop_luminance, prop_res, prop_pixel_x, prop_pixel_y = calculate_sample_contribution(prop_sample_eval, scene, cam_transform, plane_size, resolution)
            for sam_i in range(sample_size):
                prop_sample_eval[sam_i] += eps
                lum2, _, _, _ = calculate_sample_contribution_ref(prop_sample_eval, scene, cam_transform, plane_size, resolution)
                prop_gradlog[sam_i] = (- dr.log(dr.maximum(lum2, 1e-8)) + dr.log(dr.maximum(prop_luminance, 1e-8))) / eps
                prop_sample_eval[sam_i] -= eps

            # Truncated gradients
            prop_gradlog = dr.minimum(prop_gradlog, 100)

            # calculate proposal preconditioning
            if precond:
                adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffM)
                G_prop = beta * prev_G + (1 - beta) * (prop_gradlog * prop_gradlog)
                M_prop = 1 / (delta * dr.identity(mi.ArrayXf, (sample_size, 1)) + adapt_coeff * dr.sqrt(G_prop))
            else:
                M_prop = dr.ones(mi.ArrayXf, (sample_size, 1))

            # calculate proposal momentum
            if momentum:
                adapt_coeff = dr.power(mi.Float(i+1), -diminAdaptCoeffm)

                d_prop = alpha * prev_d + (1 - alpha) * prop_gradlog
                m_prop = prop_gradlog + adapt_coeff * d_prop
            else:
                m_prop = prop_gradlog


            # Calculate acceptance chance  
            # log to avoid underflow
            if large_mut:
                #a = mi.Float(1)
                a = dr.minimum(1, prop_luminance / luminance)
            else:
                covar = dr.full(mi.ArrayXf, stepsize, (sample_size, 1))
                # TODO acceptance chance formula for adaptation? Does it matter much? It is wrong anyway because not
                # time-homogeneous, and so because of diminishing adaptation M and m influence will vanish anyway
                # But there probably is a "best approximation". What is this? 
                # Currently extract d comp (actual momentum factor) from m for reverse probability and add prop_gradlog instead of gradlog
                log_alpha = dr.log(dr.maximum(prop_luminance, 1e-8)) \
                            + log_gaussian_diag(sample, prop_sample + 0.5 * stepsize * M_prop * m_prop, covar * M_prop) \
                            - dr.log(dr.maximum(luminance, 1e-8)) \
                            - log_gaussian_diag(prop_sample, sample + 0.5 * stepsize * M * m, covar * M)
                a = dr.exp(dr.minimum(0.0, log_alpha))
                # For PSS:
                if pss:
                    a = dr.minimum(1, prop_luminance / luminance)
            avgaccept += a

            # PSS weights via kelemen (NOT YET CORRECT!)
            # contrib = res / luminance / integrand
            # prop_contrib = prop_res / prop_luminance / integrand
            # prop_w1 = (a * prop_contrib) / (prop_contrib + large_mut_chance)
            # w1 = ((1-a) * contrib) / (contrib + large_mut_chance)
            # w2 = large_mut_chance / (contrib + large_mut_chance)
            # prop_w2 = large_mut_chance / (prop_contrib + large_mut_chance)
            # F = res / luminance
            # prop_F = prop_res / prop_luminance
            # if large_mut:
            #     color_block.put(mi.Point2f(pixel_x, pixel_y), w1 * F / contrib + w2 * F / contrib)
            #     color_block.put(mi.Point2f(prop_pixel_x, prop_pixel_y), prop_w1 * prop_F / prop_contrib + prop_w2 * prop_F / prop_contrib)
            # else:
            #     color_block.put(mi.Point2f(pixel_x, pixel_y), w1 * F / contrib)
            #     color_block.put(mi.Point2f(prop_pixel_x, prop_pixel_y), prop_w1 * prop_F / prop_contrib)
            
            # For now:
            color_block.put(mi.Point2f(pixel_x, pixel_y), (1-a) * res / (luminance) * integrand)
            color_block.put(mi.Point2f(prop_pixel_x, prop_pixel_y), a * prop_res / (prop_luminance) * integrand)

            if rng.uniform(mi.Float, 1) < a:
                luminance, res, pixel_x, pixel_y = prop_luminance, prop_res, prop_pixel_x, prop_pixel_y
                sample = prop_sample
                gradlog = prop_gradlog
                prev_d = d
                prev_G = G
                # TODO Experiment two options: reset and low global step chance, or 
                # no reset and higher global step chance
                # What with momentum? Reset makes intuitive sense, but dont think they reset in paper
                if large_mut:
                    #prev_G = dr.ones(mi.ArrayXf, (sample_size, 1))
                    prev_d = dr.zeros(mi.ArrayXf, (sample_size, 1))

            # Explanation: PSSMLT actually uses luminance (combination of RGB values) to explore domain 
            # (because it only takes a scalar output) to mutate the sample)
            # Then it intuitively counts pixel visits to determine luminance distribution over the whole image
            # BUT! We use RGB channels. Thus, we must keep track of how this luminance is "divided" amongst the different colors.
            # So instead of counting "1" to the pixel, we count the fractions of colors, and they average to 1.
            # Also multiply with total integrand because MLT gives *proportional* distribution
            # Update: on second inspection this still seems correct, although it might be better to 
            # let the fractions of color *sum* to 1 instead of average to 1 
            # I think this is connected to the total brightness, and it might influence the total brightness factor by 3
            # or possibly not i guess. check this.
            # Hmm, or maybe my reasoning here is completely flawed from the start, because (255, 0, 0) and (255, 255, 0)
            # can be equally bright but are simply different colors. Is that how it works?
            # or is light red more like 255 200 200? 
            # ALRIGHT: New strategy: Store luminance and color separately. 
            # Add color with correct "luminance weight" to other colors 
            # (rn just adding is fine because lum = max rgb so weigt is correct automatically)
            # Then at the end you have correct *relative* rgb values. So just rescale to correct luminance
            
            # splatting to block now done for both tentative and current sample, weighted (above)
            #luminance_block.put(mi.Point2f(pixel_x, pixel_y), [luminance])
            #color_block.put(mi.Point2f(pixel_x, pixel_y), res / (luminance) * integrand)
            i += 1
        # We have now simply counted all samples per bucket. 
        # Now we must appropriately devide the whole image to get a proper distribution that integrates to 1
        # We must devide by sample count and multiply by buckets (reasoning: 1D uniform integral, 2 buckets, 10 samples)
        luminance_tensor = luminance_block.tensor() / N * resolution.x * resolution.y

        # Rescale colors to correct luminance
        weight_tensor = dr.reshape(dr.max(color_block.tensor(), -1), (resolution.y, resolution.x, 1))
        # Mask pixels with 0 samples to result in black (0)
        result = dr.select(luminance_tensor != 0, luminance_tensor / weight_tensor, 0)
        # COOL TO KNOW: Just returning luminance tensor gives nice sample count map. Could be useful to illustrate in thesis?
        #return color_block.tensor() * result
        #TEMP
        return color_block.tensor() / N * resolution.x * resolution.y, avgaccept[0] / N