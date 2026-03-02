import drjit as dr
import mitsuba as mi

from pss_sampler import PssSampler
from trace_path import calculate_sample_contribution
from utils import log_gaussian_diag

class LMC(mi.Integrator):
    def __init__(self):
        pass
    
    @dr.syntax
    def render(self, scene: mi.Scene, sensor: mi.Sensor, total_samples, pss = False, seed: mi.UInt = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> mi.TensorXf:
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
        rng = dr.rng(seed=mi.UInt32(3))

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        # x_fov is a lie (or I am retarded somewhere else in my code)
        y_fov = mi.traverse(sensor)["x_fov"]
        plane_height = 2 * dr.tan(dr.deg2rad(y_fov / 2))
        plane_size = mi.Vector2f(plane_height * resolution.x / resolution.y, plane_height)

        # Preprocessing step: calculate total integrand of image
        N = mi.Int(100000)
        i = mi.Int(0)
        integrand = mi.Float(0)
        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))
            luminance, _, _, _ = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng)
            integrand += luminance / N
            i += 1

        sample = rng.random(mi.ArrayXf, (sample_size,1))

        dr.enable_grad(sample)
        luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng)
        dr.backward(dr.log(dr.maximum(luminance, 1e-8)), dr.ADFlag.AllowNoGrad)
        gradlog = dr.grad(sample)
        
        luminance_block.put(mi.Point2f(pixel_x, pixel_y), [luminance])
        color_block.put(mi.Point2f(pixel_x, pixel_y), res)

        N = mi.Int(total_samples)
        i = mi.Int(0)

        avgaccept = mi.Float(0)

        while i < N:
            # Generate proposal mutation
            # Large or small step mutation?
            large_mut_chance = 0.1
            # TODO When large mutation is rejected, should I attempt another large mut or just any mut specifically?
            # TODO Optionally can disable gradient comp when taking large step for performance, if I want
            # NVM, need that gradient for next mutation
            
            stepsize = 0.01

            large_mut = rng.random(mi.Float, (1)) < large_mut_chance
            if large_mut:
                prop_sample = rng.random(mi.ArrayXf, (sample_size, 1))
            else:
                w = rng.normal(mi.ArrayXf, (sample_size,1), scale=dr.sqrt(stepsize))
                mutation = 0.5 * stepsize * gradlog + w
                prop_sample = sample + mutation

            # For PSS:
            if pss:
                mutation = rng.normal(mi.ArrayXf, (sample_size,1), scale=0.1)
                prop_sample = sample + mutation

            # M-H chain lives in R^n, but sample evaluation happens in unit cube.
            # This is to fix acceptance chance incorrectness when wrapping chain.
            # prop_sample_eval is passed to the path tracer but prop_sample is used in acceptance etc
            prop_sample_eval = prop_sample - dr.floor(prop_sample)

            # Calculate proposal luminance
            dr.enable_grad(prop_sample_eval)
            prop_luminance, prop_res, prop_pixel_x, prop_pixel_y = calculate_sample_contribution(prop_sample_eval, scene, cam_transform, plane_size, resolution, rng)
            # dr.backward(dr.log(dr.maximum(prop_luminance, 1e-8)), dr.ADFlag.AllowNoGrad)
            dr.backward(dr.log(prop_luminance + 1e-3), dr.ADFlag.AllowNoGrad)
            prop_gradlog = dr.grad(prop_sample_eval)

            # Calculate acceptance chance  
            # log to avoid underflow
            if large_mut:
                a = dr.minimum(1, prop_luminance / luminance)
            else:
                covar = dr.full(mi.ArrayXf, stepsize, (sample_size, 1))
                log_alpha = dr.log(dr.maximum(prop_luminance, 1e-8)) \
                            + log_gaussian_diag(sample, prop_sample + 0.5 * stepsize * prop_gradlog, covar) \
                            - dr.log(dr.maximum(luminance, 1e-8)) \
                            - log_gaussian_diag(prop_sample, sample + 0.5 * stepsize * gradlog, covar)
                a = dr.exp(dr.minimum(0.0, log_alpha))
                # For PSS:
                if pss:
                    a = dr.minimum(1, prop_luminance / luminance)


            avgaccept += a

            if rng.uniform(mi.Float, 1) < a:
                luminance, res, pixel_x, pixel_y = prop_luminance, prop_res, prop_pixel_x, prop_pixel_y
                sample = prop_sample
                gradlog = prop_gradlog

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
            luminance_block.put(mi.Point2f(pixel_x, pixel_y), [luminance])
            color_block.put(mi.Point2f(pixel_x, pixel_y), res / (luminance) * integrand)
            i += 1
        dr.print(f'Average acceptance rate: {avgaccept/N}')
        # We have now simply counted all samples per bucket. 
        # Now we must appropriately devide the whole image to get a proper distribution that integrates to 1
        # We must devide by sample count and multiply by buckets (reasoning: 1D uniform integral, 2 buckets, 10 samples)
        luminance_tensor = luminance_block.tensor() / N * resolution.x * resolution.y

        # Rescale colors to correct luminance
        weight_tensor = dr.reshape(dr.max(color_block.tensor(), -1), (resolution.x, resolution.y, 1))
        # Mask pixels with 0 samples to result in black (0)
        result = dr.select(luminance_tensor != 0, luminance_tensor / weight_tensor, 0)
        # COOL TO KNOW: Just returning luminance tensor gives nice sample count map. Could be useful to illustrate in thesis?
        #return color_block.tensor() * result
        return color_block.tensor() / N * resolution.x * resolution.y #TEMP