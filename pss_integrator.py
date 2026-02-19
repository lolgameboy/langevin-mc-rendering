import drjit as dr
import mitsuba as mi

from pss_sampler import PssSampler

# prbi = mi.ad.integrators.prb_basic.BasicPRBIntegrator()
class Pss(mi.Integrator):
    def __init__(self):
        pass
    
    @dr.syntax
    def render(self, scene: mi.Scene, sensor: mi.Sensor, seed: mi.UInt = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> mi.TensorXf:
        film = sensor.film()
        resolution = film.crop_size()

        image_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            3
        )

        cam_transform = sensor.m_to_world
        rng = dr.rng(seed=mi.UInt32(0))

        # Instantiate a path tracer object
        integrator = mi.load_dict({
            'type': 'prb_basic',
            'max_depth': 8,
        })

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        # x_fov is a lie (or I am retarded somewhere else in my code)
        y_fov = mi.traverse(sensor)["x_fov"]
        plane_height = dr.tan(dr.deg2rad(y_fov)) # Originally divided y_fov by 2, but turns out it is already halved in spec?
        plane_size = mi.Vector2f(plane_height * resolution.x / resolution.y, plane_height)

        # Preprocessing step: calculate total integrand of image
        # TODO Maybe can make this faster by using a proper integrator and summing the final image
        N = mi.Int(10000)
        i = mi.Int(0)
        integrand = mi.Float(0)
        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))
            luminance, _, _, _ = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng)
            integrand += luminance / N
            i += 1
        image = mi.render(scene, spp=10, integrator=mi.load_dict({'type':'path', 'max_depth':8}))
        sum = dr.sum(image, axis=None) / (resolution.x * resolution.y)


        # Initial sample
        #sample = dr.full(mi.ArrayXf, 0.5, (sample_size,1))
        sample = rng.random(mi.ArrayXf, (sample_size,1))
        sample[30] += 1
        #sample += dr.full(mi.ArrayXf, 0.001, (sample_size,1))
        #grad = dr.full(mi.ArrayXf, 0.5, (sample_size, 1))

        dr.enable_grad(sample)
        luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng)
        dr.backward(luminance, dr.ADFlag.AllowNoGrad)
        dr.print(luminance)
        
        image_block.put(mi.Point2f(pixel_x, pixel_y), res / luminance * integrand)

        N = mi.Int(1000000)
        i = mi.Int(0)

        while i < N:
            # Generate proposal mutation
            #dr.forward(sample)
            #print(luminance.grad)
            mutation = rng.normal(mi.ArrayXf, (sample_size,1), scale=0.1)
            prop_sample = sample + mutation

            # Boundary conditions: loop around
            for j in range(len(prop_sample)):
                if prop_sample[j] < 0:
                    prop_sample[j] += 1
                if prop_sample[j] > 1:
                    prop_sample[j] -= 1

            # Calculate proposal luminance
            dr.enable_grad(prop_sample)
            with dr.resume_grad():
                prop_luminance, prop_res, prop_pixel_x, prop_pixel_y = calculate_sample_contribution(prop_sample, scene, cam_transform, plane_size, resolution, rng)
            dr.backward(prop_luminance, dr.ADFlag.AllowNoGrad)
            #dr.print(dr.grad(prop_sample))

            # Calculate acceptance chance            
            a = dr.min(1, prop_luminance / luminance)

            if rng.uniform(mi.Float, 1) < a:
                luminance, res, pixel_x, pixel_y = prop_luminance, prop_res, prop_pixel_x, prop_pixel_y
                sample = prop_sample

            # Explanation: PSSMLT actually uses luminance (combination of RGB values) to explore domain 
            # (because it only takes a scalar output) to mutate the sample)
            # Then it intuitively counts pixel visits to determine luminance distribution over the whole image
            # BUT! We use RGB channels. Thus, we must keep track of how this luminance is "divided" amongst the different colors.
            # So instead of counting "1" to the pixel, we count the fractions of colors, and they average to 1.
            # Also multiply with total integrand because MLT gives *proportional* distribution
            # Update: on second inspection this still seems correct, although it might be better to 
            # let the fractions of color *sum* to 1 instead of average to 1 TODO
            # I think this is connected to the total brightness, and it might influence the total brightness factor by 3
            # or possibly not i guess. check this.
            # Hmm, or maybe my reasoning here is completely flawed from the start, because (255, 0, 0) and (255, 255, 0)
            # can be equally bright but are simply different colors. Is that how it works?
            # or is light red more like 255 200 200? TODO
            image_block.put(mi.Point2f(pixel_x, pixel_y), res / luminance * integrand)
            i += 1

        # We have now simply counted all samples per bucket. 
        # Now we must appropriately devide the whole image to get a proper distribution that integrates to 1
        # We must devide by sample count and multiply by buckets (reasoning: 1D uniform integral, 2 buckets, 10 samples)
        # The / 2 is arbitrary. This determines overall brightness of image. 
        # TODO: the / 2 (or even / 5) shows that a lot of samples are taken on the light itself
        return image_block.tensor() / N * resolution.x * resolution.y


# Essentially the path tracing function
# TODO @dr.syntax ???
def calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng, max_depth = 6):
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
    L = mi.Spectrum(0.0)

    # ------------------------------------------------------------------
    # Path tracing loop
    # ------------------------------------------------------------------
    for depth in range(max_depth):
        si = scene.ray_intersect(ray, active)
        bsdf = si.bsdf()
        ctx = mi.BSDFContext()

        # Add emission (only on depth 0, see comment below)
        if depth == 0:
            emitter = si.emitter(scene)
            L += throughput * emitter.eval(si, active)

        # TODO Now what the hell is MIS and where does it come in?
        # If I understand it correctly, currently this should be wrong:
        # emitter = si.emitter(scene)
        # L += throughput * emitter.eval(si, active)
        # because I have two sources of radiance or something and thus im counting them double
        # And yes the image confirms this, way too bright walls
        # Now, for now I could just keep it like this and temp remove emission,
        # and only use light rays. This is perfectly fine, only change is that
        # the emitter is not directly visible.
        # UPDATE: Only adding emission on depth 0 seems to fix this issue.
        # Only question now is if this is theoretically correct, especially when adding
        # more light sources. That's a TODO for the future!

        # Add light ray
        # TODO light ray active checks
        ds, weight = scene.sample_emitter_direction(si, pss_sampler.next_2d())

        # ds.d is unit direction from si point to sampled point on emitter
        wo = si.to_local(ds.d)
        bsdf_val = bsdf.eval(ctx, si, wo, True)

        L += throughput * weight * bsdf_val
        # Volgens mij include weight al wel als lichtbron niet zichtbaar is, omdat we hier geen valid bool als return krijgen

        # Stop if no hit
        active &= si.is_valid()

        # TODO Doesnt work because active is symbolic or something
        #if not active:
        #    print("No hit!")
        #    break

        # BSDF sampling

        u_bsdf = pss_sampler.next_2d()
        u_component = pss_sampler.next_1d()

        bsdf_sample, bsdf_weight = bsdf.sample(
            ctx,
            si,
            u_component,
            u_bsdf,
            active
        )
        # Thought about this for a very long time, basically:
        # bsdf_weight includes division by p(x). But this is needed! 
        # Because of the reparametrisation of the pss. Basically, we need to calculate 
        # fhat(x) = f(x) / p(x) with f(x) the radiance along the path and p(x) the 
        # probability distribution of that path. If paths are chosen uniformely 
        # (i.e. if primary sample is converted to paths "uniformely") no division is necessary.
        # But! If a path is not chosen uniformely, but e.g. importance sampled for a bsdf like they
        # do here, we *do* need to divide by this probability of the path.
        # Sorry for vague explanation to future me woops. But look at paper, that might help.
        throughput *= bsdf_weight

        # Spawn next ray
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # Russian roulette
        rr_prob = 0.99

        rr_sample = rng.random(mi.Float, (1))
        survive = rr_sample < rr_prob
        throughput /= rr_prob

        active &= survive

    luminance = dr.sum(L)# / 3 # TODO This is where luminance weights come in, now I use equal weights
    return luminance, L, pixel_x, pixel_y




def calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution, rng, max_depth = 6):
    '''
    Reference method to calculate the ray contribution. Affected pixel is dependent on sample.
    Individual output not neccesarily equal to output of calculate_sample_contribution with same sample, 
    because a different path tracer is used (built-in vs own) but the final image should be the same if my path tracer is correct
    This method is intended to be used to test the monte-carlo or (maybe?) pss sampling code separately from my actual
    path tracing code (calculate_sample_contribution above).
    '''
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
    diffray = mi.RayDifferential3f(ray)

    integrator=mi.load_dict({'type':'path', 'max_depth':8})
    res = integrator.sample(scene, pss_sampler, diffray)
    luminance = dr.sum(res[0]) / 3 # TODO This is where luminance weights come in, now I use equal weights
    return luminance, res[0], pixel_x, pixel_y