import drjit as dr
import mitsuba as mi
from pss_sampler import PssSampler


# Essentially the path tracing function
@dr.syntax
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

        # TODO TEMP Disabled
        # Russian roulette
        #rr_prob = 1
        # TODO I think this isn't correct for pss because f(x) is now not a correct function of x (the sample) because it 
        # does not deterministically depend on x
        #rr_sample = rng.random(mi.Float, (1))
        #survive = rr_sample < rr_prob
        #throughput /= rr_prob

        #active &= survive

    luminance = mi.luminance(L) # TODO This is where luminance weights come in
    # luminance = dr.sum(L) / 3
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

    # WORKING code to calc ray using built-in code. For now appears the same,
    # not sure if there are any differences? Shouldn't be, i think?
    #pixel = mi.Point2f(pixel_x, pixel_y)
    #film_size = mi.Point2f(512, 512)

    #ray, ray_weight = scene.sensors()[0].sample_ray_differential(
    #    time=0,
    #    sample1=0,
    #    sample2=pixel / film_size,
    #    sample3=mi.Point2f(0.5, 0.5)
    #)

    ray = mi.Ray3f(o=cam_transform.translation() + ray_origin_local, d=cam_transform.transform_affine(ray_direction_local))
    diffray = mi.RayDifferential3f(ray)

    integrator=mi.load_dict({'type':'path', 'max_depth':max_depth})
    res = integrator.sample(scene, pss_sampler, diffray)
    result = dr.select(res[1] == True, res[0], 0)
    luminance = mi.luminance(result) # TODO This is where luminance weights come in
    return luminance, result, pixel_x, pixel_y