import drjit as dr
import mitsuba as mi
from pss_sampler import PssSampler


# Essentially the path tracing function
@dr.syntax
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
    L = mi.Spectrum(0.0)

    # ------------------------------------------------------------------
    # Path tracing loop
    # ------------------------------------------------------------------
    si = scene.ray_intersect(ray, active)
    mis_weight = 1
    for depth in range(max_depth):
        bsdf = si.bsdf()
        ctx = mi.BSDFContext()

        emitter = si.emitter(scene)
        L += throughput * emitter.eval(si, active) * mis_weight

        # Add light ray
        # TODO light ray active checks
        ds, weight = scene.sample_emitter_direction(si, pss_sampler.next_2d())

        # ds.d is unit direction from si point to sampled point on emitter
        wo = si.to_local(ds.d)
        bsdf_val = bsdf.eval(ctx, si, wo, active)
        bsdf_pdf = bsdf.pdf(ctx, si, wo, active)

        light_pdf = ds.pdf
        # TODO Small bias because of fix to division by 0 when inactive. How fix? 
        # Can probably be fixed if I get the active mask right finally
        mis_weight = light_pdf / (light_pdf + bsdf_pdf + 1e-10)

        L += throughput * weight * bsdf_val * mis_weight
        # Volgens mij include weight al wel als lichtbron niet zichtbaar is, omdat we hier geen valid bool als return krijgen

        # Stop if no hit
        active = active & si.is_valid()

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
        si_next = scene.ray_intersect(ray, active)

        # MIS weight
        ds = mi.DirectionSample3f(scene, si_next, si)
        light_pdf = scene.pdf_emitter_direction(si, ds, active)
        mis_weight = bsdf_sample.pdf / (bsdf_sample.pdf + light_pdf + 1e-10)
        st = bsdf_sample.sampled_type
        is_delta = mi.has_flag(st, mi.BSDFFlags.Delta)
        # If we sampled a dirac_delta, mis_weight must be 1
        if is_delta:
            mis_weight = mi.Float(1)

        # TODO mis_weight for delta bsdf (e.g. mirror). pdf is 0, so weight is 0, but should not be
        si = si_next

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
        

def calculate_sample_contribution_ref(sample, scene, cam_transform, plane_size, resolution, max_depth = 6):
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

    integrator=mi.load_dict({'type':'path', 'max_depth':max_depth,'rr_depth':max_depth+1})
    res = integrator.sample(scene, pss_sampler, diffray)
    result = dr.select(res[1] == True, res[0], 0)
    luminance = mi.luminance(result) # TODO This is where luminance weights come in
    return luminance, result, pixel_x, pixel_y




# Essentially the path tracing function
@dr.syntax
def calculate_sample_contribution_bidir(sample, scene, cam_transform, plane_size, resolution, max_depth = 6):
    pss_sampler = PssSampler(sample)

    # ------------------------------------------
    # BUILD CAMERA PATH
    # ------------------------------------------
    camera_path = []

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


        camera_path.append({'si': si, 'throughput': throughput, 'L':L})

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


    # ------------------------------------------
    # BUILD LIGHT PATH
    # ------------------------------------------

    ray, weight, _ = scene.sample_emitter_ray(mi.Float(0), pss_sampler.next_1d(), pss_sampler.next_2d(), pss_sampler.next_2d(), True)

    light_path = []

    throughput = 1
    L = weight

    # Path tracing loop
    for depth in range(max_depth):
        si = scene.ray_intersect(ray, active)
        bsdf = si.bsdf()
        ctx = mi.BSDFContext()

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

        light_path.append({'si': si, 'throughput': throughput, 'L':L})

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


    
    def geometry_term(si_a, si_b):
        d = si_b.p - si_a.p
        dist2 = dr.squared_norm(d)
        d = dr.normalize(d)

        G = (dr.abs_dot(si_a.n, d) *
             dr.abs_dot(si_b.n, -d)) / dist2

        return G, d, dist2
    
    def visible(si_a, si_b):
        d = mi.Vector3f(si_b.p - si_a.p)
        dist = dr.norm(d)
        d = d / dist

        ray = mi.Ray3f(si_a.p, d, mi.Float(mi.math.ShadowEpsilon))
        ray = mi.Ray3f(ray, dist - mi.math.ShadowEpsilon)
        return ~scene.ray_test(ray)

    # --------------------------------------------------------
    # Main integrator loop (connect all possible paths)
    # --------------------------------------------------------

    result = mi.Spectrum(0.0)

    # All (s, t) connections
    for t in range(len(camera_path)):
        for s in range(len(light_path)):
            vc = camera_path[t]
            vl = light_path[s]

            si_c = vc['si']
            si_l = vl['si']

            G, d, _ = geometry_term(si_c, si_l)

            vis = visible(si_c, si_l)

            bsdf_c = si_c.bsdf()
            bsdf_l = si_l.bsdf()

            ctx = mi.BSDFContext()
            f_c = bsdf_c.eval(ctx, si_c, si_c.to_local(d), active=True)
            f_l = bsdf_l.eval(ctx, si_l, si_l.to_local(-d), active=True)

            contrib = (
                vl['L'] *
                vc['throughput'] *
                vl['throughput'] *
                f_c * f_l * G
            )

            # NEE contribution
            contrib += vc['L']

            # TODO needed?
            # contrib /= 2

            contrib = dr.select(vis, contrib, mi.Spectrum(0))

            result += contrib

    luminance = mi.luminance(result)
    return luminance, result, pixel_x, pixel_y