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

        # Stop if no hit
        active = active & si.is_valid()

        emitter = si.emitter(scene)
        L += dr.select(active, throughput * emitter.eval(si, active) * mis_weight, mi.luminance(0))

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

        L += dr.select(active, throughput * weight * bsdf_val * mis_weight, mi.luminance(0))

        # Volgens mij include weight al wel als lichtbron niet zichtbaar is, omdat we hier geen valid bool als return krijgen

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
        throughput = dr.select(active, throughput * bsdf_weight, 0)

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




# Essentially the bidirectional path tracing function
# Implementation heavily based on a discription in Veach's Phd thesis.
# Especially MIS weights are probably impossible to understand without looking at Veach's Phd thesis first
# because this is very finnicky math
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

    ray_origin = cam_transform.translation() + ray_origin_local
    ray_direction = cam_transform.transform_affine(ray_direction_local)
    ray = mi.Ray3f(o=ray_origin, d=ray_direction)
    active = mi.Bool(True)
    
    
    # Intersect ray and calculate first vertex area probability
    si = scene.ray_intersect(ray, active)

    d = si.p - ray_origin
    dist2 = dr.squared_norm(d)
    d = dr.normalize(d)
    # Z-axis is forward direction
    cam_normal = cam_transform.transform_affine(mi.Vector3f(0,0,1))

    G = dr.abs_dot(cam_normal, d) * dr.abs_dot(si.n, -d) / dist2
    p_a_v0 = (1/(plane_size.x * plane_size.y)) / (dr.abs_dot(cam_normal, d) ** 4)
    #p_a_v0 = p_a_v0 * G # G-term cancels out like the rest, so TODO like this, p_a_v0 is not really an area measure
    
    # Throughput is f(x)/p(x). Path radiance divided by path probability.
    # As such, throughput initial value is reciprocal area probability of initial vertex
    # (Throughput is equal to alpha_E from veach thesis)
    throughput = mi.Float(1.0 / p_a_v0)

    # TODO turns out my p_a_v0 is either incorrect or throughput should be 1 anyway? 
    # Or there is another issue somewhere else in my code
    #throughput = 1

    # ------------------------------------------------------------------
    # Path tracing loop
    # ------------------------------------------------------------------
    for depth in range(max_depth):
        bsdf = si.bsdf()
        ctx = mi.BSDFContext()

        # Zero light vertex subpath (camera path hits emitter)
        emitter = si.emitter(scene)

        # Stop if no hit
        active = active & si.is_valid()

        # TODO Doesnt work because active is symbolic or something
        #if not active:
        #    print("No hit!")
        #    break

        camera_path.append({'si': si, 'throughput': throughput, 'emission': emitter.eval(si, active), 'active':active})

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
        
        # This one really takes the cake. THERE IS A DIFFERENCE BETWEEN *= AND = ... * ???? How the hell does that happen???
        # Alright, apparently: *= in _in place_ mutation, = * creates new object. This does not make a difference for literals
        # but it *can* make a difference for objects, which mitsuba uses here apparently 
        #                   (even for floats, makes sense tho because of their whole retargeting system).

        throughput = throughput * bsdf_weight
        #throughput *= bsdf_weight # DOES NOT WORKK!! in-place, so mutates previous throughput value
        
        # Spawn next ray
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        si_next = scene.ray_intersect(ray, active)

        # Update si
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


    # ------------------------------------------
    # BUILD LIGHT PATH
    # ------------------------------------------

    ray, weight, emitter = scene.sample_emitter_ray(mi.Float(0), pss_sampler.next_1d(), pss_sampler.next_2d(), pss_sampler.next_2d(), True)
    
    # TODO TEMP (Tested, manual emitter weight made no measurable diff with emitter_weight = weight)
    ps, w = emitter.sample_position(0, pss_sampler.next_2d())
    emitter_value = mi.Color3f(17, 12, 4)

                                # TODO TEMPorarily used w as P_A, but not harmonous with emitter_weight = weight above
                                # Update harmonous now, because i manually compute emitter weight
                                # but TODO this manual computation does not work with non-uniform emitters or multiple emitters
    light_path = [{'point': ray.o, 'normal': ps.n, 'P_A': w, 'active':True}]
    throughput = 1 / (w * (1 / (2 * dr.pi))) # TODO Make alpha_l. Currently throughput is alpha_l without emitter value

    active = mi.Bool(True)

    si = scene.ray_intersect(ray, active)
    # Path tracing loop
    for depth in range(max_depth):
        bsdf = si.bsdf()
        ctx = mi.BSDFContext()

        active = active & si.is_valid()

        light_path.append({'si': si, 'throughput': throughput, 'emitter_value':emitter_value, 'active':active})
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
        throughput = throughput * bsdf_weight

        # Spawn next ray
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
        si_next = scene.ray_intersect(ray, active)
        
        # Update si
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


    
    def geometry_term(si_a, si_b):
        d = si_b.p - si_a.p
        dist2 = dr.squared_norm(d)
        d = dr.normalize(d)

        G = (dr.abs_dot(si_a.n, d) *
             dr.abs_dot(si_b.n, -d)) / dist2

        return G, d, dist2
    
    def visible(si_a, pb):
        d = mi.Vector3f(pb - si_a.p)
        dist = dr.norm(d)
        d = d / dist

        # TODO Fix false positive visibility when pb is on other side of a plane (need info about normals to fix?)
        # Spawn ray offsets ray slightly from origin to prevent intersecting with its own origin due to numerical errors
        ray = si_a.spawn_ray(d)
        ray = mi.Ray3f(ray, dist - 2 * mi.math.ShadowEpsilon)
        return ~scene.ray_test(ray)

    # --------------------------------------------------------
    # Main integrator loop (connect all possible paths)
    # --------------------------------------------------------

    # Function to calculate the MIS weight of the path with l + 1 light vertices and c + 1 camera vertices
    # Remember! Paths with zero camera vertices are not included!
    # l + c should be at least 0
    def mis_weight(l, c):
        cam_path_copy = camera_path.copy()
        cam_path_copy.reverse()
        path = light_path[0:l+1] + cam_path_copy[0:c+1]
        path_length = l + 1 + c + 1
        assert len(path) == path_length

        s = l + 1
        ctx = mi.BSDFContext()
        p_s = 1 # arbitrary because only need relative values for mis formula. 
        # Note that this means that the p_i's calculated below are *relative to* ps. That is: p_i / p_s
        # This is fine because we only need to know them relative to p_s to calculate the mis weight.
        # Should I want to calculate the p_i in absolute values, I simply need to supply the correct p_s below
        #                                                         (And several lines below also!)

        p_i_list = [0] * path_length
        p_i = p_s
        p_i_list[s] = p_i

        # First create list of all relative ratios, as we will need to use these multiple times
        ratio_list = [0] * path_length

        # Ratio 0
        # unless l == -1 (no light path vertices), v0 has no si.
        # But we can create one, all I need is normal and position for geometric term
        if l == -1:
            si_b = path[0]['si']
        else:
            si_b = mi.SurfaceInteraction3f()
            si_b.p = path[0]['point']
            si_b.n = path[0]['normal']

        si_c = path[1]['si']
        if l == -1:
            numerator = dr.select(si_b.emitter(scene) == None, 1e-10, light_path[0]['P_A'])
            # TODO To work with emitters with varying emission (thus varying surface probability), 
            # I would need the following line as last argument in dr.select, but this method is not implemented yet -_-
            # si_b.emitter(scene).pdf_position(mi.PositionSample3f(si_b))
        else:
            numerator = light_path[0]['P_A']
        denominator = (si_c.bsdf().pdf(ctx, si_c, si_c.to_local(dr.normalize(si_b.p - si_c.p)))
                * geometry_term(si_c, si_b)[0])
        ratio_list[0] = numerator / denominator

        # Ratio 1
        # unless l == -1, v0 has no si so we need a workaround to find pdf
        if path_length >= 3: # Only calc ratio 1 if path is at least length 3
            si_b = path[1]['si']
            si_c = path[2]['si']

            denominator = (si_c.bsdf().pdf(ctx, si_c, si_c.to_local(dr.normalize(si_b.p - si_c.p)))
                    * geometry_term(si_c, si_b)[0])
        
            # Get bsdf angular pdf of vertex on light source
            # TODO Currently assume uniform probability. How to handle directional light sources?
            light_ang_pdf = 1 / (2 * dr.pi)

            # Create half-initialized si_a for geometry term calculation
            if l == -1:
                si_a = path[0]['si']
                # If no light vertices, v0 does have a si
                ang_pdf = si_a.bsdf().pdf(ctx, si_a, si_a.to_local(dr.normalize(si_b.p - si_a.p)))
            else:
                si_a = mi.SurfaceInteraction3f()
                si_a.p = path[0]['point']
                si_a.n = path[0]['normal']
                ang_pdf = light_ang_pdf
            numerator = ang_pdf * geometry_term(si_a, si_b)[0]
            ratio_list[1] = numerator / denominator

        # Rest of ratios
        for i in range(2, path_length - 1):
            si_a = path[i-1]['si']
            si_b = path[i]['si']
            si_c = path[i+1]['si']
            numerator = (si_a.bsdf().pdf(ctx, si_a, si_a.to_local(dr.normalize(si_b.p - si_a.p)))
                  * geometry_term(si_a, si_b)[0]) 
            denominator = (si_c.bsdf().pdf(ctx, si_c, si_c.to_local(dr.normalize(si_b.p - si_c.p)))
                  * geometry_term(si_c, si_b)[0])
            ratio_list[i] = numerator / denominator


        # Now calculate p_i's

        # Since I do not use paths with zero camera samples, I only need to loop i up to path_length - 1
        # (i.e. calculate p_i for paths with maximum (path_length - 1) light vertices)
        for i in range(s, path_length - 1):
            p_i = p_i * ratio_list[i]
            p_i_list[i+1] = p_i

        # Now the reverse: from s-1 to 0
        p_i = p_s
        for i in range(s-1, -1, -1):
            # Reciprocal ratio so division!
            p_i = p_i / ratio_list[i]
            p_i_list[i] = p_i # index in p_i_list here is i, not i + 1!

        # Calculate MIS weight (TODO currently power heuristic, use balanced maybe? Experiments?)
        final_weight = 0
        for i in range(len(p_i_list)):
            final_weight = final_weight + (p_i_list[i] ** 2)
        final_weight = 1 / final_weight
        return final_weight
    
    def naive_weight(l, c):
        return 1
        return 1 / (l + 1 + c + 1)

    # All (s, t) connections except camera path length 0
    # (Camera path is always at least length 1, light path can be 0)
    # Attention! c and l indices in path array. c=0 means camera path length 1
    result = mi.Spectrum(0.0)

    # All camera paths length >= 1
    for c in range(0, 1): #len(camera_path)):
        vc = camera_path[c]
        si_c = vc['si']

        # Light path length 0 ----------------------
        contrib = vc['emission'] * vc['throughput']
        if c >= 1: # MIS weight for c = 0 and l = 0 (camera ray hits emitter) is 1, 
                   # because there in only one strategy for this case
            contrib *= naive_weight(-1, c)
        result += dr.select(vc['active'], contrib, mi.Spectrum(0.0))

        # Light path length 1 ----------------------

        # Connect camera vertex directly to light source
        vl = light_path[0]

        vis = visible(si_c, vl['point'])

        d = mi.Vector3f(vl['point'] - si_c.p)
        d = dr.normalize(d)

        ray = si_c.spawn_ray(d)
        si_l = scene.ray_intersect(ray)
        ds = mi.DirectionSample3f(scene, si_l, si_c)
        radiance = scene.eval_emitter_direction(si_c, ds)

        G, _, _ = geometry_term(si_c, si_l)

        bsdf_c = si_c.bsdf()

        ctx = mi.BSDFContext()
        f_c = bsdf_c.eval(ctx, si_c, si_c.to_local(d), active=True)

        contrib = radiance * vc['throughput'] * f_c * G / vl['P_A']
        contrib = dr.select(vis, contrib, mi.Spectrum(0)) * naive_weight(0, c)

        #result += dr.select(vc['active'] & vl['active'], contrib, mi.Spectrum(0.0))

        # Light path lengths 2 and up
        for l in range(1, 1): #len(light_path)):
            vl = light_path[l]
            si_l = vl['si']

            G, d, _ = geometry_term(si_c, si_l)

            vis = visible(si_c, si_l.p)

            bsdf_c = si_c.bsdf()
            bsdf_l = si_l.bsdf()

            ctx = mi.BSDFContext()
            f_c = bsdf_c.eval(ctx, si_c, si_c.to_local(d), active=True)
            f_l = bsdf_l.eval(ctx, si_l, si_l.to_local(-d), active=True)

            contrib = (
                vl['emitter_value'] *
                vc['throughput'] *
                vl['throughput'] *
                f_c * f_l * G
            )

            contrib = dr.select(vis, contrib, mi.Spectrum(0)) * naive_weight(l, c)

            result += dr.select(vc['active'] & vl['active'], contrib, mi.Spectrum(0.0))

    luminance = mi.luminance(result)
    return luminance, result, pixel_x, pixel_y