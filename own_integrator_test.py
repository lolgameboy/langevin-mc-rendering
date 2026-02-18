import mitsuba as mi
import drjit as dr

mi.set_variant("llvm_ad_rgb")

def render_lmc(scene,
               primary_samples,   # dr.Float, shape = (N,)
               max_depth=4):
    """
    primary_samples:
        1D differentiable array containing PSS numbers
    """

    # Primary sample indexing helper
    index = 0

    def next_1d():
        nonlocal index
        value = primary_samples[index]
        index += 1
        return value

    def next_2d():
        return mi.Point2f(next_1d(), next_1d())

    # Camera sample
    sensor = scene.sensors()[0]

    film = sensor.film()
    crop_size = film.crop_size()
 
    ray_origin_local = mi.Vector3f(0, 0, 0)
    ray_direction_local = mi.Vector3f(next_1d() * 10, next_1d() * 10, 1)
    cam_transform = scene.sensors()[0].m_to_world

    # Also determine affected pixel already
    ray = mi.Ray3f(o=cam_transform.transform_affine(ray_origin_local), d=cam_transform.transform_affine(ray_direction_local))
    throughput = 1
    L = mi.Spectrum(0.0)

    active = mi.Bool(True)

    # ------------------------------------------------------------------
    # Path tracing loop (fixed depth)
    # ------------------------------------------------------------------
    for depth in range(max_depth):
        # Test
        if False:
            si = scene.ray_intersect(ray)

            bsdf = si.bsdf()
            ctx = mi.BSDFContext()
        
            u_test1 = rng.random(mi.Float, (1))
            u_test2 = rng.random(mi.Float, (2))
            dr.enable_grad(u_test1)
            dr.enable_grad(u_test2)
            with dr.resume_grad():
                sample, weight = bsdf.sample(ctx, si, u_test1, u_test2)
            #print(sample.wo)
            dr.backward(weight, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
            #dr.backward(sample.wo, flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
            #print(dr.grad(u_test1))
            print(dr.grad(u_test2))
        
        si = scene.ray_intersect(ray, active)

        # Add emission
        emitter = si.emitter(scene)
        L += throughput * emitter.eval(si, active)

        # Stop if no hit
        active &= si.is_valid()
        print(si.is_valid())

        if not dr.any(active):
            print("No hit!")
            break

        # ------------------------------------------------------------------
        # BSDF sampling
        # ------------------------------------------------------------------
        bsdf = si.bsdf()

        ctx = mi.BSDFContext()

        u_bsdf = next_2d()
        u_component = next_1d()

        bsdf_sample, bsdf_weight = bsdf.sample(
            ctx,
            si,
            u_component,
            u_bsdf,
            active
        )

        throughput *= bsdf_weight

        # Spawn next ray
        ray = si.spawn_ray(si.to_world(bsdf_sample.wo))

        # ------------------------------------------------------------------
        # Russian roulette (deterministic dimension)
        # ------------------------------------------------------------------
        rr_prob = 0.95
        #rr_sample = next_1d()
        rr_sample = rng.random(mi.Float, (1))
        survive = rr_sample < rr_prob
        throughput /= rr_prob

        active &= survive
    print(ray.o)
    return ray.d
    return L



dim = 12  # number of primary sample dimensions
u = dr.zeros(mi.Float, dim)

# Initialize with random numbers
rng = dr.rng(seed=mi.UInt32(0))
u = rng.random(mi.ArrayXf, (dim,1))

scene = mi.load_file("scenes/scene.xml")

dr.enable_grad(u)
with dr.resume_grad():
    L = render_lmc(scene, u, 8)

dr.backward(L)
print(L)
grad_u = dr.grad(u)
print(grad_u)