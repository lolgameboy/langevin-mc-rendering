import mitsuba as mi
import drjit as dr

mi.set_variant("llvm_ad_rgb")

def eval_pss_sample(scene,
               primary_samples,   # dr.Float, shape = (N,)
               max_depth=4):
    """
    Takes a PSS sample (array of uniform numbers), turns it into a path, 
    and evaluates the radiance along that path.

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
    ray_direction_local = dr.normalize(mi.Vector3f(next_1d(), next_1d(), 1))
    cam_transform = scene.sensors()[0].m_to_world

    # Also determine affected pixel already
    ray = mi.Ray3f(o=cam_transform.transform_affine(ray_origin_local), d=cam_transform.transform_affine(ray_direction_local))
    throughput = 1
    L = mi.Spectrum(0.0)

    active = mi.Bool(True)
    test = 0
    # ------------------------------------------------------------------
    # Path tracing loop (fixed depth)
    # ------------------------------------------------------------------
    for depth in range(max_depth):
        si = scene.ray_intersect(ray, active)

        # Add emission
        emitter = si.emitter(scene)
        L += throughput * emitter.eval(si, active)
        test += emitter.eval(si, active)

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

    print(test)
    return L



dim = 12  # number of primary sample dimensions
u = dr.zeros(mi.Float, dim)

# Initialize with random numbers
rng = dr.rng(seed=mi.UInt32(0))
u = rng.random(mi.ArrayXf, (dim,1))

scene = mi.load_file("scenes/scene_all_emitters.xml")

dr.enable_grad(u)
with dr.resume_grad():
    L = eval_pss_sample(scene, u, 8)

dr.backward(L)
print(L)
grad_u = dr.grad(u)
print(grad_u)