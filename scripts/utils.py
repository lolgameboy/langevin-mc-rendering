from pathlib import Path
import drjit as dr
import mitsuba as mi

from trace_path import calculate_sample_contribution, calculate_sample_contribution_ref

# Calculates the log of the gaussian pdf of a multivariate. log to avoid underflow
# Checked, is correct (for diagonal SIGMA case!)
def log_gaussian_diag(x, mu, var):
    k = x.shape[0]
    # x, mu, var are same-shaped Dr.Jit arrays
    diff = x - mu

    log_det = dr.sum(dr.log(var))
    quad = dr.sum((diff * diff) / var)

    return -0.5 * (k * dr.log(2.0 * dr.pi) + log_det + quad)

# Used for testing if my own path tracer is correct
@dr.syntax
def render_mc(scene: mi.Scene, sensor: mi.Sensor, N : mi.Int, seed: mi.UInt = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> mi.TensorXf:
        film = sensor.film()
        # I now actually use the film to collect the samples,
        # it is fine like this, but 
        # this is not neccesary as long as I just use box filter in scene
        # Also, for PSS I need box filter anyway, so there i do it manually
        # Reason I need box filter is that otherwise block.put adds some 
        # filter weight making sample count division invalid (see pss code)
        film.prepare([])

        resolution = film.crop_size()
        image_block = film.create_block()

        cam_transform = sensor.m_to_world
        rng = dr.rng(seed=mi.UInt32(seed))

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        # x_fov is a lie (or I am retarded somewhere else in my code)
        y_fov = mi.traverse(sensor)["x_fov"]
        plane_height = 2 * dr.tan(dr.deg2rad(y_fov / 2))
        plane_size = mi.Vector2f(plane_height * resolution.x / resolution.y, plane_height)

        i = mi.Int(0)

        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))

            luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng)
        
            value = dr.zeros(mi.ArrayXf, (image_block.channel_count(), 1))
            value[0] = res[0]
            value[1] = res[1]
            value[2] = res[2]
            value[3] = 1
            image_block.put(mi.Point2f(pixel_x, pixel_y), value)
            i += 1
        
        film.put_block(image_block)
        return film.develop()

def render_ref(scene, spp=100, max_depth=-1):
    refimage = mi.render(scene, spp=spp, integrator=mi.load_dict({'type':'path', 'rr_depth':5, 'max_depth':max_depth}))
    return refimage

def render_convergence(scene, scene_name, rmsediff_max, use_cached=True):
    converged = False
    i = 1
    prevImage = render_ref(scene, 1)
    while not converged:
        pathStr = f'cache/{scene_name}/ref_{2**i}.exr'
        path = Path(pathStr)
        if path.exists() and use_cached:
            bmp = mi.Bitmap(pathStr)
            image = mi.TensorXf(bmp)
        else:
            image = render_ref(scene, 2**i)
            mi.Bitmap(image).write(pathStr)
            # TEMP: also save as png for viewing pleasure
            mi.Bitmap(image).convert(
                component_format=mi.Struct.Type.UInt8,
                srgb_gamma=True
            ).write(f'cache/{scene_name}/ref_{2**i}.png')
        rmsediff = dr.sqrt(dr.mean(dr.square(prevImage - image)))
        print(f'i:{i}, spp:{2**i}, rmsediff:{rmsediff}')
        if rmsediff < rmsediff_max:
            converged = True
        i += 1
        prevImage = image
    return image