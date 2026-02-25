from pathlib import Path
import drjit as dr
import mitsuba as mi

from trace_path import calculate_sample_contribution

# Calculates the log of the gaussian pdf of a multivariate. log to avoid underflow
# Checked, is correct (for diagonal SIGMA case!)
def log_gaussian_diag(x, mu, var):
    # x, mu, var are same-shaped Dr.Jit arrays
    k = len(x)

    diff = x - mu

    log_det = dr.sum(dr.log(var))
    quad = dr.sum((diff * diff) / var)

    return -0.5 * (k * dr.log(2.0 * dr.pi) + log_det + quad)

# Used for testing if my own path tracer is correct
@dr.syntax
def render_mc(scene: mi.Scene, sensor: mi.Sensor, seed: mi.UInt = 0, spp: int = 0, develop: bool = True, evaluate: bool = True) -> mi.TensorXf:
        film = sensor.film()
        resolution = film.crop_size()

        image_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            3
        )

        sample_counts_block = mi.ImageBlock(
            resolution,
            mi.ScalarPoint2i(0,0),
            1
        )

        cam_transform = sensor.m_to_world
        rng = dr.rng(seed=mi.UInt32(0))

        sample_size = 50
        
        # Determine size of "physical" image plane from fov parameter
        # x_fov is a lie (or I am retarded somewhere else in my code)
        y_fov = mi.traverse(sensor)["x_fov"]
        plane_height = dr.tan(dr.deg2rad(y_fov)) # Originally divided y_fov by 2, but turns out it is already halved in spec?
        plane_size = mi.Vector2f(plane_height * resolution.x / resolution.y, plane_height)

        N = mi.Int(3000000)
        i = mi.Int(0)

        while i < N:
            sample = rng.random(mi.ArrayXf, (sample_size,1))

            dr.enable_grad(sample)
            luminance, res, pixel_x, pixel_y = calculate_sample_contribution(sample, scene, cam_transform, plane_size, resolution, rng)
            dr.backward(luminance, dr.ADFlag.AllowNoGrad)
            dr.print(dr.grad(sample))
        
            image_block.put(mi.Point2f(pixel_x, pixel_y), res)
            sample_counts_block.put(mi.Point2f(pixel_x, pixel_y), [1])
            i += 1
        
        image = image_block.tensor() / sample_counts_block.tensor()
        return image

def render_ref(scene, spp=100, max_depth=-1):
    refimage = mi.render(scene, spp=spp, integrator=mi.load_dict({'type':'path', 'rr_depth':5, 'max_depth':max_depth}))
    return refimage

def render_convergence(scene, rmsediff_max, use_cached=True):
    converged = False
    i = 1
    prevImage = render_ref(scene, 1)
    while not converged:
        pathStr = f'cache/ref_{2**i}.exr'
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
            ).write(f'cache/ref_{2**i}.png')
        rmsediff = dr.sqrt(dr.mean(dr.square(prevImage - image)))
        print(f'i:{i}, spp:{2**i}, rmsediff:{rmsediff}')
        if rmsediff < rmsediff_max:
            converged = True
        i += 1
        prevImage = image
    return image