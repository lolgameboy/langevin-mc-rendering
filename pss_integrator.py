import drjit as dr
import mitsuba as mi

from pss_sampler import PssSampler

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
        rng = dr.rng(0)

        # Instantiate a path tracer object
        pathtracer = mi.load_dict({
            'type': 'path',
            'max_depth': 4
        })

        sample_size = 15
        
        # Determine size of "physical" image plane from fov parameter
        x_fov = mi.traverse(sensor)["x_fov"]
        plane_height = dr.tan(x_fov / 2)
        plane_size = mi.Vector2f(plane_height, plane_height)

        # Initial sample
        sample = dr.full(mi.ArrayXf, 0.5, (sample_size,1)) # [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        print(sample)
        luminance, res, pixel_x, pixel_y = self.calculate_sample_contribution(sample, scene, pathtracer, cam_transform, plane_size, resolution)

        image_block.put(mi.Point2f(pixel_x, pixel_y), res[0] / luminance)

        N = 10000
        for i in range(N):
            # Generate proposal mutation
            mutation = rng.normal(mi.ArrayXf, (sample_size,1)) * 0.1
            # mutation = [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05]
            # prop_sample = []
            # for i in range(len(sample)):   
            #    prop_sample.append(sample[i] + mutation[i])
            prop_sample = sample + mutation

            # Boundary conditions: loop around
            for i in range(len(prop_sample)):
                if prop_sample[i] < 0:
                    prop_sample[i] += 1
                if prop_sample[i] > 1:
                    prop_sample[i] -= 1

            # TODO make it more efficient (if it is worth it)
            # if prop_sample < 0:
            #    prop_sample += 1
            #if prop_sample > 1:
            #    prop_sample -= 1

            # Calculate proposal luminance
            prop_luminance, prop_res, prop_pixel_x, prop_pixel_y = self.calculate_sample_contribution(prop_sample, scene, pathtracer, cam_transform, plane_size, resolution)
            
            # Calculate acceptence chance            
            a = dr.min(1, prop_luminance / luminance)

            print(a)
            # print(rng.uniform(mi.Float, 1)[0])
            # TODO Is dit correct?
            if rng.uniform(mi.Float, 1)[0] < a:
               luminance, res, pixel_x, pixel_y = prop_luminance, prop_res, prop_pixel_x, prop_pixel_y
               sample = prop_sample

            print(sample)
            # Explanation: PSSMLT actually uses luminance (combination of RGB values) to explore domain 
            # (because it only takes a scalar output) to mutate the sample)
            # Then it intuitively counts pixel visits to determine luminance distribution over the whole image
            # BUT! We use RGB channels. Thus, we must keep track of how this luminance is "divided" amongst the different colors.
            # So instead of counting "1" to the pixel, we count the fractions of colors, and they average to 1.
            # print("RES")
            # print(res[0] / luminance)
            image_block.put(mi.Point2f(pixel_x, pixel_y), res[0] / luminance)
        
        # Samples have now simply counted all samples per bucket. 
        # Now we must appropriately devide the whole image to get a proper distribution that integrates to 1
        # We must devide by sample count and multiply by buckets (reasoning: 1D uniform integral, 2 buckets, 10 samples)
        # The / 2 is arbitrary. This determines overall brightness of image. 
        # TODO: the / 2 (or even / 5) shows that a lot of samples are taken on the light itself
        print(image_block.tensor() / N * resolution.x * resolution.y / 3)
        return image_block.tensor() / N * resolution.x * resolution.y / 3
    
    def calculate_sample_contribution(self, sample, scene, pathtracer, cam_transform, plane_size, resolution):
        pss_sampler = PssSampler(sample)

        # Sample an initial ray through the image plane
        ray_origin_local = mi.Vector3f(0, 0, 0)
        rand_x = pss_sampler.next_1d()
        rand_y = pss_sampler.next_1d()
        x = -plane_size.x / 2 + rand_x * plane_size.x
        y = -plane_size.y / 2 + rand_y * plane_size.y
        ray_direction_local = dr.normalize(mi.Vector3f(x, y, 1))

        # Also determine affected pixel already
        pixel_x = mi.Float(rand_x * resolution.x)
        pixel_y = mi.Float(rand_y * resolution.y)

        ray = mi.Ray3f(o=cam_transform.translation() + ray_origin_local, d=cam_transform.transform_affine(ray_direction_local))
        diffray = mi.RayDifferential3f(ray)

        # Sample this ray
        res = pathtracer.sample(scene, pss_sampler, diffray)
        luminance = dr.sum(res[0]) / 3 # TODO This is where luminance weights come in, now I use equal weights

        return luminance, res, pixel_x, pixel_y