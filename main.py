import drjit as dr
import mitsuba as mi

dr.set_flag(dr.JitFlag.Debug, True)

# Before importing own code!
mi.set_variant("llvm_ad_rgb")

import matplotlib.pyplot as plt
from pss_integrator import Pss




# from pss_sampler import PssSampler
# scene = mi.load_file("scene.xml")
# sample = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

# print(sample)
# pss_sampler = PssSampler(sample)
# ray_origin_local = mi.Vector3f(0, 0, 0)
# ray_direction_local = mi.Vector3f(0, 0, 1)
# cam_transform = scene.sensors()[0].m_to_world
# # Also determine affected pixel already
# ray = mi.Ray3f(o=cam_transform.transform_affine(ray_origin_local), d=cam_transform.transform_affine(ray_direction_local))
# diffray = mi.RayDifferential3f(ray)


# pathtracer = mi.load_dict({
#     'type': 'path',
#     'max_depth': 4
# })


# # Sample this ray
# res = pathtracer.sample(scene, pss_sampler, diffray)

# print(res)

















pss = Pss()

scene = mi.load_file("scene.xml")
img = pss.render(scene, scene.sensors()[0])

plt.axis("off")
plt.imshow(img)

plt.show()
