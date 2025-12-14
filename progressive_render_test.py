import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../mitsuba3/build/python')


import mitsuba as mi
mi.set_variant("scalar_rgb")

import matplotlib.pyplot as plt
import numpy as np
import time

# Load a scene
scene = mi.load_dict(mi.cornell_box())

sensor = scene.sensors()[0]
spp_per_pass = 1
passes = 200

accum = None

plt.ion()
fig, ax = plt.subplots()

for i in range(passes):
    # render 1 spp at a time
    img_pass = mi.render(scene, sensor=sensor, spp=spp_per_pass, seed=i, integrator='path')
    img_np = np.array(img_pass)

    if accum is None:
        accum = img_np
    else:
        accum = accum + img_np

    preview = accum / (i + 1)

    ax.clear()
    ax.imshow(np.clip(preview ** (1/2.2), 0, 1))
    ax.set_title(f"Pass {i+1}/{passes}")
    plt.pause(0.01)