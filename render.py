import sys
# caution: path[0] is reserved for script path (or '' in REPL)
# Only include when I want to use my own mitsuba build
sys.path.insert(1, '../mitsuba3/build/python')

import mitsuba as mi
import drjit as dr
from drjit.llvm import TensorXf

import matplotlib.pyplot as plt
import numpy as np

mi.set_variant("scalar_rgb")

scene = mi.load_file("scene.xml")

img = mi.render(scene)

plt.axis("off")
plt.imshow(img)

plt.show()
