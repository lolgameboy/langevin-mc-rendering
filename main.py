import drjit as dr
import mitsuba as mi

# Before importing own code!
#define PYBIND11_DETAILED_ERROR_MESSAGES
mi.set_variant("llvm_ad_rgb")

import matplotlib.pyplot as plt

import pss_integrator

#dr.set_flag(dr.JitFlag.Debug, True)

mi.register_integrator("pss", lambda props: pss_integrator.Pss)

print(mi.PluginManager.instance().plugin_type("pss"))

scene = mi.load_file("scene.xml")

img = mi.render(scene)

plt.axis("off")
plt.imshow(img)

plt.show()
