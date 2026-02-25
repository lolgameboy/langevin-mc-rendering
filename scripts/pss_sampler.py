import drjit as dr
import mitsuba as mi

class PssSampler(mi.Sampler):
    # Les geleerd: ik MOET de superklasse insantieren om de sampler te laten herkennen als mi.Sampler
    # Vermoedelijk was dit ook het probleem bij het registeren van de integrator!!
    def __init__(self, sample_array):
        super().__init__(mi.Properties())
        self.sample_array = sample_array
        self.sample_index = 0

    def next_1d(self, active = True):
        sample = self.sample_array[self.sample_index]
        self.sample_index += 1
        return sample
    
    def next_2d(self, active = True):
        return mi.Point2f(self.next_1d(), self.next_1d())