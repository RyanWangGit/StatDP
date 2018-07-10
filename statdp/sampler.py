import numpy
from scipy import stats


class Sampler:
    def __init__(self):
        self._configs = {}

        self._weight = 1

        self._sample_map = {
            'laplace': numpy.random.laplace,
            'normal': numpy.random.normal
        }

        self._pdf_map = {
            'laplace': stats.laplace.pdf,
            'normal': stats.norm.pdf
        }

    def weight(self):
        return self._weight

    def add_config(self, old, new, f=lambda x: x, g=lambda x: x):
        assert isinstance(old, str) and old in self._sample_map, old + ' sampling mechanism not supported.'
        assert isinstance(old, str) and new in self._sample_map, new + ' sampling mechanism not supported.'
        self._configs[old] = (new, f, g)

    def _sample(self, mech, loc=0.0, scale=1.0):
        assert isinstance(mech, str) and mech in self._sample_map

        if mech in self._configs:
            new_mech, f, g = self._configs[mech]
            new_loc, new_scale = f(loc), g(scale)
            res = self._sample_map[new_mech](new_loc, new_scale)
            self._weight *= \
                self._pdf_map[mech](res, new_loc, new_scale) / self._pdf_map[new_mech](res, new_loc, new_scale)
            return res
        else:
            return self._sample_map[mech](loc, scale)

    def laplace(self, loc=0.0, scale=1.0):
        return self._sample('laplace', loc, scale)

    def normal(self, loc=0.0, scale=1.0):
        return self._sample('normal', loc, scale)
