from statdp.algorithms import *
import math


def test_noisymax():
    # add no noise to the array
    assert noisy_max_v1a([1, 2, 1], math.inf) == 1
    assert noisy_max_v1b([1, 3, 1], math.inf) == 3
    assert noisy_max_v2a([1, 3, 1], math.inf) == 1
    assert noisy_max_v2b([1, 3, 1], math.inf) == 3
