from statdp.__main__ import detect_counter_example
from statdp.algorithms import noisy_max_v1a


def test_main():
    assert detect_counter_example(noisy_max_v1a, 0.5, {'epsilon': 0.2}) >= 0.95
