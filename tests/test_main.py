from statdp.__main__ import detect_counter_example
from statdp.algorithms import noisy_max_v1a


def test_main():
    result = detect_counter_example(noisy_max_v1a, 0.5, {'epsilon': 0.2})
    assert isinstance(result, list) and len(result) == 1
    epsilon, p = result[0]
    assert epsilon == 0.5 and p >= 0.95
    result = detect_counter_example(noisy_max_v1a, 0.2, {'epsilon': 0.5})
    epsilon, p = result[0]
    assert epsilon == 0.2 and p <= 0.05
