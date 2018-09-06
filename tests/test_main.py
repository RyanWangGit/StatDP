from statdp.__main__ import detect_counterexample
from statdp.algorithms import noisy_max_v1a


def test_main():
    import logging
    result = detect_counterexample(noisy_max_v1a, 0.5, {'epsilon': 0.2}, loglevel=logging.DEBUG)
    assert isinstance(result, list) and len(result) == 1
    epsilon, p = result[0]
    assert epsilon == 0.5 and p >= 0.95
    result = detect_counterexample(noisy_max_v1a, 0.2, {'epsilon': 0.5}, loglevel=logging.DEBUG)
    epsilon, p = result[0]
    assert epsilon == 0.2 and p <= 0.05
