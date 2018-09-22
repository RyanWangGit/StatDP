from statdp.core import detect_counterexample
from statdp.algorithms import noisy_max_v1a


def test_main():
    import logging
    result = detect_counterexample(noisy_max_v1a, 0.5, {'epsilon': 0.2}, loglevel=logging.DEBUG)
    assert isinstance(result, list) and len(result) == 1
    epsilon, p, *_ = result[0]
    assert epsilon == 0.5 and p >= 0.95
    result = detect_counterexample(noisy_max_v1a, 0.2, {'epsilon': 0.5}, loglevel=logging.DEBUG)
    epsilon, p, *_ = result[0]
    assert epsilon == 0.2 and p <= 0.05
    d1, d2 = [0] + [2 for _ in range(4)], [1 for _ in range(5)]
    result = detect_counterexample(noisy_max_v1a, 0.2, {'epsilon': 0.5},
                                   databases=(d1, d2), loglevel=logging.DEBUG)
    epsilon, p, *_ = result[0]
    assert epsilon == 0.2 and p <= 0.05
