from statdp.core import detect_counterexample
from statdp.algorithms import noisy_max_v1a, noisy_max_v1b, noisy_max_v2a, noisy_max_v2b,\
    SVT, iSVT1, iSVT2, iSVT3, iSVT4


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

    # test for noisy max v2a
    result = detect_counterexample(noisy_max_v2a, (0.2, 0.7, 1.5), {'epsilon': 0.7}, loglevel=logging.DEBUG)
    assert isinstance(result, list) and len(result) == 3
    epsilon, p, *_ = result[0]
    assert epsilon == 0.2 and p <= 0.05, 'result {} is not expected.'.format(result[0])
    epsilon, p, *_ = result[1]
    assert epsilon == 0.7 and p >= 0.05, 'result {} is not expected.'.format(result[0])
    epsilon, p, *_ = result[2]
    assert epsilon == 1.5 and p >= 0.95, 'result {} is not expected.'.format(result[0])