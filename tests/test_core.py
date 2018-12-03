import logging
from statdp.core import detect_counterexample
from statdp.algorithms import noisy_max_v1a, noisy_max_v1b, noisy_max_v2a, noisy_max_v2b,\
    SVT, iSVT1, iSVT2, iSVT3, iSVT4


def assert_correct_algorithm(algorithm, kwargs=None):
    if kwargs and isinstance(kwargs, dict):
        kwargs.update({'epsilon': 0.7})
    else:
        kwargs = {'epsilon': 0.7}
    result = detect_counterexample(algorithm, (0.6, 0.7, 0.8), kwargs, loglevel=logging.DEBUG)
    assert isinstance(result, list) and len(result) == 3
    epsilon, p, *extras = result[0]
    assert p <= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)
    epsilon, p, *extras = result[1]
    assert p >= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)
    epsilon, p, *extras = result[2]
    assert p >= 0.95, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)


def assert_incorrect_algorithm(algorithm, kwargs=None):
    if kwargs and isinstance(kwargs, dict):
        kwargs.update({'epsilon': 0.7})
    else:
        kwargs = {'epsilon': 0.7}
    result = detect_counterexample(algorithm, 0.7, kwargs, loglevel=logging.DEBUG)
    assert isinstance(result, list) and len(result) == 1
    epsilon, p, *extras = result[0]
    assert p <= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)


def test_detection():
    assert_correct_algorithm(noisy_max_v1a)
    assert_incorrect_algorithm(noisy_max_v1b)
    assert_correct_algorithm(noisy_max_v2a)
    assert_incorrect_algorithm(noisy_max_v2b)
    assert_correct_algorithm(SVT, {'N': 1, 'T': 0.5})
    assert_incorrect_algorithm(iSVT1, {'N': 1, 'T': 1})
    assert_incorrect_algorithm(iSVT2, {'N': 1, 'T': 1})
    assert_incorrect_algorithm(iSVT3, {'N': 1, 'T': 1})
