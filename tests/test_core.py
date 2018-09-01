from statdp.algorithms import noisy_max_v1a
from statdp.core import hypothesis_test


def test_core_single():
    D1 = [0] + [2 for _ in range(4)]
    D2 = [1 for _ in range(5)]
    event = [0]
    p1, p2 = hypothesis_test(noisy_max_v1a, {'epsilon': 0.5}, D1, D2, event, 0.25, 100000, cores=1)
    assert 0 <= p1 <= 0.05
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, {'epsilon': 0.5}, D1, D2, event, 0.5, 100000, cores=1)
    assert 0.05 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, {'epsilon': 0.5}, D1, D2, event, 0.75, 100000, cores=1)
    assert 0.95 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0


def test_core_multi():
    D1 = [0] + [2 for _ in range(4)]
    D2 = [1 for _ in range(5)]
    event = [0]
    p1, p2 = hypothesis_test(noisy_max_v1a, {'epsilon': 0.5}, D1, D2, event, 0.25, 100000, cores=0)
    assert 0 <= p1 <= 0.05
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, {'epsilon': 0.5}, D1, D2, event, 0.5, 100000, cores=0)
    assert 0.05 <= p1 <= 1.0
    assert 0.05 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, {'epsilon': 0.5}, D1, D2, event, 0.75, 100000, cores=0)
    assert 0.95 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0
