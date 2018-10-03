from statdp.algorithms import noisy_max_v1a
from statdp.hypotest import hypothesis_test


def test_core_single():
    D1 = [0] + [2 for _ in range(4)]
    D2 = [1 for _ in range(5)]
    event = [0]
    p1, p2 = hypothesis_test(noisy_max_v1a, D1, D2, {'epsilon': 0.5}, event, 0.25, 100000)
    assert 0 <= p1 <= 0.05
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, D1, D2, {'epsilon': 0.5}, event, 0.5, 100000)
    assert 0.05 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, D1, D2, {'epsilon': 0.5}, event, 0.75, 100000)
    assert 0.95 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0


def test_core_multi():
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())
    D1 = [0] + [2 for _ in range(4)]
    D2 = [1 for _ in range(5)]
    event = [0]
    p1, p2 = hypothesis_test(noisy_max_v1a, D1, D2, {'epsilon': 0.5}, event, 0.25, 100000, process_pool=pool)
    assert 0 <= p1 <= 0.05
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, D1, D2, {'epsilon': 0.5}, event, 0.5, 100000, process_pool=pool)
    assert 0.05 <= p1 <= 1.0
    assert 0.05 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0
    p1, p2 = hypothesis_test(noisy_max_v1a, D1, D2, {'epsilon': 0.5}, event, 0.75, 100000, process_pool=pool)
    assert 0.95 <= p1 <= 1.0
    assert 0.95 <= p2 <= 1.0
