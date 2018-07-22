from statdp.algorithms import noisy_max_v1a
from statdp.selectors import select_event


def test_select_event():
    D1 = [0] + [2 for _ in range(4)]
    D2 = [1 for _ in range(5)]
    event = select_event(noisy_max_v1a, (), {'eps': 0.5}, D1, D2, 0.5, 100000, [[i] for i in range(6)], cores=0)
    assert event == [0]
