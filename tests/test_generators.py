from statdp.algorithms import noisy_max_v1a
from statdp.generators import generate_databases, generate_arguments


def test_generate_databases():
    input_list = generate_databases(noisy_max_v1a, 5, {'epsilon': 0.5})
    assert isinstance(input_list, list) and len(input_list) >= 1
    for input_ in input_list:
        assert isinstance(input_, (list, tuple)) and len(input_) == 3
        d1, d2, args = input_
        assert isinstance(d1, (tuple, list)) and isinstance(d2, (tuple, list))
        assert len(d1) == 5 and len(d2) == 5
        assert isinstance(args, (tuple, list, dict))


def test_generate_arguments():
    d1, d2 = tuple(1 for _ in range(5)), tuple(2 for _ in range(5))
    assert generate_arguments(noisy_max_v1a, d1, d2, {}) is None
