from statdp.algorithms import noisy_max_v1a
from statdp.generators import simple_generator


def test_database_generator():
    input_list = simple_generator(noisy_max_v1a, 5)
    assert isinstance(input_list, list) and len(input_list) >= 1
    for input_ in input_list:
        assert isinstance(input_, (list, tuple)) and len(input_) == 3
        d1, d2, args = input_
        assert isinstance(d1, (tuple, list)) and isinstance(d2, (tuple, list))
        assert len(d1) == 5 and len(d2) == 5
        assert isinstance(args, (tuple, list, dict))
