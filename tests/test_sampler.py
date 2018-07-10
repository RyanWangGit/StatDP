import statdp


def test_prng():
    prng = statdp.Sampler()
    assert isinstance(prng.laplace(0, 1), float)
    assert prng.weight() == 1
    prng.add_config('laplace', 'normal')
    assert isinstance(prng.laplace(0, 1), float)
    assert prng.weight() != 1
    prng.add_config('normal', 'laplace', lambda x: x - 1, lambda x: x + 1)
    assert isinstance(prng.normal(0, 1), float)
