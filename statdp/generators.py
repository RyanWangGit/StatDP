def simple_generator(algorithm, args, kwargs, num_input, search_space):
    """
    :param algorithm: The algorithm to run.
    :param args: The argument for the algorithm.
    :param kwargs: The keyword argument for the algorithm.
    :param num_input: The number of inputs to be generated
    :return: Database (d1, d2)
    """
    assert isinstance(args, tuple)
    assert isinstance(kwargs, dict)
    from .core import hypothesis_test
    from .selectors import event_selector
    import numpy as np
    # assume maximum distance is 1
    d1 = [1 for _ in range(num_input)]
    candidates = [
        (d1, [0] + [1 for _ in range(num_input - 1)]),  # one below
        (d1, [2] + [1 for _ in range(num_input - 1)]),  # one above
        (d1, [2] + [0 for _ in range(num_input - 1)]),  # one above rest below
        (d1, [0] + [2 for _ in range(num_input - 1)]),  # one below rest above
        (d1, [2 for _ in range(int(num_input / 2))] + [0 for _ in range(num_input - int(num_input / 2))]),  # half half
        (d1, [2 for _ in range(num_input)]),  # all above
        (d1, [0 for _ in range(num_input)]),  # all below
    ]

    results = []

    for d1, d2 in candidates:
        candidate_result = []
        for algorithm_epsilon in [0.2, 0.5, 0.7, 1, 2, 3]:
            kwargs['eps'] = algorithm_epsilon

            rising_epsilon = 0.1
            steady_epsilon = algorithm_epsilon + 2.0
            previous_p = [0.0, 0.0]
            for test_epsilon in np.arange(max(algorithm_epsilon - 0.5, 0.1), algorithm_epsilon + 2.0, 0.1):
                s = event_selector(algorithm, args, kwargs, d1, d2, test_epsilon, search_space=search_space)
                p1= hypothesis_test(algorithm, args, kwargs, d1, d2, s, test_epsilon, iterations=100000, cores=0)

                rising_epsilon = test_epsilon if p1 < 0.05 and previous_p < [0.05, 0.05] else rising_epsilon

                steady_epsilon = test_epsilon if p1 > 0.95 and previous_p > [0.95, 0.95] else steady_epsilon

                # store the new p into the history
                previous_p[0], previous_p[1] = previous_p[1], previous_p[0]
                previous_p[0] = p1

                # stop early for best performance
                if steady_epsilon < algorithm_epsilon + 2.0:
                    break
            candidate_result.append(rising_epsilon - algorithm_epsilon - (steady_epsilon - rising_epsilon))
        results.append(np.mean(candidate_result))

    return candidates[np.argmax(results)]
