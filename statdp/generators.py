import math


def argument_generator(algorithm, d1, d2, default_kwargs):
    """
    :param algorithm: The algorithm to test for.
    :param d1: The database 1
    :param d2: The database 2
    :param default_kwargs: The default arguments that are given or have a default value.
    :return: Extra argument needed for the algorithm besides Q and epsilon.
    """
    # TODO: implement argument_generator
    return {}


def simple_generator(algorithm, num_input, default_kwargs):
    """
    :param algorithm: The algorithm to test for.
    :param num_input: The number of inputs to be generated
    :param default_kwargs: The default arguments that are given or have a default value.
    :return: List of (d1, d2, args) with length num_input
    """

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
        # x shape
        ([1 for _ in range(int(math.floor(num_input / 2.0)))] + [0 for _ in range(int(math.ceil(num_input / 2.0)))],
         [0 for _ in range(int(math.floor(num_input / 2.0)))] + [1 for _ in range(int(math.ceil(num_input / 2.0)))])
    ]

    input_list = []
    for d1, d2 in candidates:
        kwargs = argument_generator(algorithm, d1, d2, default_kwargs)
        input_list.append((d1, d2, kwargs))

    return input_list
