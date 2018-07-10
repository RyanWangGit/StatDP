import numpy as np
import multiprocessing as mp
import math
import codecs
import os
from scipy import stats


class __HyperGeometric:
    """
    Used by test_statistics function to pass hypergeometric function to multiprocessing.Pool().map,
    which only accepts pickle-able functions or objects.
    """
    def __init__(self, cy, iterations):
        self.cy = cy
        self.iterations = iterations

    def __call__(self, cx):
        np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 16))
        return 1 - stats.hypergeom.cdf(cx, 2 * self.iterations, self.iterations, cx + self.cy)


class __RunAlgorithm:
    """
    Used by hypothesis_test to run algorithm using different database concurrently.
    """
    def __init__(self, algorithm, args, kwargs, D1, D2, S):
        self.algorithm = algorithm
        self.args = args
        self.kwargs = kwargs
        self.D1 = D1
        self.D2 = D2
        self.S = S

    def __call__(self, iterations):
        return self.run(iterations)

    def run(self, iterations):
        np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 16))
        cx = sum(1 for _ in range(iterations) if self.algorithm(self.D1, *self.args, **self.kwargs) in self.S)
        cy = sum(1 for _ in range(iterations) if self.algorithm(self.D2, *self.args, **self.kwargs) in self.S)
        return cx, cy


# global process pool
_process_pool = mp.Pool(mp.cpu_count())


def test_statistics(cx, cy, epsilon, iterations):
    """ old test method
    counter = 0
    for i in range(iterations):
        r = np.random.binomial(cx, 1.0 / (np.exp(epsilon)))
        t = np.random.binomial(cy + r, 0.5)
        if t >= r:
            counter += 1

    return counter
    """
    global _process_pool
    # use a multiprocessing.Pool to parallel average p value calculation
    return np.mean(_process_pool.map(__HyperGeometric(cy, iterations),
                                     np.random.binomial(cx, 1.0 / (np.exp(epsilon)), 1000),
                                     chunksize=int(1000 / mp.cpu_count())))


def hypothesis_test(algorithm, args, kwargs, D1, D2, S, epsilon, iterations, cores=0):
    """
    :param algorithm: The algorithm to run on
    :param args: The arguments the algorithm needs
    :param kwargs: The keyword arguments the algorithm needs
    :param D1: Database 1
    :param D2: Database 2
    :param S: The S set
    :param iterations: Number of iterations to run
    :param epsilon: The epsilon value to test for
    :param cores: Number of processes to run, default is 1 and 0 means utilizing all cores.
    :return: p value.
    """
    np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 16))
    if cores == 1:
        cx, cy = __RunAlgorithm(algorithm, args, kwargs, D1, D2, S).run(iterations)
        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return test_statistics(cx, cy, epsilon, iterations)
    else:
        global _process_pool
        process_count = mp.cpu_count() if cores == 0 else cores

        process_iterations = [int(math.floor(float(iterations) / process_count)) for _ in range(process_count)]
        # add the remaining iterations to the last index
        process_iterations[process_count - 1] += iterations % process_iterations[process_count - 1]

        result = _process_pool.map(__RunAlgorithm(algorithm, args, kwargs, D1, D2, S), process_iterations)

        cx, cy = 0, 0
        for process_cx, process_cy in result:
            cx += process_cx
            cy += process_cy

        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return test_statistics(cx, cy, epsilon, iterations)
