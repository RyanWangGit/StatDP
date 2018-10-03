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
    def __init__(self, algorithm, kwargs, D1, D2, event):
        self.algorithm = algorithm
        self.kwargs = kwargs
        self.D1 = D1
        self.D2 = D2
        self.event = event

    def __call__(self, iterations):
        return self.run(iterations)

    def run(self, iterations):
        np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 16))
        cx = sum(1 for _ in range(iterations) if self.algorithm(self.D1, **self.kwargs) in self.event)
        cy = sum(1 for _ in range(iterations) if self.algorithm(self.D2, **self.kwargs) in self.event)
        return cx, cy


def test_statistics(cx, cy, epsilon, iterations, process_pool=None):
    if process_pool is None:
        return np.mean(tuple(map(__HyperGeometric(cy, iterations),
                                 np.random.binomial(cx, 1.0 / (np.exp(epsilon)), 1000))))
    else:
        # use a multiprocessing.Pool to parallel average p value calculation
        return np.mean(process_pool.map(__HyperGeometric(cy, iterations),
                                        np.random.binomial(cx, 1.0 / (np.exp(epsilon)), 1000),
                                        chunksize=int(1000 / mp.cpu_count())))


def hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, iterations, process_pool=None):
    """
    :param algorithm: The algorithm to run on
    :param kwargs: The keyword arguments the algorithm needs
    :param d1: Database 1
    :param d2: Database 2
    :param event: The event set
    :param iterations: Number of iterations to run
    :param epsilon: The epsilon value to test for
    :param process_pool: The process pool to use, run with single process if None
    :return: p values
    """
    np.random.seed(int(codecs.encode(os.urandom(4), 'hex'), 16))
    if process_pool is None:
        cx, cy = __RunAlgorithm(algorithm, kwargs, d1, d2, event).run(iterations)
        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
    else:
        process_iterations = [int(math.floor(float(iterations) / mp.cpu_count())) for _ in range(mp.cpu_count())]
        # add the remaining iterations to the last index
        process_iterations[mp.cpu_count() - 1] += iterations % process_iterations[mp.cpu_count() - 1]

        result = process_pool.map(__RunAlgorithm(algorithm, kwargs, d1, d2, event), process_iterations)

        cx, cy = 0, 0
        for process_cx, process_cy in result:
            cx += process_cx
            cy += process_cy

        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
