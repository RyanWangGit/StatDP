import numpy as np
import multiprocessing as mp
import math
import functools
from scipy import stats


def _hypergeometric(cx, cy, iterations):
    # survival function = 1 - cdf, "sometimes more accurate" according to
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.hypergeom.html
    return stats.hypergeom.sf(cx, 2 * iterations, iterations, cx + cy)


def _run_algorithm(algorithm, d1, d2, kwargs, event, iterations):
    np.random.seed()
    cx = sum(1 for _ in range(iterations) if algorithm(d1, **kwargs) in event)
    cy = sum(1 for _ in range(iterations) if algorithm(d2, **kwargs) in event)
    return cx, cy


def test_statistics(cx, cy, epsilon, iterations, process_pool=None):
    """ Calculate p-value based on observed results.
    :param cx: The observed count of running algorithm with database 1 that falls into the event
    :param cy:The observed count of running algorithm with database 2 that falls into the event
    :param epsilon: The epsilon to test for.
    :param iterations: The total iterations for running algorithm.
    :param process_pool: The process pool to run on, run in single core if None
    :return: p-value
    """
    # average p value
    if process_pool is None:
        return np.mean(tuple(_hypergeometric(cx, cy, iterations)
                             for cx in np.random.binomial(cx, 1.0 / (np.exp(epsilon)), 1000)))
    else:
        # bind cy and iterations to _hypergeometric function and feed different cx into it
        return np.mean(process_pool.map(functools.partial(_hypergeometric, cy=cy, iterations=iterations),
                                        np.random.binomial(cx, 1.0 / (np.exp(epsilon)), 1000),
                                        chunksize=int(1000 / mp.cpu_count())))


def hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, iterations, process_pool=None):
    """ Run hypothesis tests on given input and events.
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
    np.random.seed()
    if process_pool is None:
        cx, cy = _run_algorithm(algorithm, d1, d2, kwargs, event, iterations)
        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
    else:
        process_iterations = [int(math.floor(float(iterations) / mp.cpu_count())) for _ in range(mp.cpu_count())]
        # add the remaining iterations to the last index
        process_iterations[mp.cpu_count() - 1] += iterations % process_iterations[mp.cpu_count() - 1]

        result = process_pool.map(functools.partial(_run_algorithm, algorithm, d1, d2, kwargs, event),
                                  process_iterations)

        cx, cy = sum(process_cx for process_cx, _ in result), sum(process_cy for _, process_cy in result)
        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
