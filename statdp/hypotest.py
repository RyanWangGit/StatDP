import numpy as np
import multiprocessing as mp
import math
import functools
from scipy import stats


def _hypergeometric(cx, cy, iterations):
    return 1 - stats.hypergeom.cdf(cx, 2 * iterations, iterations, cx + cy)


def _run_algorithm(algorithm, d1, d2, kwargs, event, iterations):
    np.random.seed()
    result_d1 = np.fromiter((algorithm(d1, **kwargs) for _ in range(iterations)), dtype=np.float64)
    result_d2 = np.fromiter((algorithm(d2, **kwargs) for _ in range(iterations)), dtype=np.float64)
    cx = np.count_nonzero(result_d1 == event if isinstance(event, (int, float)) else
                          np.logical_and(result_d1 > event[0], result_d1 < event[1]))
    cy = np.count_nonzero(result_d2 == event if isinstance(event, (int, float)) else
                          np.logical_and(result_d2 > event[0], result_d2 < event[1]))
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


def hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, iterations, report_p2=True, process_pool=None):
    """ Run hypothesis tests on given input and events.
    :param algorithm: The algorithm to run on
    :param kwargs: The keyword arguments the algorithm needs
    :param d1: Database 1
    :param d2: Database 2
    :param event: The event set
    :param iterations: Number of iterations to run
    :param epsilon: The epsilon value to test for
    :param report_p2: The boolean to whether report p2 or not
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
        if report_p2:
            return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
        else:
            return test_statistics(cx, cy, epsilon, iterations)
