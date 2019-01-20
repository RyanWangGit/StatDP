# MIT License
#
# Copyright (c) 2018 Ryan Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import functools
import math
import multiprocessing as mp

import numpy as np
from scipy import stats


def _hypergeometric(cx, cy, iterations):
    return 1 - stats.hypergeom.cdf(cx, 2 * iterations, iterations, cx + cy)


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


def _run_algorithm(algorithm, d1, d2, kwargs, event, iterations):
    np.random.seed()
    result_d1 = np.fromiter((algorithm(d1, **kwargs) for _ in range(iterations)), dtype=np.float64, count=iterations)
    result_d2 = np.fromiter((algorithm(d2, **kwargs) for _ in range(iterations)), dtype=np.float64, count=iterations)
    cx = np.count_nonzero(result_d1 == event if isinstance(event, (int, float)) else
                          np.logical_and(result_d1 > event[0], result_d1 < event[1]))
    cy = np.count_nonzero(result_d2 == event if isinstance(event, (int, float)) else
                          np.logical_and(result_d2 > event[0], result_d2 < event[1]))
    return cx, cy


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
        if report_p2:
            return test_statistics(cx, cy, epsilon, iterations), test_statistics(cy, cx, epsilon, iterations)
        else:
            return test_statistics(cx, cy, epsilon, iterations)
    else:
        process_iterations = [int(math.floor(float(iterations) / mp.cpu_count())) for _ in range(mp.cpu_count())]
        # add the remaining iterations to the last index
        process_iterations[mp.cpu_count() - 1] += iterations % process_iterations[mp.cpu_count() - 1]

        # start the pool to run the algorithm and collects the statistics
        cx, cy = 0, 0
        for local_cx, local_cy in process_pool.imap_unordered(
                functools.partial(_run_algorithm, algorithm, d1, d2, kwargs, event), process_iterations):
            cx += local_cx
            cy += local_cy
        cx, cy = (cx, cy) if cx > cy else (cy, cx)

        # calculate and return p value
        if report_p2:
            return test_statistics(cx, cy, epsilon, iterations, process_pool), \
                   test_statistics(cy, cx, epsilon, iterations, process_pool)
        else:
            return test_statistics(cx, cy, epsilon, iterations, process_pool)
