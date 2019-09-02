# MIT License
#
# Copyright (c) 2018-2019 Yuxin Wang
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
import logging
import math
import multiprocessing as mp

import numpy as np

from statdp.core import run_algorithm
import statdp._hypergeom as hypergeom

logger = logging.getLogger(__name__)


def _hypergeometric(cx, cy, iterations):
    # here we use `cx - 1` because pvalue should be P(random variable >= test_statistic) rather than > test_statistic
    return 1 - hypergeom.cdf(cx - 1, 2 * iterations, iterations, cx + cy)


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
    sample_num = 200
    if process_pool is None or hypergeom.use_gsl:
        return np.fromiter((_hypergeometric(cx, cy, iterations)
                            for cx in np.random.binomial(cx, 1.0 / (np.exp(epsilon)), sample_num)),
                           dtype=np.float64, count=sample_num).mean()
    else:
        # bind cy and iterations to _hypergeometric function and feed different cx into it
        return np.fromiter(process_pool.imap_unordered(functools.partial(_hypergeometric, cy=cy, iterations=iterations),
                                                       np.random.binomial(cx, 1.0 / (np.exp(epsilon)), sample_num),
                                                       chunksize=int(sample_num / mp.cpu_count())),
                           dtype=np.float64, count=sample_num).mean()


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
    if process_pool is None:
        ((cx, cy), *_), _ = run_algorithm(algorithm, d1, d2, kwargs, event, iterations)
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
        for ((local_cx, local_cy), *_), _ in process_pool.imap_unordered(
                functools.partial(run_algorithm, algorithm, d1, d2, kwargs, event), process_iterations):
            cx += local_cx
            cy += local_cy
        cx, cy = (cx, cy) if cx > cy else (cy, cx)

        # calculate and return p value
        if report_p2:
            return test_statistics(cx, cy, epsilon, iterations, process_pool), \
                   test_statistics(cy, cx, epsilon, iterations, process_pool)
        else:
            return test_statistics(cx, cy, epsilon, iterations, process_pool)
