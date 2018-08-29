from inspect import isfunction
import multiprocessing as mp
import numpy as np
from collections import Counter
from intervals import Interval
import logging

logger = logging.getLogger(__name__)


class __EvaluateEvent:
    def __init__(self, a, b, iterations):
        self.a = a
        self.b = b
        self.iterations = iterations

    def __call__(self, s):
        cx = sum(1 for x in self.a if x in s)
        cy = sum(1 for y in self.b if y in s)
        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return cx, cy


_process_pool = mp.Pool(mp.cpu_count())


def select_event(algorithm, input_list, epsilon, iterations=100000, search_space=None, cores=0):
    assert isfunction(algorithm)
    from .core import test_statistics

    result_d1 = [algorithm(D1, *args, **kwargs) for _ in range(iterations)]
    result_d2 = [algorithm(D2, *args, **kwargs) for _ in range(iterations)]

    if search_space is None:
        # determine the search space based on the return type
        # a subset of results to determine return type
        sub_result = result_d1 + result_d2
        counter = Counter(sub_result)

        # categorical output
        if len(counter) < iterations * 0.02 * 0.1:
            search_space = tuple((key,) for key in counter.keys())
        else:
            sub_result_sorted = np.sort(sub_result)
            average = np.average(sub_result_sorted)
            idx = np.searchsorted(sub_result_sorted, average, side='left')
            # find the densest 70% range
            search_min = int(idx - 0.35 * len(sub_result_sorted)) if int(idx - 0.4 * len(sub_result_sorted)) > 0 else 0
            search_max = int(0.7 * len(sub_result_sorted) - (idx - search_min))

            search_space = tuple(Interval((-float('inf'), alpha)) for alpha in
                                 np.linspace(sub_result_sorted[search_min], sub_result_sorted[search_max], num=25))

    logger.info('search space is set to {0}'.format(search_space))

    global _process_pool

    # find an event which has minimum p value from search space
    threshold = 0.001 * iterations * np.exp(epsilon)

    results = list(map(__EvaluateEvent(result_d1, result_d2, iterations), search_space)) if cores == 1 \
        else _process_pool.map(__EvaluateEvent(result_d1, result_d2, iterations), search_space)

    p_values = [test_statistics(x[0], x[1], epsilon, iterations)
                if x[0] + x[1] > threshold else float('inf') for x in results]

    for i, (s, (cx, cy), p) in enumerate(zip(search_space, results, p_values)):
        logger.debug('event: %s | p: %f | cx: %d | cy: %d | ratio: %f' %
                     (s, p, cx, cy, float(cy) / cx if cx != 0 else float('inf')))

    return search_space[np.argmin(p_values)]
