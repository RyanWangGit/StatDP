from inspect import isfunction
import numpy as np
from collections import Counter
from intervals import Interval
import logging
import functools

logger = logging.getLogger(__name__)


def _evaluate_event(event, result_d1, result_d2):
    cx = sum(1 for x in result_d1 if x in event)
    cy = sum(1 for y in result_d2 if y in event)
    cx, cy = (cx, cy) if cx > cy else (cy, cx)
    return cx, cy


def select_event(algorithm, input_list, epsilon, iterations=100000, search_space=None, process_pool=None):
    """
    :param algorithm: The algorithm to run on
    :param input_list: list of (d1, d2, kwargs) input pair for the algorithm to run
    :param epsilon: Test epsilon value
    :param iterations: The iterations to run algorithms
    :param search_space: The result search space to run on, auto-determine based on return type if None
    :param process_pool: The process pool to use, run with single process if None
    :return: (d1, d2, kwargs, event) pair which has minimum p value from search space
    """
    assert isfunction(algorithm)
    from .hypotest import test_statistics

    input_event_pairs = []
    p_values = []

    for (d1, d2, kwargs) in input_list:
        result_d1 = [algorithm(d1, **kwargs) for _ in range(iterations)]
        result_d2 = [algorithm(d2, **kwargs) for _ in range(iterations)]

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

        threshold = 0.001 * iterations * np.exp(epsilon)

        # bind the result_d1 and result_d2 to _evaluate_event, leaving out `event` argument to be filled
        partial_evaluate_event = functools.partial(_evaluate_event, result_d1=result_d1, result_d2=result_d2)

        results = list(map(partial_evaluate_event, search_space)) if process_pool is None else \
            process_pool.map(partial_evaluate_event, search_space)

        input_p_values = [test_statistics(cx, cy, epsilon, iterations)
                          if cx + cy > threshold else float('inf') for (cx, cy) in results]

        for (s, (cx, cy), p) in zip(search_space, results, input_p_values):
            logger.debug('d1: %s | d2: %s | event: %s | p: %f | cx: %d | cy: %d | ratio: %f' %
                         (d1, d2, s, p, cx, cy, float(cy) / cx if cx != 0 else float('inf')))

        input_event_pairs.extend(list((d1, d2, kwargs, event) for event in search_space))
        p_values.extend(input_p_values)

    # find an (d1, d2, kwargs, event) pair which has minimum p value from search space
    return input_event_pairs[np.argmin(p_values)]
