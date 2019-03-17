# MIT License
#
# Copyright (c) 2018 Yuxin (Ryan) Wang
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
import inspect

import numpy as np
import tqdm

from statdp.hypotest import test_statistics, run_algorithm

logger = logging.getLogger(__name__)


def _evaluate_input(input_triplet, algorithm, iterations):
    d1, d2, kwargs = input_triplet
    return run_algorithm(algorithm, d1, d2, kwargs, None, iterations)


def select_event(algorithm, input_list, epsilon, iterations=100000, process_pool=None, quiet=False):
    """
    :param algorithm: The algorithm to run on
    :param input_list: list of (d1, d2, kwargs) input pair for the algorithm to run
    :param epsilon: Test epsilon value
    :param iterations: The iterations to run algorithms
    :param process_pool: The process pool to use, run with single process if None
    :param quiet: Do not print progress bar or messages, logs are not affected, default is False.
    :return: (d1, d2, kwargs, event) pair which has minimum p value from search space.
    """
    if not inspect.isfunction(algorithm):
        raise ValueError('Algorithm must be runnable')

    # fill in other arguments for _evaluate_input function, leaving out `input` to be filled
    partial_evaluate_input = functools.partial(_evaluate_input,
                                               algorithm=algorithm, iterations=iterations)

    results = process_pool.imap_unordered(partial_evaluate_input, input_list) if process_pool else \
        map(partial_evaluate_input, input_list)

    counts, input_event_pairs = [], []
    # flatten the results for all input/event pairs
    for local_counts, local_input_event_pair in results:
        counts.extend(local_counts)
        input_event_pairs.extend(local_input_event_pair)

    # calculate p-values based on counts
    threshold = 0.001 * iterations * np.exp(epsilon)
    p_values_generator = (test_statistics(cx, cy, epsilon, iterations, process_pool=process_pool)
                          if cx + cy > threshold else float('inf') for (cx, cy) in counts)

    # wrap the tqdm around the generator for progress information
    with tqdm.tqdm(p_values_generator, desc='Evaluating events', total=len(counts), unit='event', disable=quiet) \
            as wrapper:
        input_p_values = np.fromiter(wrapper, dtype=np.float64, count=len(counts))

    # log the information for debug purposes
    for ((d1, d2, kwargs, event), (cx, cy), p) in zip(input_event_pairs, counts, input_p_values):
        logger.debug('d1: {} | d2: {} | kwargs: {} | event: {} | p-value: {:5.3f} | cx: {} | cy: {} | ratio: {:5.3f}'
                     .format(d1, d2, kwargs, event, p, cx, cy, float(cy) / cx if cx != 0 else float('inf')))

    # find an (d1, d2, kwargs, event) pair which has minimum p value from search space
    return input_event_pairs[input_p_values.argmin()]
