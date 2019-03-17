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
import logging
import multiprocessing as mp

import tqdm

from statdp.generators import generate_arguments, generate_databases
from statdp.hypotest import hypothesis_test
from statdp.selectors import select_event

logger = logging.getLogger(__name__)


def detect_counterexample(algorithm, test_epsilon, default_kwargs=None, databases=None, num_input=(5, 10),
                          event_iterations=100000, detect_iterations=500000, cores=0,
                          quiet=False, loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument.
    :param databases: The databases to run for detection, optional.
    :param num_input: The length of input to generate, not used if database param is specified.
    :param event_iterations: The iterations for event selector to run, default is 100000.
    :param detect_iterations: The iterations for detector to run, default is 500000.
    :param cores: The cores to utilize, 0 means auto-detection.
    :param quiet: Do not print progress bar or messages, logs are not affected, default is False.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
    # initialize an empty default kwargs if None is given
    default_kwargs = default_kwargs if default_kwargs else {}

    logging.basicConfig(level=loglevel)
    logger.info('Starting to find counterexample on algorithm {} with test epsilon {}'
                .format(algorithm.__name__, test_epsilon))
    logger.info('Options -> default_kwargs: {} | databases: {} | cores:{}'.format(default_kwargs, databases, cores))

    input_list = []
    if databases is not None:
        d1, d2 = databases
        kwargs = generate_arguments(algorithm, d1, d2, default_kwargs=default_kwargs)
        input_list = ((d1, d2, kwargs),)
    else:
        num_input = (int(num_input), ) if isinstance(num_input, (int, float)) else num_input
        for num in num_input:
            input_list.extend(generate_databases(algorithm, num, default_kwargs=default_kwargs))

    result = []

    # convert int/float or iterable into tuple (so that it has length information)
    test_epsilon = (test_epsilon, ) if isinstance(test_epsilon, (int, float)) else test_epsilon
    if isinstance(test_epsilon, (int, float)):
        test_epsilon = (test_epsilon, )
    elif not isinstance(test_epsilon, (tuple, list)):
        test_epsilon = tuple(test_epsilon)

    pool = mp.Pool(mp.cpu_count()) if cores == 0 else (mp.Pool(cores) if cores != 1 else None)
    try:
        for i, epsilon in tqdm.tqdm(enumerate(test_epsilon), total=len(test_epsilon), unit='test', desc='Detection'):
            d1, d2, kwargs, event = select_event(algorithm, input_list, epsilon, event_iterations, quiet=quiet,
                                                 process_pool=pool)
            p = hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, detect_iterations,
                                report_p2=False, process_pool=pool)
            result.append((epsilon, p, d1, d2, kwargs, event))
            tqdm.tqdm.write('Epsilon: {} | p-value: {:5.3f} | Event: {}'
                            .format(epsilon, p, event))
            logger.debug('D1: {} | D2: {} | kwargs: {}'.format(d1, d2, kwargs))
    finally:
        if pool:
            pool.close()
            pool.join()
        else:
            pass

    return result
