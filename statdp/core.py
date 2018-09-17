from statdp.generators import simple_generator, argument_generator
from statdp.hypotest import hypothesis_test
from statdp.selectors import select_event
import logging
logger = logging.getLogger(__name__)


def detect_counterexample(algorithm, test_epsilon, default_kwargs,
                          event_search_space=None, databases=None,
                          event_iterations=100000, detect_iterations=500000, cores=0,
                          loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument, 'epsilon' must be provided.
    :param event_search_space: The search space for event selector to reduce search time, optional.
    :param databases: The databases to run for detection, optional.
    :param event_iterations: The iterations for event selector to run, default is 100000.
    :param detect_iterations: The iterations for detector to run, default is 500000.
    :param cores: The cores to utilize, 0 means auto-detection.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
    logging.basicConfig(level=loglevel)
    logger.info('Starting to find counter example on algorithm {} with test epsilon {}\n'
                .format(algorithm.__name__, test_epsilon))
    logger.info('\nExtra arguments:\n'
                'default_kwargs: {}\n'
                'event_search_space: {}\n'
                'databases: {}\n'
                'cores:{}\n'.format(default_kwargs, event_search_space, databases, cores))

    if databases is not None:
        d1, d2 = databases
        kwargs = argument_generator(algorithm, d1, d2, default_kwargs=default_kwargs)
        input_list = ((d1, d2, kwargs),)
    else:
        input_list = simple_generator(algorithm, 5, default_kwargs=default_kwargs)

    result = []

    test_epsilon = (test_epsilon, ) if isinstance(test_epsilon, (int, float)) else test_epsilon

    for i, epsilon in enumerate(test_epsilon):
        d1, d2, kwargs, event = select_event(algorithm, input_list, epsilon, event_iterations,
                                             search_space=event_search_space, cores=cores)

        # fix the database and arguments if selected for performance
        input_list = ((d1, d2, kwargs),) if len(input_list) > 1 else input_list

        p1, _ = hypothesis_test(algorithm, kwargs, d1, d2, event, epsilon, detect_iterations, cores=cores)
        result.append((epsilon, p1, d1, d2, kwargs, event))
        print('Epsilon: {} | p-value: {:5.3f} | Event: {} | {:5.1f}%'
              .format(epsilon, p1, event, float(i + 1) / len(test_epsilon) * 100))
        logger.debug('D1: {} | D2: {} | kwargs: {}'.format(d1, d2, kwargs))

    return result




