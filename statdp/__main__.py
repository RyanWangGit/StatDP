from .generators import simple_generator, argument_generator
from .core import hypothesis_test
from .selectors import select_event
import logging
logger = logging.getLogger(__name__)


def detect_counter_example(algorithm, test_epsilon, default_kwargs,
                           event_search_space=None, databases=None,
                           event_iterations=100000, detect_iterations=500000, cores=0,
                           loglevel=logging.INFO):
    logging.basicConfig(loglevel=loglevel)
    logger.info('Starting to find counter example on algorithm {0} with test epsilon {1}\n'
                .format(algorithm.__name__, test_epsilon))
    logger.info('Extra arguments:\n'
                '\t{0}\n\t{1}\n\t{2}\n\t{3}\n'.format(default_kwargs, event_search_space, databases, cores))

    if databases is not None:
        d1, d2 = databases
        kwargs = argument_generator(algorithm, d1, d2, default_kwargs=default_kwargs)
        input_list = ((d1, d2, kwargs),)
    else:
        input_list = simple_generator(algorithm, 5, default_kwargs=default_kwargs)

    result = []

    test_epsilon = (test_epsilon, ) if isinstance(test_epsilon, (int, float)) else test_epsilon

    for epsilon in test_epsilon:
        d1, d2, kwargs, event = select_event(algorithm, input_list, test_epsilon, event_iterations,
                                             search_space=event_search_space, cores=cores)

        # fix the database and arguments if selected for performance
        input_list = ((d1, d2, kwargs),) if len(input_list) > 1 else input_list

        p1, _ = hypothesis_test(algorithm, kwargs, d1, d2, event, test_epsilon, detect_iterations, cores=cores)
        result.append((epsilon, p1))

    return result




