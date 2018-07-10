from inspect import isfunction
import multiprocessing as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)


def frequency_s_selector(algorithm, args, kwargs, D1, D2, iterations=10000):
    assert isfunction(algorithm)
    from collections import Counter
    a = [algorithm(D1, *args, **kwargs) for _ in range(iterations)]
    b = [algorithm(D2, *args, **kwargs) for _ in range(iterations)]

    countitem = Counter(a + b).most_common(2)

    return [countitem[0][0], countitem[1][0]]


def difference_s_selector(algorithm, args, kwargs, D1, D2, iterations=10000):
    assert isfunction(algorithm)
    from collections import Counter
    a = [algorithm(D1, *args, **kwargs) for _ in range(iterations)]
    b = [algorithm(D2, *args, **kwargs) for _ in range(iterations)]
    count_a, count_b = Counter(a), Counter(b)

    # make a contain all keys(a may not contain all b's keys)
    for key in count_b.keys():
        if key not in count_a:
            count_a[key] = 0

    diff = []
    for key, item in count_a.items():
        diff.append((key, abs(item - count_b[key])))

    diff = sorted(diff, key=lambda k: k[1], reverse=True)

    return [diff[i][0] for i in range(1)]


def sd_s_selector(algorithm, args, kwargs, D1, D2, iterations=10000):
    assert isfunction(algorithm)
    a = [algorithm(D1, *args, **kwargs) for _ in range(iterations)]
    b = [algorithm(D2, *args, **kwargs) for _ in range(iterations)]
    c = 10

    Scand = list(range(0, len(D1)))
    max = 0
    maxi = 0

    for i in Scand:
        p = (a.count(i) + c) / (len(a) + c)
        q = (b.count(i) + c) / (len(b) + c)

        if p == 0 and q == 0:
            sddiff = 0
        else:
            sddiff = abs((p - q) / ((p * (1 - p) + q * (1 - q)) ** (1 / 2)))

        if max < sddiff:
            max = sddiff
            maxi = i

    return [maxi]


class __EvaluateS:
    def __init__(self, a, b, epsilon, iterations):
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.iterations = iterations

    def __call__(self, s):
        cx = sum(1 for x in self.a if x in s)
        cy = sum(1 for y in self.b if y in s)
        cx, cy = (cx, cy) if cx > cy else (cy, cx)
        return cx, cy


_process_pool = mp.Pool(mp.cpu_count())


def fisher_s_selector(algorithm, args, kwargs, D1, D2, epsilon, iterations=100000, search_space=(), cores=0):
    assert isfunction(algorithm)
    from .core import test_statistics
    import math

    a = [algorithm(D1, *args, **kwargs) for _ in range(iterations)]
    b = [algorithm(D2, *args, **kwargs) for _ in range(iterations)]

    global _process_pool

    # find S which has minimum p value from search space
    threshold = 0.001 * iterations * np.exp(epsilon)

    results = list(map(__EvaluateS(a, b, epsilon, iterations), search_space)) if cores == 1 \
        else _process_pool.map(__EvaluateS(a, b, epsilon, iterations), search_space)

    p_values = [test_statistics(x[0], x[1], epsilon, iterations)
                if x[0] + x[1] > threshold else math.inf for x in results]

    for i, (s, (cx, cy), p) in enumerate(zip(search_space, results, p_values)):
        logger.debug('S: %s p: %f cx: %d cy: %d ratio: %f' % (s, p, cx, cy, float(cy) / cx if cx != 0 else math.inf))

    return search_space[np.argmin(p_values)]
