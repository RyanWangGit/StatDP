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
import numpy as np
from itertools import zip_longest


def _argmax(iterable):
    # implement numpy.argmax in pure python, faster if iterable is plain python list
    return max(enumerate(iterable), key=lambda t: t[1])[0]


def _hamming_distance(result1, result2):
    # implement hamming distance in pure python, faster than np.count_zeros if inputs are plain python list
    return sum(res1 != res2 for res1, res2 in zip_longest(result1, result2))


def noisy_max_v1a(queries, epsilon):
    # find the largest noisy element and return its index
    return _argmax(query + np.random.laplace(scale=2.0 / epsilon) for query in queries)


def noisy_max_v1b(queries, epsilon):
    return max(query + np.random.laplace(scale=2.0 / epsilon) for query in queries)


def noisy_max_v2a(queries, epsilon):
    return _argmax(query + np.random.exponential(scale=2.0 / epsilon) for query in queries)


def noisy_max_v2b(queries, epsilon):
    return max(query + np.random.exponential(scale=2.0 / epsilon) for query in queries)


def histogram_eps(queries, epsilon):
    noisy_array = tuple(query + np.random.laplace(scale=epsilon) for query in queries)
    return noisy_array[0]


def histogram(queries, epsilon):
    noisy_array = tuple(query + np.random.laplace(scale=1.0 / epsilon) for query in queries)
    return noisy_array[0]


def laplace_mechanism(queries, epsilon):
    noisy_array = tuple(query + np.random.laplace(scale=len(queries)/epsilon) for query in queries)
    lower = 1 - 0.27
    upper = 1 + 0.75
    return sum(1 for element in noisy_array if lower <= element <= upper)


def SVT(queries, epsilon, N, T):
    out = []
    eta1 = np.random.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = np.random.laplace(scale=4.0 * N / epsilon)
        if query + eta2 >= noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out.count(False)


def iSVT1(queries, epsilon, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=2.0 * delta / epsilon)
    noisy_T = T + eta1
    for query in queries:
        eta2 = 0
        if (query + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)

    true_count = int(len(queries) / 2)
    return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


def iSVT2(queries, epsilon, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=2.0 * delta / epsilon)
    noisy_T = T + eta1
    for query in queries:
        eta2 = np.random.laplace(scale=2.0 * delta / epsilon)
        if (query + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)

    true_count = int(len(queries) / 2)
    return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


def iSVT3(queries, epsilon, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=4.0 * delta / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = np.random.laplace(scale=(4.0 * delta) / (3.0 * epsilon))
        if query + eta2 > noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)

    true_count = int(len(queries) / 2)
    return _hamming_distance((True if i < true_count else False for i in range(len(queries))), out)


def iSVT4(queries, epsilon, N, T):
    out = []
    eta1 = np.random.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = np.random.laplace(scale=2.0 * N / epsilon)
        if query + eta2 > noisy_T:
            out.append(query + eta2)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out.count(False)
