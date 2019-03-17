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

from flaky import flaky

from statdp.algorithms import (SVT, iSVT1, iSVT2, iSVT3, iSVT4, noisy_max_v1a,
                               noisy_max_v1b, noisy_max_v2a, noisy_max_v2b)
from statdp.core import detect_counterexample


def assert_correct_algorithm(algorithm, kwargs=None, num_input=5):
    if kwargs and isinstance(kwargs, dict):
        kwargs.update({'epsilon': 0.7})
    else:
        kwargs = {'epsilon': 0.7}
    result = detect_counterexample(algorithm, (0.6, 0.7, 0.8), kwargs, num_input=num_input, loglevel=logging.DEBUG)
    assert isinstance(result, list) and len(result) == 3
    epsilon, p, *extras = result[0]
    assert p <= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)
    epsilon, p, *extras = result[1]
    assert p >= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)
    epsilon, p, *extras = result[2]
    assert p >= 0.95, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)


def assert_incorrect_algorithm(algorithm, kwargs=None, num_input=5):
    if kwargs and isinstance(kwargs, dict):
        kwargs.update({'epsilon': 0.7})
    else:
        kwargs = {'epsilon': 0.7}
    result = detect_counterexample(algorithm, 0.7, kwargs, num_input=num_input, loglevel=logging.DEBUG)
    assert isinstance(result, list) and len(result) == 1
    epsilon, p, *extras = result[0]
    assert p <= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(epsilon, p, extras)


def test_noisy_max_v1a():
    assert_correct_algorithm(noisy_max_v1a)


def test_noisy_max_v1b():
    assert_incorrect_algorithm(noisy_max_v1b)


def test_noisy_max_v2a():
    assert_correct_algorithm(noisy_max_v2a)


def test_noisy_max_v2b():
    assert_incorrect_algorithm(noisy_max_v2b)

# SVT sometimes may fail, retry 5 times claim failure
@flaky(max_runs=5)
def test_SVT():
    assert_correct_algorithm(SVT, {'N': 1, 'T': 0.5}, num_input=10)


def test_iSVT1():
    assert_incorrect_algorithm(iSVT1, {'N': 1, 'T': 1}, num_input=10)


def test_iSVT2():
    assert_incorrect_algorithm(iSVT2, {'N': 1, 'T': 1}, num_input=10)


def test_iSVT3():
    assert_incorrect_algorithm(iSVT3, {'N': 1, 'T': 1}, num_input=10)


def test_iSVT4():
    assert_incorrect_algorithm(iSVT4, {'N': 1, 'T': 1}, num_input=10)
