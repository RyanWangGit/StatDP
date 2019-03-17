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
from statdp.algorithms import *


def test_noisymax():
    # add no noise to the array
    assert noisy_max_v1a([1, 2, 1], float('inf')) == 1
    assert noisy_max_v1b([1, 3, 1], float('inf')) == 3
    assert noisy_max_v2a([1, 3, 1], float('inf')) == 1
    assert noisy_max_v2b([1, 3, 1], float('inf')) == 3


def test_sparsevector():
    assert SVT([1, 2, 3, 4], float('inf'), 1, 2.5) == 2
    assert iSVT1([1, 2, 3, 4], float('inf'), 1, 1.5) == 3
    assert iSVT1([1, 2, 3, 4], float('inf'), 1, 3.5) == 3
    assert iSVT1([1, 2, 3, 4], float('inf'), 1, 2.5) == 4
    assert iSVT1([4, 3, 2, 1], float('inf'), 1, 2.5) == 0
    assert iSVT2([1, 2, 3, 4], float('inf'), 1, 1.5) == 3
    assert iSVT2([1, 2, 3, 4], float('inf'), 1, 3.5) == 3
    assert iSVT3([1, 2, 3, 4], float('inf'), 1, 1.5) == 3
    assert iSVT3([1, 2, 3, 4], float('inf'), 1, 3.5) == 3
    assert iSVT4([1, 2, 3, 4], float('inf'), 1, 2) == (2, 3.0)


def test_histogram():
    assert histogram([1, 2], float('inf')) == 1
    assert isinstance(histogram([1, 2], 1), float)
    assert histogram_eps([1, 2], 0) == 1
    assert isinstance(histogram_eps([1, 2], 1), float)


def test_laplace_mechanism():
    assert laplace_mechanism([1, 2, 3], float('inf')) == 1
