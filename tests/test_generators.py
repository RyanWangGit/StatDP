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
from statdp.algorithms import noisy_max_v1a
from statdp.generators import generate_arguments, generate_databases


def test_generate_databases():
    input_list = generate_databases(noisy_max_v1a, 5, {'epsilon': 0.5})
    assert isinstance(input_list, (list, tuple)) and len(input_list) >= 1
    for input_ in input_list:
        assert isinstance(input_, (list, tuple)) and len(input_) == 3
        d1, d2, args = input_
        assert isinstance(d1, (tuple, list)) and isinstance(d2, (tuple, list))
        assert len(d1) == 5 and len(d2) == 5
        assert isinstance(args, (tuple, list, dict))


def test_generate_arguments():
    d1, d2 = tuple(1 for _ in range(5)), tuple(2 for _ in range(5))
    assert generate_arguments(noisy_max_v1a, d1, d2, {}) is None
