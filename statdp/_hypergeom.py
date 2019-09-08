# MIT License
#
# Copyright (c) 2019 Yuxin Wang
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
import os
import sys
import subprocess
import shutil
import ctypes
import logging

from scipy.stats import hypergeom

logger = logging.getLogger(__name__)

cdf = None
use_gsl = True

# try to import hypergeometric cdf from Gnu Scientific Library (GSL)
if shutil.which('gsl-config'):
    proc = subprocess.run(['gsl-config', '--prefix'], stdout=subprocess.PIPE)

    # try to find the library file
    lib_path = os.path.join(proc.stdout.decode('utf-8').strip(), 'lib')
    ext = '.dylib' if 'darwin' in sys.platform else ('.so' if 'linux' in sys.platform else '.dll')
    if not os.path.exists(os.path.join(lib_path, 'libgslcblas{}'.format(ext))):
        lib_path = os.path.join(lib_path, 'x86_64-linux-gnu')
        if not os.path.exists(os.path.join(lib_path, 'libgslcblas{}'.format(ext))):
            # libgsl not found
            lib_path = None

    if lib_path:
        dll = ctypes.CDLL if 'darwin' in sys.platform or 'linux' in sys.platform else ctypes.WinDLL
        # load libgslcblas first
        dll(os.path.join(lib_path, 'libgslcblas{}'.format(ext)), mode=ctypes.RTLD_GLOBAL)
        # load libgsl
        libgsl = dll(os.path.join(lib_path, 'libgsl{}'.format(ext)))
        hyper = libgsl.gsl_cdf_hypergeometric_P
        hyper.restype = ctypes.c_double

        def cdf(k, M, n, N):
            return float(hyper(ctypes.c_int(k), ctypes.c_int(n), ctypes.c_int(M - n), ctypes.c_int(N)))

# if failed to load GSL, fall back to scipy implementation
if not cdf:
    use_gsl = False
    cdf = hypergeom.cdf
