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
