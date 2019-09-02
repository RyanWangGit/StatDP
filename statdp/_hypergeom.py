import os
import sys
import subprocess
import shutil
import ctypes
import logging

from scipy.stats import hypergeom

logger = logging.getLogger(__name__)

# try to import hypergeometric cdf from Gnu Scientific Library (GSL)
cdf = None
use_gsl = True
if shutil.which('gsl-config'):
    proc = subprocess.run(['gsl-config', '--prefix'], stdout=subprocess.PIPE)
    lib_path = os.path.join(proc.stdout.decode('utf-8').strip(), 'lib')
    ext = '.dylib' if 'darwin' in sys.platform else ('.so' if 'linux' in sys.platform else '.dll')
    lib_path = os.path.join(lib_path, 'x86_64-linux-gnu') if 'linux' in sys.platform else lib_path
    # load libgslcblas first
    ctypes.CDLL(os.path.join(lib_path, 'libgslcblas{}'.format(ext)), mode=ctypes.RTLD_GLOBAL)
    # load libgsl
    libgsl = ctypes.CDLL(os.path.join(lib_path, 'libgsl{}'.format(ext)))
    hyper = libgsl.gsl_cdf_hypergeometric_P
    hyper.restype = ctypes.c_double

    def cdf(k, M, n, N):
        return float(hyper(ctypes.c_int(k), ctypes.c_int(n), ctypes.c_int(M - n), ctypes.c_int(N)))

# fall back to scipy implementation
if not cdf:
    use_gsl = False
    cdf = hypergeom.cdf
