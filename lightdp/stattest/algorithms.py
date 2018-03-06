# TODO: the whole package should be removed in the future since we should have user-input algorithms

import numpy as np


def noisy_max_v1a(Q, eps):
    # add laplace noise
    noisy_array = [a + np.random.laplace(scale=2.0 / eps) for a in Q]

    # find the largest noisy element and return its index
    return np.argmax(noisy_array)


def noisy_max_v1b(Q, eps):
    noisy_array = [a + np.random.laplace(scale=2.0 / eps) for a in Q]
    return max(noisy_array)


def noisy_max_v2a(Q, eps):
    noisy_array = [a + np.random.exponential(scale=2.0 / eps) for a in Q]
    return np.argmax(noisy_array)


def noisy_max_v2b(Q, eps):
    noisy_array = [a + np.random.exponential(scale=2.0 / eps) for a in Q]
    return max(noisy_array)


def sparse_vector(Q, eps, N, T):
    out = []
    eta1 = np.random.laplace(scale=2.0 / eps)
    Tbar = T + eta1
    c1, c2, i = 0, 0, 0
    while i < len(Q):
        eta2 = np.random.laplace(scale=4 * N / eps)
        if Q[i] + eta2 >= Tbar:
            out.append(True)
            c1 += 1
        else:
            out.append(False)
            c2 += 1
        i += 1

    return out
