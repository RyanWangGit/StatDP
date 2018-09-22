import numpy as np


def noisy_max_v1a(Q, epsilon):
    # add laplace noise
    noisy_array = tuple(a + np.random.laplace(scale=2.0 / epsilon) for a in Q)

    # find the largest noisy element and return its index
    return np.argmax(noisy_array)


def noisy_max_v1b(Q, epsilon):
    noisy_array = tuple(a + np.random.laplace(scale=2.0 / epsilon) for a in Q)
    return max(noisy_array)


def noisy_max_v2a(Q, epsilon):
    noisy_array = tuple(a + np.random.exponential(scale=2.0 / epsilon) for a in Q)
    return np.argmax(noisy_array)


def noisy_max_v2b(Q, epsilon):
    noisy_array = tuple(a + np.random.exponential(scale=2.0 / epsilon) for a in Q)
    return max(noisy_array)


def histogram_eps(Q, epsilon):
    noisy_array = tuple(a + np.random.laplace(scale=epsilon) for a in Q)
    return noisy_array[0]


def histogram(Q, epsilon):
    noisy_array = tuple(a + np.random.laplace(scale=1.0 / epsilon) for a in Q)
    return noisy_array[0]


def laplace_mechanism(Q, epsilon):
    noisy_array = tuple(a + np.random.laplace(scale=len(Q)/epsilon) for a in Q)
    lower = 1 - 0.27
    upper = 1 + 0.75
    return sum(1 for element in noisy_array if lower <= element <= upper)


def SVT(Q, epsilon, N, T):
    out = []
    eta1 = np.random.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for q in Q:
        eta2 = np.random.laplace(scale=4.0 * N / epsilon)
        if q + eta2 >= noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out.count(False)


def iSVT1(Q, epsilon, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=2.0 * delta / epsilon)
    noisy_T = T + eta1
    for q in Q:
        eta2 = 0
        if (q + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)

    true_count = int(len(Q) / 2)
    return np.count_nonzero(out != ([True for _ in range(true_count)] + [False for _ in range(len(Q) - true_count)]))


def iSVT2(Q, epsilon, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=2.0 * delta / epsilon)
    noisy_T = T + eta1
    for q in Q:
        eta2 = np.random.laplace(scale=2.0 * delta / epsilon)
        if (q + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)

    true_count = int(len(Q) / 2)
    return np.count_nonzero(out != ([True for _ in range(true_count)] + [False for _ in range(len(Q) - true_count)]))


def iSVT3(Q, epsilon, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=4.0 * delta / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for q in Q:
        eta2 = np.random.laplace(scale=(4.0 * delta) / (3.0 * epsilon))
        if q + eta2 > noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)

    true_count = int(len(Q) / 2)
    return np.count_nonzero(out != ([True for _ in range(true_count)] + [False for _ in range(len(Q) - true_count)]))


def iSVT4(Q, epsilon, N, T):
    out = []
    eta1 = np.random.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for q in Q:
        eta2 = np.random.laplace(scale=2.0 * N / epsilon)
        if q + eta2 > noisy_T:
            out.append(q + eta2)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out.count(False)
