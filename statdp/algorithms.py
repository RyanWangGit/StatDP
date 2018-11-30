import numpy as np


def _argmax(iterable):
    # implement numpy.argmax in pure python, faster if iterable is plain python list
    return max(enumerate(iterable), key=lambda t: t[1])[0]


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
    return np.count_nonzero(out != ([True for _ in range(true_count)] + [False for _ in range(len(queries) - true_count)]))


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
    return np.count_nonzero(out != ([True for _ in range(true_count)] + [False for _ in range(len(queries) - true_count)]))


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
    return np.count_nonzero(out != ([True for _ in range(true_count)] + [False for _ in range(len(queries) - true_count)]))


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
