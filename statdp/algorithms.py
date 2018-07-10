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


def histogrameps(Q, eps):
    noisy_array = [a + np.random.laplace(scale=eps) for a in Q]
    return noisy_array[0]


def histogram(Q, eps):
    noisy_array = [a + np.random.laplace(scale=1/eps) for a in Q]
    return noisy_array[0]


def laplace_mechanismq_eps(Q, eps):
    noisy_array = [a + np.random.laplace(scale=len(Q)/eps) for a in Q]
    # floor=np.mean(noisy_array)-np.std(noisy_array)/len(Q)
    # ceiling=np.mean(noisy_array)+np.std(noisy_array)/len(Q)
    floor=1-0.27
    ceiling=1+0.75
    count=0
    for i in noisy_array:
        if i>=floor and i <=ceiling:
            count+=1
    return count


def laplace_mechanismeps_q(Q, eps):
    noisy_array = [a + np.random.laplace(scale=eps/len(Q)) for a in Q]
    # floor=np.mean(noisy_array)-np.std(noisy_array)/len(Q)
    # ceiling=np.mean(noisy_array)+np.std(noisy_array)/len(Q)
    floor = 1-0.27
    ceiling = 1+0.75
    count = 0
    for i in noisy_array:
        if i>=floor and i <=ceiling:
            count+=1
    return count


def sparse_vector_no_threshold_noise(Q, eps, N, T):
    out = []
    c1, c2, i = 0, 0, 0
    while i < len(Q) and c1 < N:
        eta = np.random.laplace(scale=4.0 * N / eps)
        if Q[i] + eta >= T:
            out.append(True)
            c1 += 1
        else:
            out.append(False)
            c2 += 1
        i += 1
    return c2


def sparse_vector_lyu(Q, eps, N, T):
    out = []
    eta1 = np.random.laplace(scale=2.0 / eps)
    Tbar = T + eta1
    c1, c2, i = 0, 0, 0
    while i < len(Q) and c1 < N:
        eta2 = np.random.laplace(scale=4 * N / eps)
        if Q[i] + eta2 >= Tbar:
            out.append(True)
            c1 += 1
        else:
            out.append(False)
            c2 += 1
        i += 1
    return c2


def sparse_vector_1(Q, eps, N, T):
    out = []
    eta1 = np.random.laplace(scale=2.0 / eps)
    noisy_T = T + eta1
    count = 0
    for q in Q:
        eta2 = np.random.laplace(scale=2.0 * N / eps)
        if q + eta2 > noisy_T:
            out.append(q+eta2)
            count += 1
            if count >= N:
                break
        else:
            out.append(False)
    return out.count(False)


def sparse_vector_2(Q, eps, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=4.0 * delta / eps)
    noisy_T = T + eta1
    count = 0
    for q in Q:
        eta2 = np.random.laplace(scale=(4.0 * delta) / (3.0 * eps))
        if (q + eta2) > noisy_T:
            out.append(True)
            count += 1
            if count >= N:
                break
        else:
            out.append(False)
    hdist = 0
    for index, value in enumerate(out):
        if index < len(Q) / 2 and value == True:
            hdist += 1
        if index >= len(Q) / 2 and value == False:
            hdist += 1
    return hdist


def sparse_vector_3(Q, eps, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=2.0 * delta / eps)
    noisy_T = T + eta1
    for q in Q:
        eta2 = 0
        if (q + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)
    hdist = 0
    for index, value in enumerate(out):
        if index < len(Q) / 2 and value == True:
            hdist += 1
        if index >= len(Q) / 2 and value == False:
            hdist += 1
    return hdist


def sparse_vector_4(Q, eps, N, T):
    out = []
    delta = 1
    eta1 = np.random.laplace(scale=2.0 * delta / eps)
    noisy_T = T + eta1
    for q in Q:
        eta2 = np.random.laplace(scale=2.0 * delta / eps)
        if (q + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)
    hdist = 0
    for index, value in enumerate(out):
        if index < len(Q) / 2 and value == True:
            hdist += 1
        if index >= len(Q) / 2 and value == False:
            hdist += 1
    return hdist
