from statdp import *
from statdp.algorithms import *
from intervals import Interval
from math import inf
import time


def draw_graph(xlabel, ylabel, data, title, name):
    import matplotlib.pyplot as plt
    plt.ion()
    plt.show()
    assert isinstance(data, dict)
    x = []
    p2 = []
    for epsilon, points in data.items():
        x = [i[0] for i in points]
        p1 = [i[1] for i in points]
        p2 = [i[2] for i in points]
        plt.plot(x, p1, 'o-', label='p1 $\epsilon = %.1f$' % epsilon, markersize=4)
        plt.axvline(x=epsilon, color='black', linestyle='dashed', alpha=0.8)

    plt.plot(x, p2, 'o-', label='p2', markersize=3)
    plt.axhline(y=0.05, color='green', linestyle='dashed', alpha=0.8)
    plt.axhline(y=0.01, color='red', linestyle='dashed', alpha=0.8)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    lgd = plt.legend(bbox_to_anchor=(1.13, 0.5), loc=8)
    plt.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.draw()
    plt.pause(0.001)
    plt.gcf().clear()
    return


def main():
    # auto-generated data for sparsevector
    # D1 = [10.747997673023637, 9.176688735248929, 9.685779806342222, 9.276506422051003, 10.14197198493314, 10.777285751217708, 10.033518297158158, 10.739107887579431, 9.466024642462145, 10.046506781898248]
    # D2 = [9.74799, 9.999989999999999, 10.19486, 9.99999, 9.57099, 9.888639999999999, 9.08919, 9.76694, 9.99999, 9.73776]

    # auto-generated data for noisymax
    # D1 = [10.047838788362302, 9.156301318712105, 9.934576874381165, 10.818923545533439, 10.964414221707846, 9.814549818258723, 10.343685529387757, 10.270895438249106, 10.422325599687007, 9.498322242077997]
    # D2 = [11.04783, 10.10206, 10.85322, 10.04784, 10.04784, 10.40406, 10.04784, 9.664679999999999, 9.660929999999999, 10.06784]

    # manual data for niosymax version 1a and 2a
    #D1 = [0] + [2 for _ in range(4)]
    #D2 = [1 for _ in range(5)]

    # manual data for noisymax version 1b and 2b
    D1 = [2 for _ in range(5)]
    D2 = [1 for _ in range(5)]
    jobs = [
        {
            "algorithm": noisy_max_v1a,
            "D1": lambda eps: [0] + [2 for _ in range(4)],
            "D2": lambda eps: [1 for _ in range(5)],
            "S": lambda: event_selector(algorithm, (), kwargs, D1, D2, epsilon,
                                        search_space=[[i] for i in range(5)])
        },
        {
            "algorithm": noisy_max_v1b,
            "D1": lambda eps: [2 for _ in range(5)],
            "D2": lambda eps: [1 for _ in range(5)],
            "S": lambda: event_selector(algorithm, (), kwargs, D1, D2, epsilon,
                                        search_space=[Interval([-inf, alpha]) for alpha in range(-5, 6)])
        },
        {
            "algorithm": noisy_max_v2a,
            "D1": lambda eps: [0] + [2 for _ in range(4)],
            "D2": lambda eps: [1 for _ in range(5)],
            "S": lambda: event_selector(algorithm, (), kwargs, D1, D2, epsilon,
                                        search_space=[[i] for i in range(5)])
        },
        {
            "algorithm": noisy_max_v2b,
            "D1": lambda eps: [2] + [0 for _ in range(4)],
            "D2": lambda eps: [1 for _ in range(5)],
            "S": lambda: event_selector(algorithm, (), kwargs, D1, D2, epsilon,
                                        search_space=[Interval([-inf, 1 + alpha / 10.0]) for alpha in range(0, 80, 2)])
        },
        {
            "algorithm": histogram,
            "D1": lambda eps: [2 for _ in range(5)],
            "D2": lambda eps: [1] + [2 for _ in range(4)],
            "S": lambda: event_selector(algorithm, (), kwargs, D1, D2, epsilon,
                                        search_space=[Interval([2 + alpha / 10.0, inf]) for alpha in range(0, 20, 2)])
        }
    ]

    # manual data for sparsevector
    results = {}

    start_time = time.time()

    for job in jobs:
        results.clear()

        algorithm = job['algorithm']
        for algorithm_epsilon in [0.2, 0.5, 0.7] + list(range(1, 4)):
            results[algorithm_epsilon] = []
            for epsilon in [x / 10.0 for x in range(1, 34, 1)]:
                # algorithm's fixed privacy budget
                kwargs = {'eps': algorithm_epsilon}
                kwargs.update(job.get('kwargs', {}))

                D1 = job['D1'](epsilon)
                D2 = job['D2'](epsilon)

                # call s selector to find s
                S = job['S']()

                p1, p2 = hypothesis_test(algorithm, (), kwargs, D1, D2, S, epsilon, 100000, cores=0)

                results[algorithm_epsilon].append((epsilon, p1, p2))
                print("epsilon: %f, p1 = %f, p2 = %f | S = %s" % (epsilon, p1, p2, S))
                draw_graph('Test $\epsilon$', 'P Value', results,
                           algorithm.__name__.replace('_', ' ').title(),
                           algorithm.__name__ + '.pdf')

        # print output
        print("%s" % algorithm.__name__)
        print("D1 : %s" % D1)
        print("D2 : %s" % D2)
        print("Time elapsed: %f seconds" % (time.time() - start_time))


if __name__ == '__main__':
    main()
