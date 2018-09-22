from statdp import *
from statdp.algorithms import *
from intervals import Interval
from math import inf
import time
import jsonpickle


def draw_graph(xlabel, ylabel, data, title, output_filename):
    """
    :param xlabel: The label for x axis.
    :param ylabel: The label for y axis.
    :param data: The input data sets to plots. e.g., {algorithm_epsilon: test_result}
    :param title: The title of the figure.
    :param output_filename: The output file name.
    :return:
    """
    import matplotlib
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = '\\usepackage[bold]{libertine},\\usepackage[libertine]{newtxmath},\\usepackage{sfmath},\\usepackage[T1]{fontenc}'
    matplotlib.rcParams['xtick.labelsize'] = '12'
    matplotlib.rcParams['ytick.labelsize'] = '12'

    import matplotlib.pyplot as plt

    markers = ['s', 'o', '^', 'x', '*', '+', 'p']
    colorcycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
    plt.show()
    plt.ylim(0.0, 1.0)

    for i, (epsilon, points) in enumerate(data.items()):
        x = [i[0] for i in points]
        p = [i[1] for i in points]
        plt.plot(x, p, 'o-', label='\\large{$\epsilon_0$ = ' + '{0}'.format(epsilon) + '}',
                 markersize=6, marker=markers[i])
        plt.axvline(x=float(epsilon), color=colorcycle[i], linestyle='dashed', linewidth=1.2)

    plt.axhline(y=0.05, color='black', linestyle='dashed', linewidth=1.2)
    plt.xlabel('\\Large{ ' + xlabel + '}')
    plt.ylabel('\\Large{' + ylabel + '}')
    if title is not None and not title == '':
        plt.title(title)
    plt.legend()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.draw()
    return


def main():
    jobs = [
        {
            "algorithm": noisy_max_v1a,
            "kwargs": {},
            "databases": ([0] + [2 for _ in range(4)], [1 for _ in range(5)]),
            "search_space": tuple([i] for i in range(5))
        },
        {
            "algorithm": noisy_max_v1b,
            "kwargs": {},
            "databases": ([2 for _ in range(5)], [1 for _ in range(5)]),
            "search_space": tuple(Interval([-inf, alpha]) for alpha in range(-5, 6))
        },
        {
            "algorithm": noisy_max_v2a,
            "kwargs": {},
            "databases": ([0] + [2 for _ in range(4)], [1 for _ in range(5)]),
            "search_space": tuple([i] for i in range(5))
        },
        {
            "algorithm": noisy_max_v2b,
            "kwargs": {},
            "databases": ([2] + [0 for _ in range(4)], [1 for _ in range(5)]),
            "search_space": tuple(Interval([-inf, 1 + alpha / 10.0]) for alpha in range(0, 80, 2))
        },
        {
            "algorithm": histogram,
            "kwargs": {},
            "databases": ([2 for _ in range(5)], [1] + [2 for _ in range(4)]),
            "search_space": [Interval([-inf, alpha]) for alpha in range(-17, 17)]
        },
        {
            "algorithm": histogram_eps,
            "kwargs": {},
            "databases": ([0] + [1 for _ in range(4)], [1 for _ in range(5)]),
            "search_space": [Interval([-inf, alpha / 10]) for alpha in range(-30, 30, 2)]
        },
        {
            "algorithm": iSVT1,
            "kwargs": {'T': 1, 'N': 1},
            "databases": ([1 for _ in range(10)], [0 for _ in range(5)] + [2 for _ in range(5)]),
            "search_space": [[i] for i in range(10)]
        },
        {
            "algorithm": iSVT2,
            "kwargs": {'T': 1, 'N': 1},
            "databases": ([1 for _ in range(5)] + [0 for _ in range(5)],
                          [0 for _ in range(5)] + [1 for _ in range(5)]),
            "search_space": [[i] for i in range(10)]
        },
        {
            "algorithm": iSVT3,
            "kwargs": {'T': 1, 'N': 1},
            "databases": ([1 for _ in range(5)] + [0 for _ in range(5)],
                          [0 for _ in range(5)] + [1 for _ in range(5)]),
            "search_space": [[i] for i in range(10)]
        },
    ]

    results = {}

    start_time = time.time()

    for job in jobs:
        results.clear()

        algorithm, search_space,  = job['algorithm'], job['search_space']
        databases, kwargs = job['databases'], job['kwargs']
        for algorithm_epsilon in [0.2, 0.5, 0.7] + list(range(1, 4)):
            kwargs['epsilon'] = algorithm_epsilon
            results[algorithm_epsilon] = detect_counterexample(algorithm,
                                                               [x / 10.0 for x in range(1, 34, 1)],
                                                               kwargs,
                                                               search_space, databases)

        draw_graph('Test $\epsilon$', 'P Value', results,
                   algorithm.__name__.replace('_', ' ').title(),
                   algorithm.__name__ + '.pdf')

        # dump the results to file
        with open('./{}.json'.format(algorithm.__name__), 'w') as f:
            # json.dump(results, f)
            # cannot use json.dump since Interval class is not JSON-serializable
            f.write(jsonpickle.encode(results))

        print('{} | D1: {} | D2: {} | Time: {}'.format(algorithm.__name__, databases[0], databases[1],
                                                       time.time() - start_time))

        start_time = time.time()


if __name__ == '__main__':
    main()
