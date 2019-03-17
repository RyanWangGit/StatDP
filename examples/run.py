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
import time
import json
import coloredlogs
import logging
import matplotlib
from statdp import detect_counterexample
from statdp.algorithms import *

# switch matplotlib backend for running in background
matplotlib.use('agg')

coloredlogs.install('DEBUG', fmt='%(asctime)s [0x%(process)x] %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def plot_result(xlabel, ylabel, data, title, output_filename):
    """
    :param xlabel: The label for x axis.
    :param ylabel: The label for y axis.
    :param data: The input data sets to plots. e.g., {algorithm_epsilon: test_result}
    :param title: The title of the figure.
    :param output_filename: The output file name.
    :return:
    """
    import matplotlib.pyplot as plt

    matplotlib.rcParams['xtick.labelsize'] = '12'
    matplotlib.rcParams['ytick.labelsize'] = '12'

    markers = ['s', 'o', '^', 'x', '*', '+', 'p']
    colorcycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
                  '#17becf']
    plt.ylim(0.0, 1.0)

    for i, (epsilon, points) in enumerate(data.items()):
        x = [item[0] for item in points]
        p = [item[1] for item in points]
        plt.plot(x, p, 'o-', label=r'$\epsilon_0$ = {0}'.format(epsilon), markersize=6, marker=markers[i])
        plt.axvline(x=float(epsilon), color=colorcycle[i], linestyle='dashed', linewidth=1.2)

    plt.axhline(y=0.05, color='black', linestyle='dashed', linewidth=1.2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title is not None and not title == '':
        plt.title(title)
    plt.legend()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.gcf().clear()


def main():
    """The selected databases/kwargs/event found for different algorithms
        |   Algorithm   | Databases | kwargs | Event |
        |:-------------:|:---------:|:------:|:-----:|
        | noisy_max_v1a |           |        |       |
        | noisy_max_v1b |           |        |       |
        | noisy_max_v2a |           |        |       |
        | noisy_max_v2b |           |        |       |
        |   histogram   |           |        |       |
        | histogram_eps |           |        |       |
        |      SVT      |           |        |       |
        |     iSVT1     |           |        |       |
        |     iSVT2     |           |        |       |
        |     iSVT3     |           |        |       |
        |     iSVT4     |           |        |       |
    """
    tasks = [(noisy_max_v1a, {}), (noisy_max_v1b, {}), (noisy_max_v2a, {}), (noisy_max_v2b, {}),
             (histogram, {}), (histogram_eps, {}),
             (SVT, {'N': 1, 'T': 0.5}),
             (iSVT1, {'T': 1, 'N': 1}), (iSVT2, {'T': 1, 'N': 1}), (iSVT3, {'T': 1, 'N': 1}), (iSVT4, {'T': 1, 'N': 1})]

    for i, (algorithm, kwargs) in enumerate(tasks):
        start_time = time.time()
        results = {}
        for privacy_budget in (0.2, 0.7, 1.5):
            kwargs['epsilon'] = privacy_budget
            results[privacy_budget] = detect_counterexample(algorithm, tuple(x / 10.0 for x in range(1, 34, 1)), kwargs)

        plot_result(r'Test $\epsilon$', 'P Value',
                    results, algorithm.__name__.replace('_', ' ').title(), algorithm.__name__ + '.pdf')

        # dump the results to file
        with open('./{}.json'.format(algorithm.__name__), 'w') as f:
            json.dump(results, f)

        logger.info('[{} / {}]: {} | Time elapsed: {}'
                    .format(i + 1, len(tasks), algorithm.__name__, time.time() - start_time))


if __name__ == '__main__':
    main()
