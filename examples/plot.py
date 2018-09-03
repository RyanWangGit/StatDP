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
    if not title == '' and title is not None:
        plt.title(title)
    plt.legend()
    plt.savefig(output_filename, bbox_inches='tight')
    plt.draw()
    return
