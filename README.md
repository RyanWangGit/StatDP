# StatDP 
[![Build Status](https://travis-ci.com/RyanWangGit/StatDP.svg?token=6D8zTzZr7SPui6PzhT2a&branch=master)](https://travis-ci.com/RyanWangGit/StatDP)  [![codecov](https://codecov.io/gh/RyanWangGit/StatDP/branch/master/graph/badge.svg?token=1esLM0E5BZ)](https://codecov.io/gh/RyanWangGit/StatDP) [![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2FRyanWangGit%2FStatDP.svg?type=shield)](https://app.fossa.io/projects/git%2Bgithub.com%2FRyanWangGit%2FStatDP?ref=badge_shield)

Statistical Counterexample Detector for Differential Privacy.

## Usage
You have to define your algorithm with the fisrt two arguments being (Queries, Privacy Budget).

Then you can simply call the detection tool with automatic database generation and event selection:
```python
from statdp import detect_counterexample

def your_algorithm(Q, epsilon, ...):
     # your algorithm implementation here
 
if __name__ == '__main__':
    # algorithm privacy budget argument(`epsilon`) is required
    result = detect_counterexample(your_algorithm, {'epsilon': algorithm_epsilon}, test_epsilon)
```

The result is returned in variable `result`, which is stored as `[(epsilon, p, d1, d2, kwargs, event), (...)]`. 

The `detect_counterexample` accepts multiple extra arguments to customize the process, check the signature and notes of `detect_counterexample` method to see how to use.

```python
def detect_counterexample(algorithm, test_epsilon, default_kwargs,
                           event_search_space=None, databases=None,
                           event_iterations=100000, detect_iterations=500000, cores=0,
                           loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument, 'epsilon' must be provided.
    :param event_search_space: The search space for event selector to reduce search time, optional.
    :param databases: The databases to run for detection, optional.
    :param event_iterations: The iterations for event selector to run, default is 100000.
    :param detect_iterations: The iterations for detector to run, default is 500000.
    :param cores: The cores to utilize, 0 means auto-detection.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
```

## Visualizing the results
A nice python library `matplotlib` is recommended for visualizing your result. 

There's a python code snippet at `/examples/run.py`(`draw_graph` method) to show an example of plotting the results.

Then you can generate a figure like the iSVT 4 in our paper.
![iSVT4](https://raw.githubusercontent.com/RyanWangGit/StatDP/master/examples/isvt4.svg?sanitize=true)

## Customizing the detection

## Citing this work

You are encouraged to cite the following [paper](https://arxiv.org/pdf/1805.10277.pdf) if you use this tool for academic research:

```bibtex
@article{ding2018detecting,
  title={Detecting Violations of Differential Privacy},
  author={Ding, Zeyu and Wang, Yuxin and Wang, Guanhong and Zhang, Danfeng and Kifer, Daniel},
  journal={arXiv preprint arXiv:1805.10277},
  year={2018}
}
```

## License
[MIT](https://github.com/RyanWangGit/StatDP/blob/master/LICENSE).
