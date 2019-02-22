# StatDP 
[![Build Status](https://travis-ci.com/RyanWangGit/statdp.svg?branch=master)](https://travis-ci.com/RyanWangGit/statdp) [![Build status](https://ci.appveyor.com/api/projects/status/b6ul9tami06i2yge/branch/master?svg=true)](https://ci.appveyor.com/project/RyanWangGit/statdp/branch/master)  [![codecov](https://codecov.io/gh/RyanWangGit/statdp/branch/master/graph/badge.svg)](https://codecov.io/gh/RyanWangGit/statdp)

Statistical Counterexample Detector for Differential Privacy.

## Usage
You have to define your algorithm with the first argument being `Queries`.

Then you can simply call the detection tool with automatic database generation and event selection:
```python
from statdp import detect_counterexample

def your_algorithm(Q, epsilon, ...):
     # your algorithm implementation here
 
if __name__ == '__main__':
    # algorithm privacy budget argument(`epsilon`) is needed
    # otherwise detector won't work properly since it will try to generate a privacy budget
    result = detect_counterexample(your_algorithm, {'epsilon': privacy_budget}, test_epsilon)
```

The result is returned in variable `result`, which is stored as `[(epsilon, p, d1, d2, kwargs, event), (...)]`. 

The `detect_counterexample` accepts multiple extra arguments to customize the process, check the signature and notes of `detect_counterexample` method to see how to use.

```python
def detect_counterexample(algorithm, test_epsilon, default_kwargs=None, databases=None, num_input=(5, 10),
                          event_iterations=100000, detect_iterations=500000, cores=0,
                          quiet=False, loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument.
    :param databases: The databases to run for detection, optional.
    :param num_input: The length of input to generate, not used if database param is specified.
    :param event_iterations: The iterations for event selector to run, default is 100000.
    :param detect_iterations: The iterations for detector to run, default is 500000.
    :param cores: The cores to utilize, 0 means auto-detection.
    :param quiet: Do not print progress bar or messages, logs are not affected, default is False.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
```

## Visualizing the results
A nice python library `matplotlib` is recommended for visualizing your result. 

There's a python code snippet at `/examples/run.py`(`draw_graph` method) to show an example of plotting the results.

Then you can generate a figure like the iSVT 4 in our paper.
![iSVT4](https://raw.githubusercontent.com/RyanWangGit/StatDP/master/examples/iSVT4.svg?sanitize=true)

## Customizing the detection
Our tool is designed to be modular and components are fully decoupled. You can write your own `input generator`/`event selector` and apply them to `hypothesis test`.

In general the detection process is `test_epsilon --> generate_databases --((d1, d2, kwargs), ...), epsilon--> select_event --(d1, d2, kwargs, event), epsilon--> hypothesis_test --> (d1, d2, kwargs, event, p-value), epsilon`, you can checkout the definition and docstrings of the functions respectively to define your own generator/selector. Basically the `detect_counterexample` function in `statdp.core` module is just shortcut function to take care of the above process for you.

`test_statistics` function in `hypotest` module can be used universally by all algorithms (this function is to calculate p-value based on the observed statistics). However, you may need to design your own generator or selector for your own algorithm, since our input generator and event selector are designed to work with numerical queries on databases.

## Citing this work

You are encouraged to cite the following [paper](https://arxiv.org/pdf/1805.10277.pdf) if you use this tool for academic research:

```bibtex
@inproceedings{ding2018detecting,
  title={Detecting Violations of Differential Privacy},
  author={Ding, Zeyu and Wang, Yuxin and Wang, Guanhong and Zhang, Danfeng and Kifer, Daniel},
  booktitle={Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security},
  pages={475--489},
  year={2018},
  organization={ACM}
}
```

## License
[MIT](https://github.com/RyanWangGit/StatDP/blob/master/LICENSE).
