# StatDP 
[![Build Status](https://travis-ci.com/RyanWangGit/statdp.svg?token=6D8zTzZr7SPui6PzhT2a&branch=master)](https://travis-ci.com/RyanWangGit/StatDP)  [![codecov](https://codecov.io/gh/RyanWangGit/statdp/branch/master/graph/badge.svg?token=1esLM0E5BZ)](https://codecov.io/gh/RyanWangGit/statdp)

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
def detect_counterexample(algorithm, test_epsilon, default_kwargs={},
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
![iSVT4](https://raw.githubusercontent.com/RyanWangGit/StatDP/master/examples/iSVT4.svg?sanitize=true)

## Customizing the detection
Our tool is designed to be modular and components are fully decoupled. You can write your own `input generator`/`event selector` and apply them to `hypothesis test`.

In general the detection process is `generate_databases -> select_event -> hypothesis_test`, you can checkout the definition and docstrings of the functions respectively to define your own generator/selector.Basically the `detect_counterexample` function in `statdp.core` package is just shortcut function to take care of the above process for you.

`hypothesis_test` function can be used universally by all algorithms, but you may need to design your own generator or selector for your own algorithm since our input generator and event selector are designed to work with numerical queries on databases.

## Citing this work

You are encouraged to cite the following [paper](https://arxiv.org/pdf/1805.10277.pdf) if you use this tool for academic research:

```bibtex
@inproceedings{Ding:2018:DVD:3243734.3243818,
 author = {Ding, Zeyu and Wang, Yuxin and Wang, Guanhong and Zhang, Danfeng and Kifer, Daniel},
 title = {Detecting Violations of Differential Privacy},
 booktitle = {Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security},
 series = {CCS '18},
 year = {2018},
 isbn = {978-1-4503-5693-0},
 location = {Toronto, Canada},
 pages = {475--489},
 numpages = {15},
 url = {http://doi.acm.org/10.1145/3243734.3243818},
 doi = {10.1145/3243734.3243818},
 acmid = {3243818},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {counterexample detection, differential privacy, statistical testing},
} 
```

## License
[MIT](https://github.com/RyanWangGit/StatDP/blob/master/LICENSE).
