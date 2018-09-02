# StatDP [![Build Status](https://travis-ci.com/RyanWangGit/StatDP.svg?token=6D8zTzZr7SPui6PzhT2a&branch=master)](https://travis-ci.com/RyanWangGit/StatDP)  [![codecov](https://codecov.io/gh/RyanWangGit/StatDP/branch/master/graph/badge.svg?token=1esLM0E5BZ)](https://codecov.io/gh/RyanWangGit/StatDP)

Counterexample Detection Using Statistical Methods for Incorrect Differential-Privacy Algorithms.

# Usage
You have to define your algorithm wit the fisrt two arguments being (Querys, Privacy Budget), and the privacy budget should be named 'epsilon'.

Then you can simply call the detection tool with automatic database generation and event selection:
```python
from statdp import detect_counter_example

def your_algorithm(Q, epsilon, ...):
     # your algorithm implementation here
 
if __name__ == '__main__':
    result = detect_counter_example(your_algorithm, test_epsilon)
```

Then the result is returned in variable `result`. the `detect_counter_example` accepts multiple extra arguments to customize the process:

| Argument | Usage |
| -------- | ----- |
| `test_epsilon`| either be a concrete number or a list, which is the epsilon(s) to be tested  |


## Results

Results are returned using dicts.

## Visualizing the results
A nice python library `matplotlib` is recommended for visualizing your result.

Then you will generate a figure like the iSVT 4 in our paper.
![iSVT4](https://raw.githubusercontent.com/RyanWangGit/StatDP/master/examples/isvt4.svg?sanitize=true)

## Customizing the detection


# Citing this work

You are encouraged to cite the following [paper](https://arxiv.org/pdf/1805.10277.pdf) if you use this tool for academic research:

```
@article{ding2018detecting,
  title={Detecting Violations of Differential Privacy},
  author={Ding, Zeyu and Wang, Yuxin and Wang, Guanhong and Zhang, Danfeng and Kifer, Daniel},
  journal={arXiv preprint arXiv:1805.10277},
  year={2018}
}
```

# License
[MIT](https://github.com/RyanWangGit/StatDP/blob/master/LICENSE).