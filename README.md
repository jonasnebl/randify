# Randify

Randomize your existing code with randify.
Evaluate how your code reacts to randomly distributed inputs and evaluate the probability distributions of your code outputs.
Randify performs Monte Carlo simulations to estimate probability density distributions of your outputs. Designed for simple syntax.

Full API documentation at https://jonasnebl.github.io/randify/randify.html

## Installation
```
$ git clone https://github.com/jonasnebl/randify.git
$ cd randify
$ pip install -e .
``` 

## Quick guide

Randify works on functions with any number and type of input and output arguments.
In this his quick example we have a function `y1, x2 = f(x1, x2)` returning the sum `x1 + x2` and the product `x1 * x2` of two arguments `x1` and `x2`. For more in-depth examples check-out the Jupyter-Notebooks in `examples`.
```
def f(x1, x2):
    return x1 + x2, x1 * x2

x1 = 1
x2 = 2
y1, x2 = f(x1, x2)
```
Now we want to evaluate how random inputs `x1` and `x2` influence the results `y1`and `y2` using randify. We can do this using two steps.
1. Define `x1`and/or `x2`as a RandomVariable. For defining the RandomVariable you need to pass a function that generates random samples of this RandomVariables.
In this example, we use functions from `numpy`'s random module, but you can also define custom functions (see the examples for that).
2. Call the `randify` function wrapper with the RandomVariables as arguments.
```
from randify import randify, RandomVariable
import numpy as np

def f(x1, x2):
    return x1 + x2, x1 * x2

x1 = RandomVariable(np.random.normal, loc=0, scale=1)
x2 = RandomVariable(np.random.uniform, low=-1, high=1)
y1, y2 = randify(f)(x1, x2)
```
`y1`and `y2` are now also RandomVariables. You can calculate the resulting statistical measure like expected value or variance. To display the estimated probability distributions, you can use randify's `plot_pdf` function:
```
print(f"E[y1] = {y1.expected_value}")
print(f"Var[y1] = {y1.variance}")

from randify import plot_pdf
plot_pdf(x1=x1, x2=x2, y1=y1, y2=y2)
```

## Documentation

Full API documentation can be found at https://jonasnebl.github.io/randify/randify.html. The documentation is generated automatically using `pdoc`.
```
$ pip install pdoc
$ pdoc randify --math - o ./doc
```

## Formatting 
`ruff` is used to format randify. 
```
$ pip install ruff
$ ruff format
```
