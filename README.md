# PDFtimate

Estimate probability density function (PDF) transformations on your existing code by just adding decorators and random variables. Designed for syntactic simplicity.

## Installation

TBD
```
pip install PDFtimate
``` 

## Usage

You have a function `y = f(x)` peforming some kind of calculation. Variable `x` and also variable `y` are normal scalars.
```
def f(x):
    return 2 * x + 1

x = 0
y = f(x)
print(y)
```
Now `x` becomes a random variable. By using the `@randify` decorator, you can evaluate your function `f` with Monte-Carlo. The result y will also be a random variable with an altered distribution.

```
from PDFtimate import randify, RandomVariable

@randify
def f(x):
    return 2 * x + 1

x = RandomVariable(mean=0, variance=1)
y = f(x)
print(y.mean, y.variance)
```