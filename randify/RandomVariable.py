from functools import cached_property, wraps
import numpy as np
from .utils import pdf


class RandomVariable:
    """
    Class extending an arbitrary python object to a RandomVariable.
    """

    def __init__(self, generator_func_or_samples, *args, **kwargs):
        """
        Constructor of the RandomVariable.
        The RandomVariable can be defined in two ways:
        1. generator_func that generates samples of the RandomVariable.
        2. Samples of the RandomVariable. \n
        Most functionalities are based on the samples of the RandomVariable.
        If a generator_func is used, samples are generated internally based on the generator_func.
        As generator function e.g. distributions from numpy.random can be used:
        .. code-block:: python
            x1 = RandomVariable(np.random.normal, loc=0, scale=1)
        The arguments are piped through to np.random.normal.
        You can also use lambda expression for custom generator functions:
        .. code-block:: python
            x2 = RandomVariable(lambda: np.random.uniform(0, 1, size=(3,)))
        :param generator_func: Function that generates samples of the RandomVariable.
        :param samples: Samples of the RandomVariable.
        :param *args: Arguments for the generator_func.
        :param **kwargs: Keyword arguments for the generator_func.
        :return: RandomVariable object.
        """
        self._N_samples_default = int(1e3)

        # parse input and check whether samples or generator_func are used for initialization
        if callable(generator_func_or_samples):
            self.generator_func = generator_func_or_samples
            self.generator_args = args
            self.generator_kwargs = kwargs
        elif isinstance(generator_func_or_samples, list):
            self._samples = generator_func_or_samples
        else:
            raise ValueError(
                "Invalid initialization. Provide a callable generator_func or a list of samples."
            )

        # save example sample for determining type and properties of the randomized variable
        if "_samples" in self.__dict__:  # avoid hasattr to not trigger cached_property
            self.example_sample = self._samples[0]
        elif hasattr(self, "generator_func"):
            self.example_sample = self.generator_func(*self.generator_args, **self.generator_kwargs)

    def __call__(self, property_: str = None):
        """
        Returns whole object or single property as RandomVariables.
        :param property_: optional, Property to extract from the RandomVariable.
        :return: RandomVariable object representing the whole object or a single property.
        """
        if property_ is None:
            return self
        elif not hasattr(self.example_sample, property_):
            raise ValueError(f"Property {property_} not available.")
        else:
            if callable(getattr(self.example_sample, property_)):
                return RandomVariable([getattr(sample, property_)() for sample in self._samples])
            else:
                return RandomVariable([getattr(sample, property_) for sample in self._samples])

    def __getitem__(self, key):
        """
        If the randomVariable is a list, ndarray or dict, return a RandomVariable of the key element.
        :param key: Key of the element to return as RandomVariable.
        :return: RandomVariable of the key element.
        """
        return RandomVariable([sample[key] for sample in self._samples])

    def sample(self, N: int = 1):
        """
        Return N random samples of the RandomVariable.
        :param N: Number of samples
        :return: N samples of the RandomVariable.
        """
        if hasattr(self, "generator_func"):
            if N == 1:
                return self._return_N_new_samples_from_generator_func(N)[0]
            else:
                return self._return_N_new_samples_from_generator_func(N)
        else:
            selected_indices = np.random.choice(len(self._samples), size=N, replace=True)
            ret_samples = [self._samples[i] for i in selected_indices]
            if N == 1:
                return ret_samples[0]
            else:
                return ret_samples

    def _return_N_samples(self, N):
        """
        Returns a list of N samples of the RandomVariable. If more than N samples are available,
        N samples are randomly selected. If less than N samples are available,
        the samples are extended to the number N. Instead of sample(),
        this function may change the number of self._samples and is for interal use only.
        :param N: Number of samples
        :return: list of N samples of the RandomVariable.
        """
        if len(self._samples) == N:
            return self._samples

        elif len(self._samples) < N and hasattr(self, "generator_func"):
            # new samples from generator function
            self._samples += self._return_N_new_samples_from_generator_func(N - len(self._samples))

            # delete cached properties based on samples
            # because more samples are available for more accurate statistical measures
            if hasattr(self, "expected_value"):
                del self.expected_value
            if hasattr(self, "variance"):
                del self.variance
            if hasattr(self, "skewness"):
                del self.skewness
            if hasattr(self, "kurtosis"):
                del self.kurtosis

            return self._samples

        elif len(self._samples) > N:
            selected_indices = np.random.choice(len(self._samples), size=N, replace=True)
            return [self._samples[i] for i in selected_indices]  # don't reduce number of samples

        elif len(self._samples) < N:
            selected_indices = np.random.choice(len(self._samples), size=N, replace=True)
            self._samples = [self._samples[i] for i in selected_indices]
            return self._samples

    def _return_N_new_samples_from_generator_func(self, N: int):
        """
        Generate N new samples of the RandomVariable based on the generator_func.
        :param N: Number of samples to generate. Overwrites existing samples.
        :return: N new samples of the RandomVariable based on the generator_func.
        """
        try:  # E.g. numpy distributions allow a size argument, faster than list comprehension
            samples = list(
                self.generator_func(*self.generator_args, **self.generator_kwargs, size=N)
            )
            assert len(self._samples) == N
            assert isinstance(self._samples[0], type(self.example_sample))
        except:
            samples = [
                self.generator_func(*self.generator_args, **self.generator_kwargs) for _ in range(N)
            ]
        return samples

    @cached_property
    def _samples(self):
        """
        Samples attribute as cached property.
        If no samples are provided and samples are needed (e.g. for statistical measure calculation),
        this function will be called and generate samples based on the generator_func.
        :return: Generated samples of the RandomVariable.
        """
        return self._return_N_new_samples_from_generator_func(N=self._N_samples_default)

    def pdf(self, x):
        """
        Calculate the probability density function of the RandomVariable at x.
        Based on a kernel density estimation using KDEpy.
        :param x: Value to evaluate the probability density function at.
        :return: Probability density function at x.
        """
        return pdf(self)(x)

    def _try_statistical_measure(foo):
        """
        Try to calculate a statistical measure of the RandomVariable.
        If the RandomVariable is not numeric, a TypeError is raised.
        :param foo: Function to calculate the statistical measure.
        :return: Wrapper function with try-except block around foo().
        """

        @wraps(foo)
        def inner(self):
            try:
                return foo(self)
            except TypeError as e:
                raise TypeError(
                    "RandomVariable must be numeric to calculate "
                    "expected value, variance, skewness and kurtosis, "
                    "e.g. int, float or a numpy ndarray. "
                    "Custom classes work if they implement the needed arithmetic operations."
                )

        return inner

    @cached_property
    @_try_statistical_measure
    def expected_value(self):
        r"""
        Calculates expected value $\mu = E[X]$ of the RandomVariable.
        :return: Expected value $\mu$
        """
        return sum(self._samples) / len(self._samples)

    @cached_property
    @_try_statistical_measure
    def variance(self):
        r"""
        Calculates variance $\sigma^2 = E[(X - \mu)^2]$ of the RandomVariable.
        :return: Variance $\sigma^2$
        """
        return sum([(-self.expected_value + sample) ** 2 for sample in self._samples]) / (
            len(self._samples) - 1
        )

    @cached_property
    @_try_statistical_measure
    def skewness(self):
        r"""
        Calculates skewness $\gamma = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^3 \\right]$
        of the RandomVariable.
        :return: Skewness $\gamma$
        """
        return (
            sum([(-self.expected_value + sample) ** 3 for sample in self._samples])
            * len(self._samples)
            / ((len(self._samples) - 1) * (len(self._samples) - 2) * self.variance**1.5)
        )

    @cached_property
    @_try_statistical_measure
    def kurtosis(self):
        r"""
        Calculates kurtosis $\\beta = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^4 \\right]$
        of the RandomVariable.
        :return: Kurtosis $\\beta$
        """
        return (
            sum([(-self.expected_value + sample) ** 4 for sample in self._samples])
            * len(self._samples)
            * (len(self._samples) + 1)
            / (
                (len(self._samples) - 1)
                * (len(self._samples) - 2)
                * (len(self._samples) - 3)
                * self.variance**2
            )
        )

    def __str__(self):
        """
        Return a string representation of the RandomVariable for printing.
        :return: String representation of the RandomVariable.
        """
        return f"<RandomVariable of type <{type(self.example_sample).__name__}>>"

    def __repr__(self):
        return self.__str__()
