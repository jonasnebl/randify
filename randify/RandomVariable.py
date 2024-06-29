from functools import cached_property, wraps
import numpy as np
from .utils import pdf


class RandomVariable:
    """
    Class extending an arbitrary python object to a random variable.
    The random variable can be defined in two ways:
    1. generator_func that generates samples of the random variable
    2. samples of the random variable
    Most functionalities are based on the samples of the random variable.
    If a generator_func is used, samples are generated internally based on the generator_func.
    """

    def __init__(self, *args, **kwargs):
        self.N_samples_default = 1000

        # parse input and check whether samples or generator_func are used for initialization
        if args and isinstance(args[0], list):
            self.samples = args[0]
        elif "samples" in kwargs:
            self.samples = kwargs["samples"]
        elif args and callable(args[0]):
            self.generator_func = args[0]
            self.generator_args = args[1:]
            self.generator_kwargs = kwargs
        elif "generator_func" in kwargs:
            self.generator_func = kwargs.pop("generator_func")
            self.generator_args = []
            self.generator_kwargs = kwargs
        else:
            raise ValueError(
                "Invalid initialization. Either 'generator_func' or 'samples' must be provided."
            )

        # save example sample for determining type and properties of the randomized variable
        if "samples" in self.__dict__:  # avoid hasattr to not trigger cached_property
            self.example_sample = self.samples[0]
        elif hasattr(self, "generator_func"):
            self.example_sample = self.generator_func(*self.generator_args, **self.generator_kwargs)

    def __call__(self, property_: str = None):
        """
        Returns whole object or single property as random variables.
        :param property: optional, Property to extract from the random variable.
        :return: RandomVariable object representing the whole object or a single property.
        """
        if property_ is None:
            return self
        elif not hasattr(self.example_sample, property_):
            raise ValueError(f"Property {property_} not available.")
        elif hasattr(self, "generator_func"):
            if callable(getattr(self.example_sample, property_)):
                return RandomVariable(
                    generator_func=lambda: getattr(self.generator_func(), property_)()
                )
            else:
                return RandomVariable(
                    generator_func=lambda: getattr(self.generator_func(), property_)
                )
        else:
            if callable(getattr(self.example_sample, property_)):
                return RandomVariable(
                    samples=[getattr(sample, property_)() for sample in self.samples]
                )
            else:
                return RandomVariable(
                    samples=[getattr(sample, property_) for sample in self.samples]
                )

    def __getitem__(self, key):
        """
        If the randomVariable is a list, ndarray or dict, return a RandomVariable of the key element.
        :param key: Key of the element to return as RandomVariable.
        :return: RandomVariable of the key element.
        """
        return RandomVariable(samples=[sample[key] for sample in self.samples])

    def sample(self, N: int = 1):
        """
        Return N random samples of the random variable.
        :return: N Samples of the random variable.
        """
        if hasattr(self, "generator_func"):
            if N == 1:
                return self._return_N_new_samples_from_generator_func(N)[0]
            else:
                return self._return_N_new_samples_from_generator_func(N)
        else:
            return np.random.choice(self.samples, size=N, replace=True)

    def _return_N_samples(self, N):
        """
        Returns N samples of the random variable. If more than N samples are available,
        N samples are randomly selected. If less than N samples are available,
        the samples are extended to the number N. Instead of sample(),
        this function may change the number of self.samples and is for interal use only.
        :param N: Number of samples to generate
        """
        if len(self.samples) == N:
            return self.samples
        elif len(self.samples) > N:
            return self.samples[:N]
        elif len(self.samples) < N:
            if hasattr(self, "generator_func"):
                self.samples += self._return_N_new_samples_from_generator_func(
                    N - len(self.samples)
                )
            else:
                # augment samples by reselecting existing samples if no generator_func is available
                self.samples += np.choice(self.samples, size=N - len(self.samples), replace=True)

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

            return self.samples

    def _return_N_new_samples_from_generator_func(self, N: int):
        """
        Generate N new samples of the random variable based on the generator_func.
        :param N: Number of samples to generate. Overwrites existing samples.
        :return: N new samples of the random variable based on the generator_func.
        """
        try:  # E.g. numpy distributions allow a size argument, faster than list comprehension
            samples = list(
                self.generator_func(*self.generator_args, **self.generator_kwargs, size=N)
            )
            assert len(self.samples) == N
            assert isinstance(self.samples[0], type(self.example_sample))
        except:
            samples = [
                self.generator_func(*self.generator_args, **self.generator_kwargs) for _ in range(N)
            ]
        return samples

    @cached_property
    def samples(self):
        """
        Samples attribute as cached property.
        If no samples are provided and samples are needed (e.g. for statistical measure calculation),
        this function will be kalled and generate samples based on the generator_func.
        :return: Generated samples of the random variable.
        """
        return self._return_N_new_samples_from_generator_func(N=self.N_samples_default)

    def pdf(self, x):
        """
        Calculate the probability density function of the random variable at x.
        Based on a kernel density estimate of the random variable.
        :param x: Value to evaluate the probability density function at.
        :return: Probability density function at x.
        """
        return pdf(self)(x)

    def _try_statistical_measure(foo):
        """
        Try to calculate a statistical measure of the random variable.
        If the random variable is not numeric, a TypeError is raised.
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
                    "expected value, variance, skewness and kurtosis. "
                    "Numeric RandomVariable can e.g. be int, float or a numpy ndarray. "
                    "Specifically the class of the randomized object must "
                    "Custom classes work if they implement __add__, radd__, "
                    "__mul__, rmul__, __truediv__ and rtruediv__ methods. "
                )

        return inner

    @cached_property
    @_try_statistical_measure
    def expected_value(self):
        r"""
        Calculates expected value $\mu = E[X]$ of the random variable.
        :return: Expected value $\mu$
        """
        return sum(self.samples) / len(self.samples)

    @cached_property
    @_try_statistical_measure
    def variance(self):
        r"""
        Calculates variance $\sigma^2 = E[(X - \mu)^2]$ of the random variable.
        :return: Variance $\sigma^2$
        """
        return sum([(sample - self.expected_value) ** 2 for sample in self.samples]) / (
            len(self.samples) - 1
        )

    @cached_property
    @_try_statistical_measure
    def skewness(self):
        r"""
        Calculates skewness $\gamma = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^3 \\right]$
        of the random variable.
        :return: Skewness $\gamma$
        """
        return (
            sum([(sample - self.expected_value) ** 3 for sample in self.samples])
            * len(self.samples)
            / ((len(self.samples) - 1) * (len(self.samples) - 2) * self.variance**1.5)
        )

    @cached_property
    @_try_statistical_measure
    def kurtosis(self):
        r"""
        Calculates kurtosis $\\beta = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^4 \\right]$
        of the random variable.
        :return: Kurtosis $\\beta$
        """
        return (
            sum([(sample - self.expected_value) ** 4 for sample in self.samples])
            * len(self.samples)
            * (len(self.samples) + 1)
            / (
                (len(self.samples) - 1)
                * (len(self.samples) - 2)
                * (len(self.samples) - 3)
                * self.variance**2
            )
        )

    def __str__(self):
        """
        Return a string representation of the random variable for printing.
        :return: String representation of the random variable.
        """
        string = f"<RandomVariable of type {type(self.example_sample).__name__}"
        if hasattr(self, "generator_func"):
            string += f" with {self.generator_func.__name__} distribution"
        else:
            string += f" with custom distribution"
        return string

    def __repr__(self):
        return self.__str__()
