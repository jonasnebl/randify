from functools import cached_property
import numpy as np


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

        # determine type
        if hasattr(self, "samples"):
            self.example_sample = self.samples[0]
        elif hasattr(self, "generator_func"):
            self.example_sample = self.generator_func(*self.generator_args, **self.generator_kwargs)
        self._type = type(self.example_sample).__name__

    def __call__(self, property_: str = None):
        """
        Returns whole object or single property as random variables.
        :param property: optional, Property to extract from the random variable.
        :return: RandomVariable object representing the whole object or a single property.
        """
        if property_ is None:
            return self
        else:
            if not hasattr(self.example_sample, property_):
                raise ValueError(f"Property {property_} not available.")
            elif callable(getattr(self.example_sample, property_)):
                return RandomVariable(
                    samples=[getattr(sample, property_)() for sample in self.samples]
                )
            else:
                return RandomVariable(
                    samples=[getattr(sample, property_) for sample in self.samples]
                )

    def sample(self, N: int = 1):
        """
        Return N random samples of the random variable.
        :return: N Samples of the random variable.
        """
        if hasattr(self, "generator_func"):
            return self._return_N_new_samples_from_generator_func(N)
        else:
            return np.random.choice(self.samples, size=N, replace=True)

    def _return_N_samples(self, N):
        """
        Returns N samples of the random variable. If more than N samples are available,
        N samples are randomly selected. If less than N samples are available,
        the samples are extended to the number N.
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
        this property will generate samples based on the generator_func.
        :return: Generated amples of the random variable.
        """
        return self._return_N_new_samples_from_generator_func(N=1000)

    def pdf(self, x):
        """
        Calculate the probability density function of the random variable at x.
        Based on a kernel density estimate of the random variable.
        Only works for univariate random variables. TODO: Allow multivariate random variables.
        :param x: Value to evaluate the probability density function at.
        :return: Probability density function at x.
        """
        x = np.reshape(x, (1, -1))
        bandwith = 1e-2 * (np.max(self.samples) - np.min(self.samples))
        return (
            1
            / (len(self.samples) * bandwith * np.sqrt(2 * np.pi))
            * np.dot(
                np.ones((len(self.samples),)),
                np.exp(-0.5 * ((x - np.array(self.samples)[:, np.newaxis]) / bandwith) ** 2),
            )
        )

    def cdf(self, x):
        """
        Calculate the cumulative distribution function of the random variable at x.
        Based on the empirical cumulative distribution function of the random variable.
        Only works for univariate random variables. TODO: Allow multivariate random variables.
        :param x: Value to evaluate the cumulative distribution function at.
        :return: Cumulative distribution function at x.
        """
        return np.mean(np.array(self.samples)[:, np.newaxis] <= x, axis=0)

    @cached_property
    def expected_value(self):
        r"""
        Calculates expected value $\mu = E[X]$ of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        :return: Expected value $\mu$
        """
        return sum(self.samples) / len(self.samples)

    @cached_property
    def variance(self):
        r"""
        Calculates variance $\sigma^2 = E[(X - \mu)^2]$ of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        :return: Variance $\sigma^2$
        """
        return sum([(sample - self.expected_value) ** 2 for sample in self.samples]) / (
            len(self.samples) - 1
        )

    @cached_property
    def skewness(self):
        r"""
        Calculates skewness $\gamma = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^3 \\right]$
        of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        :return: Skewness $\gamma$
        """
        return (
            sum([(sample - self.expected_value) ** 3 for sample in self.samples])
            * len(self.samples)
            / ((len(self.samples) - 1) * (len(self.samples) - 2) * self.variance**1.5)
        )

    @cached_property
    def kurtosis(self):
        r"""
        Calculates kurtosis $\\beta = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^4 \\right]$
        of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
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
        string = f"<RandomVariable of type {self._type}"
        if hasattr(self, "generator_func"):
            string += f" with {self.generator_func.__name__} distribution"
        else:
            string += f" with custom distribution"
        return string

    def __repr__(self):
        return self.__str__()
