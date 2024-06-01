class RandomVariable:
    """
    Class extending an arbitrary python object to a random variable.
    A random variable can be in one of two modes:
    1. Sample mode: typically used for inferred random variables
    2. generator_func mode: typically used for known distributions.
        Samples can be generated from a saved generator_func function.
    """

    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], list):
            self.mode = "sample"
            self.samples = args[0]
        elif "samples" in kwargs:
            self.mode = "sample"
            self.samples = kwargs["samples"]
        elif args and callable(args[0]):
            self.mode = "generator"
            self.generator_func = args[0]
            self.generator_args = args[1:]
            self.generator_kwargs = kwargs
        elif "generator_func" in kwargs:
            self.mode = "generator"
            self.generator_func = kwargs.pop("generator_func")
            self.generator_args = []
            self.generator_kwargs = kwargs
        else:
            raise ValueError(
                "Invalid initialization. Either 'generator_func' or 'samples' must be provided."
            )

        # determine type
        if self.mode == "sample":
            self.example_sample = self.samples[0]
        elif self.mode == "generator":
            # calculate one sample to determine type
            self.example_sample = self.generator_func(*self.generator_args, **self.generator_kwargs)
        self.type_ = type(self.example_sample).__name__

    def sample(self, N: int):
        """
        Generate N samples of the random variable.
        :param N: Number of samples to generate. Overwrites existing samples.
        """
        if self.mode == "generator":
            try:  # E.g. numpy distributions allow a size argument, faster than list comprehension
                self.samples = list(
                    self.generator_func(*self.generator_args, **self.generator_kwargs, size=N)
                )
                assert len(self.samples) == N
                assert isinstance(self.samples[0], type(self.example_sample))
            except:
                self.samples = [
                    self.generator_func(*self.generator_args, **self.generator_kwargs)
                    for _ in range(N)
                ]

    def __call__(self, property: str = None):
        """
        Returns whole object or single property as random variables.
        :param property: optional, Property to extract from the random variable.
        :return: RandomVariable object representing the whole object or a single property.
        """
        if property is None:
            return self
        else:
            return RandomVariable(samples=[getattr(sample, property) for sample in self.samples])

    def expected_value(self):
        """
        Calculates expected value $\mu = E[X]$ of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        :return: Expected value $\mu$
        """
        return sum(self.samples) / len(self.samples)

    def variance(self):
        """
        Calculates variance $\sigma^2 = E[(X - \mu)^2]$ of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        :return: Variance $\sigma^2$
        """
        expected_value = self.expected_value()
        return sum([(sample - expected_value) ** 2 for sample in self.samples]) / (
            len(self.samples) - 1
        )

    def skewness(self):
        """
        Calculates skewness $\gamma = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^3 \\right]$
        of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        :return: Skewness $\gamma$
        """
        expected_value = self.expected_value()
        variance = self.variance()
        return (
            sum([(sample - expected_value) ** 3 for sample in self.samples])
            * len(self.samples)
            / ((len(self.samples) - 1) * (len(self.samples) - 2) * variance**1.5)
        )

    def kurtosis(self):
        """
        Calculates kurtosis $\\beta = E \left[ \left( \\frac{X -\mu}{\sigma} \\right)^4 \\right]$
        of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        :return: Kurtosis $\\beta$
        """
        expected_value = self.expected_value()
        variance = self.variance()
        return (
            sum([(sample - expected_value) ** 4 for sample in self.samples])
            * len(self.samples)
            * (len(self.samples) + 1)
            / (
                (len(self.samples) - 1)
                * (len(self.samples) - 2)
                * (len(self.samples) - 3)
                * variance**2
            )
        )

    def __str__(self):
        """
        Return a string representation of the random variable for printing.
        :return: String representation of the random variable.
        """
        string = f"<RandomVariable of type {self.type_}"
        if self.mode == "generator":
            string += f" with {self.generator_func.__name__} distribution"
        elif self.mode == "sample":
            string += f" with custom distribution"
        return string

    def __repr__(self):
        return self.__str__()
