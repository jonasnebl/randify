class RandomVariable:
    """
    Class extending an arbitrary python object to a random variable.
    A random variable can be in one of two modes:
    1. Sample mode: typically used for inferred random variables
    2. Generator mode: typically used for known distributions.
        Samples can be generated from a saved generator function.
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
            self.generator = args[0]
            self.generator_args = args[1:]
            self.generator_kwargs = kwargs
        elif "generator" in kwargs:
            self.mode = "generator"
            self.generator = kwargs.pop("generator")
            self.generator_args = []
            self.generator_kwargs = kwargs
        else:
            raise ValueError(
                "Invalid initialization. Either 'generator' or 'samples' must be provided."
            )

        # determine type
        if self.mode == "sample":
            self.example_sample = self.samples[0]
        elif self.mode == "generator":
            # calculate one sample to determine type
            self.example_sample = self.generator(*self.generator_args, **self.generator_kwargs)
        self.type_ = type(self.example_sample).__name__

    def sample(self, N: int):
        """
        Generate N samples of the random variable.
        :param N: Number of samples to generate. Overwrites existing samples.
        """
        if self.mode == "generator":
            try:  # E.g. numpy distributions allow a size argument, faster than list comprehension
                self.samples = list(
                    self.generator(*self.generator_args, **self.generator_kwargs, size=N)
                )
                assert len(self.samples) == N
                assert isinstance(self.samples[0], type(self.example_sample))
            except:
                self.samples = [
                    self.generator(*self.generator_args, **self.generator_kwargs) for _ in range(N)
                ]

    def __call__(self, property: str = None):
        """
        Returns whole object or single property as random variables.
        :param property: optional, Property to extract from the random variable.
        """
        if property is None:
            return self
        else:
            return RandomVariable(samples=[getattr(sample, property) for sample in self.samples])

    def expected_value(self):
        """
        Returns expected value of the random variable or the expected value of a property of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        """
        return sum(self.samples) / len(self.samples)

    def variance(self):
        """
        Returns variance of the random variable or the variance of a property of the random variable.
        Class of randomized object must implement __add__ and __truediv__ methods.
        """
        expected_value = self.expected_value()
        return sum([(sample - expected_value) ** 2 for sample in self.samples]) / (
            len(self.samples) - 1
        )

    def __str__(self) -> str:
        """
        Return a string representation of the random variable for printing.
        """
        string = f"<RandomVariable of type {self.type_}"
        if self.mode == "generator":
            string += f" with {self.generator.__name__} distribution"
        elif self.mode == "sample":
            string += f" with custom distribution"
        return string

    def __repr__(self) -> str:
        return self.__str__()
