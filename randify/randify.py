from .RandomVariable import RandomVariable
from time import perf_counter


def randify(duration: float = 1, N: int = -1, verbose: bool = False):
    """
    Decorator that takes a function foo and allows RandomVariables as input.
    Performs Monte Carlo simulation by evaluating foo on samples of the input RandomVariables.
    :param foo: function to transform probability distributions
    :param N: optional, number of iterations for Monte Carlo simulation.
        Default: -1 for automatic number of iterations.
    :param duration: optional, Approximate duration of the simulation in seconds.
        Used to determine N if N=-1.
        No effect if N!=-1 or one of the RandomVariables provides samples.
        Default: 1 second.
    :param verbose: optional, print additional information.
        Default: False
    :return: function that allows RandomVariables as input and returns RandomVariables as output
    """

    def randify_decorator(foo):
        def inner(*args, **kwargs):
            start_total = perf_counter()

            initial_arguments = list(args) + list(kwargs.values())
            arguments = initial_arguments.copy()  # altered arguments for each iteration

            # check which input variables are RandomVariables
            random_variable_indices = []
            for count, arg in enumerate(initial_arguments):
                if isinstance(arg, RandomVariable):
                    random_variable_indices.append(count)

            # evalute foo once to determine return structure and execution time
            for random_variable_index in random_variable_indices:
                arguments[random_variable_index] = initial_arguments[
                    random_variable_index
                ].sample()
            start = perf_counter()
            y = foo(*arguments)
            duration_foo = perf_counter() - start
            if N == -1:  # automatic number of iterations
                N_samples = int(duration / duration_foo)
            else:
                N_samples = N
            returns_tuple = isinstance(y, tuple)
            if returns_tuple:
                N_return = len(y)
                result_samples = [[] for _ in range(N_return)]
            else:
                N_return = 1
                result_samples = []

            # retrieve N_samples samples from Random variables
            input_samples = [None] * len(initial_arguments)
            for random_variable_index in random_variable_indices:
                input_samples[random_variable_index] = initial_arguments[
                    random_variable_index
                ]._return_N_samples(N_samples)

            # Monte-Carlo simulation
            for i in range(N_samples):
                # build function arguments
                for random_variable_index in random_variable_indices:
                    arguments[random_variable_index] = input_samples[random_variable_index][i]
                y = foo(*arguments)
                if returns_tuple:
                    for j in range(N_return):
                        result_samples[j].append(y[j])
                else:
                    result_samples.append(y)

            if verbose:
                print(
                    f"Randify: {N_samples} samples evaluated in {perf_counter() - start_total:.3f}s."
                )

            if returns_tuple:
                return tuple(RandomVariable(samples=result_samples[i]) for i in range(N_return))
            else:
                return RandomVariable(samples=result_samples)

        return inner

    return randify_decorator
