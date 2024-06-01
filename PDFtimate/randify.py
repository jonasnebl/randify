from .RandomVariable import RandomVariable
from time import perf_counter


def randify(foo, N=-1, duration=2):
    """
    Decorator that takes a function foo and allows RandomVariables as input.
    Performs Monte Carlo simulation by evaluating foo on samples of the input RandomVariables.

    :param foo: function to transform probability distributions
    :param N: number of iterations for Monte Carlo simulation. -1 means automatic number of iterations.
    :param duration: Approximate duration of the simulation in seconds. Used to determine N if N=-1.
        No effect if N!=-1 or one of the RandomVariables provides samples.
    :return: function that allows RandomVariables as input and returns RandomVariables as output
    """

    def inner(*args, **kwargs):
        initial_arguments = list(args) + list(kwargs.values())
        arguments = initial_arguments.copy()  # altered arguments for each iteration

        # check which input variables are RandomVariables
        random_variable_indices = []
        for count, arg in enumerate(initial_arguments):
            if isinstance(arg, RandomVariable):
                random_variable_indices.append(count)

        # if no RandomVariables are present, return the static function result
        if not random_variable_indices:
            return foo(*args, **kwargs)
        else:
            # check if one of the RandomVariables provide samples to infer N_samples
            N_samples = -1
            for random_variable_index in random_variable_indices:
                if initial_arguments[random_variable_index].mode == "sample":
                    if N_samples == -1:
                        N_samples = len(initial_arguments[random_variable_index].samples)
                    else:
                        assert N_samples == len(
                            initial_arguments[random_variable_index].samples
                        ), "Inconsistent number of samples for RandomVariables"
            if N_samples == -1:  # no RandomVariable provided samples
                N_samples = N

            # first iteration
            for random_variable_index in random_variable_indices:
                initial_arguments[random_variable_index].sample(1)
                arguments[random_variable_index] = initial_arguments[random_variable_index].samples[
                    0
                ]
            start = perf_counter()
            for i in range(5):
                y = foo(*arguments)
            duration_foo = perf_counter() - start
            if N_samples == -1:
                N_samples = int(0.9 * 5 * duration / duration_foo)  # 0.9 is random empirical factor

            # calculate samples for RandomVariables on generator mode
            for random_variable_index in random_variable_indices:
                if initial_arguments[random_variable_index].mode == "generator":
                    initial_arguments[random_variable_index].sample(N_samples)
                    initial_arguments[random_variable_index].mode = "sample"

            # evaluate function return structure on first iteration
            returns_tuple = isinstance(y, tuple)
            if returns_tuple:
                N_return = len(y)
                result_samples = [[y[i]] for i in range(N_return)]
            else:
                N_return = 1
                result_samples = [y]

            # Monte-Carlo simulation
            for i in range(1, N_samples):
                # build function arguments
                for random_variable_index in random_variable_indices:
                    arguments[random_variable_index] = initial_arguments[
                        random_variable_index
                    ].samples[i]
                y = foo(*arguments)
                if returns_tuple:
                    for j in range(N_return):
                        result_samples[j].append(y[j])
                else:
                    result_samples.append(y)

            # return random variable objects for all return values
            if returns_tuple:
                return tuple([RandomVariable(samples=result_samples[i]) for i in range(N_return)])
            else:
                return RandomVariable(samples=result_samples)

    return inner
