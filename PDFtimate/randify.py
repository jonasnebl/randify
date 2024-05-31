from .RandomVariable import RandomVariable

N_default = 100000


def randify(foo, N=-1):
    """
    Decorator that takes a function foo and allows RandomVariables as input.
    Performs Monte Carlo simulation by evaluating foo on samples of the input RandomVariables.

    :param foo: function to transform probability distributions
    :param N: number of iterations for Monte Carlo simulation. -1 means automatic number of iterations.
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
                N_samples = N_default if N == -1 else N

            # calculate samples for RandomVariables om generator mode
            for random_variable_index in random_variable_indices:
                if arguments[random_variable_index].mode == "generator":
                    arguments[random_variable_index].sample(N_samples)

            # Monte-Carlo simulation
            for i in range(N_samples):
                # build function arguments
                for random_variable_index in random_variable_indices:
                    arguments[random_variable_index] = initial_arguments[
                        random_variable_index
                    ].samples[i]
                y = foo(*arguments)
                if i == 0:  # init result_samples on first iteration
                    returns_tuple = isinstance(y, tuple)
                    if returns_tuple:
                        N_return = len(y)
                        result_samples = [[y[i]] for i in range(N_return)]
                    else:
                        N_return = 1
                        result_samples = [y]
                elif returns_tuple:
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
