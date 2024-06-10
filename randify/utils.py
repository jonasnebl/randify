import numpy as np
from sklearn.neighbors import KernelDensity


def _extract_samples_from_ranvar(*args):
    """
    Helper function for extracting samples and concatenating them into a single array.
    :param *args: RandomVariable objects.
    :return: np.ndarray, Concatenated samples of all RandomVariable objects.
    """

    N_samples_per_ranvar = [len(ranvar.samples) for ranvar in args]
    sample_initial_shapes = [np.shape(ranvar.example_sample) for ranvar in args]
    samples_total = np.zeros((min(N_samples_per_ranvar), 0))
    for ranvar in args:
        samples_ranvar = ranvar.samples[: min(N_samples_per_ranvar)]
        samples_ranvar = np.array([np.reshape(sample, -1) for sample in samples_ranvar])
        samples_total = np.concatenate((samples_total, samples_ranvar), axis=1)

    return samples_total, sample_initial_shapes


def _extract_given_samples(*args, sample_inital_shapes):
    """
    Helper function for extracting given samples and concatenating them into a single array.
    :param *args: iterables of samples
    :return: np.ndarray, Concatenated samples of all RandomVariable objects.
    """
    if len(args) != len(sample_inital_shapes):
        raise ValueError("Number of given samples does not match number of random variables.")
    else:
        given_samples = None
        for ranvar_values, sample_initial_shape in zip(args, sample_inital_shapes):
            if np.shape(ranvar_values) == sample_initial_shape:
                if given_samples is None:
                    given_samples = np.reshape(ranvar_values, (1, -1))
                else:
                    given_samples = np.concatenate(
                        (given_samples, np.reshape(ranvar_values, (1, -1))), axis=1
                    )
            elif np.shape(ranvar_values)[1:] == sample_initial_shape:
                if given_samples is None:
                    given_samples = np.reshape(ranvar_values, (np.shape(ranvar_values)[0], -1))
                else:
                    given_samples = np.concatenate(
                        (
                            given_samples,
                            np.reshape(ranvar_values, (np.shape(ranvar_values)[0], -1)),
                        ),
                        axis=1,
                    )
            else:
                raise ValueError(
                    "Shape of given samples does not match shape of random variable samples."
                )

    return given_samples


def pdf(*args):
    """
    Calculate the probability density function of one or multiple RandomVariables.
    Based on a kernel density estimation using sklearn's KernelDensity of the random variable.
    Examples:
    .. code-block:: python
        x1 = RandomVariable(np.random.normal, loc=0, scale=1)
        x2 = RandomVariable(np.random.normal, loc=0, scale=1)
        x3 = RandomVariable(np.random.multivariate_normal, mean=np.zeros(2,), cov=np.eye(2))
        pdf(x1)(0) # will give the marginal pdf of x1 evaluated at x1=0.
        pdf(x1, x2)(0,0) # will give the joint pdf of x1 and x2 evaluated at x1=0, x2=0.
        pdf(x3)([0,0]) # will give the pdf of x3 evaluated at x3=[0,0].
        pdf(x1, x3)(0,[0,0]) # will give the joint pdf of x1 and x3 evaluated at x1=0, x3=[0,0].
    The function allows multiple values at once if they are stacked along the 0-th axis:
    .. code-block:: python
        pdf(x1, x3)([0,1], [[0,0], [1,1]]) # will give the pdf at x1=0, x3=[0,0] and x1=1, x3=[1,1].
    :param *args: RandomVariable objects to calculate the joint probability density function of.
    :return: Probability density function at given input.
    """
    samples_total, sample_inital_shapes = _extract_samples_from_ranvar(*args)
    kde = KernelDensity(kernel="gaussian", bandwidth=1e-2).fit(samples_total)

    def _pdf(*args):
        x = _extract_given_samples(*args, sample_inital_shapes=sample_inital_shapes)
        pdf_value = np.exp(kde.score_samples(x))
        if len(pdf_value) == 1:
            return pdf_value[0]
        else:
            return pdf_value

    return _pdf
