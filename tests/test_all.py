from randify import randify, RandomVariable, plot_pdf, plot_cdf
import numpy as np
import pytest


@pytest.fixture
def ranvar(request):
    return RandomVariable(np.random.normal, 0, 1)


@pytest.fixture
def multivariate_ranvar(request):
    return RandomVariable(np.random.multivariate_normal, mean=[0, 0], cov=[[1, 0.5], [0.5, 1]])


def test_statistical_measures(ranvar):
    """
    Test expected value, variance, skewness and kurtosis of a RandomVariable.
    """
    ATOL = 0.3
    assert np.isclose(ranvar.expected_value, 0, atol=ATOL)
    assert np.isclose(ranvar.variance, 1, atol=ATOL)
    assert np.isclose(ranvar.skewness, 0, atol=ATOL)
    assert np.isclose(ranvar.kurtosis, 3, atol=ATOL)


def test_randify(ranvar):
    """
    Test the output RandomVariable of a function randified with the randify decorator.
    """

    @randify(N=int(1e4))
    def foo(x1, x2):
        return x1 + x2

    y = foo(ranvar, ranvar)

    ATOL = 0.3
    assert np.isclose(y.expected_value, 0, atol=ATOL)
    assert np.isclose(y.variance, 4, atol=ATOL)
    assert np.isclose(y.skewness, 0, atol=ATOL)
    assert np.isclose(y.kurtosis, 3, atol=ATOL)


def test_pdf(ranvar):
    """
    Test the probability distribution function of a RandomVariable.
    """
    assert np.isclose(ranvar.pdf(0), 1 / np.sqrt(2 * np.pi), atol=0.3)
    assert np.isclose(ranvar.pdf(-np.inf), 0, atol=1e-6)
    assert np.isclose(ranvar.pdf(np.inf), 0, atol=1e-6)


def test_cdf(ranvar):
    """
    Test the cumulative distribution function of a RandomVariable.
    """
    assert np.isclose(ranvar.cdf(0), 0.5, atol=0.1)
    assert np.isclose(ranvar.cdf(-np.inf), 0, atol=1e-6)
    assert np.isclose(ranvar.cdf(np.inf), 1, atol=1e-6)


def test_call_ranvar(multivariate_ranvar):
    """
    Test the __call__ method of a RandomVariable to copy the RandomVariable
    or extract properties from the RandomVariable.
    """
    mean_ranvar = multivariate_ranvar("mean")
    assert isinstance(mean_ranvar, RandomVariable)
    assert "float" in mean_ranvar._type
    assert np.allclose(mean_ranvar.expected_value, 0, atol=0.3)


def test_print_ranvar(ranvar):
    """
    Test the existence of __str__ and __repr__ method of a RandomVariable.
    """
    assert isinstance(str(ranvar), str)
    assert isinstance(repr(ranvar), str)
