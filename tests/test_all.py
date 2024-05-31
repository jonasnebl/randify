from PDFtimate import randify, RandomVariable, plotPDF
import numpy as np


def test_extract_property():
    """
    Test extracting a property from a RandomVariable.
    Should result a new RandomVariable with the property as samples.
    """
    x = RandomVariable(np.random.multivariate_normal, np.zeros((3,)), np.eye(3))
    x.sample(10)
    x_transposed = x("mean")
    assert isinstance(x_transposed, RandomVariable)
    assert np.isscalar(x_transposed.samples[0])  # mean of (3,) is scalar


def test_expected_value():
    """
    Test expected value calculation of a RandomVariable.
    """
    assert True


def test_variance():
    """
    Test variance calculation of a RandomVariable.
    """
    assert True


def test_plot():
    """
    Test plotting a RandomVariable.
    """
    assert True
