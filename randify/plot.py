from functools import wraps
from .RandomVariable import RandomVariable
import matplotlib.pyplot as plt
import numpy as np


def _plot_decorator(foo):
    """
    Decorator for plotting multiple RandomVariables in one matplotlib figure.
    :param foo: Function that plots the probability density function of one RandomVariable.
    :return: Function that plots the probability density function of multiple RandomVariables.
    """

    @wraps(foo)
    def inner(*args, **kwargs):
        N_subplots = len(kwargs)
        subplots_rows = int(np.sqrt(N_subplots))
        subplots_cols = int(np.ceil(N_subplots / subplots_rows))
        counter_rows, counter_cols = 0, 0

        fig, axs = plt.subplots(
            subplots_rows, subplots_cols, figsize=(5 * subplots_cols, 5 * subplots_rows)
        )
        if subplots_rows == 1:
            axs = [axs]
        if subplots_cols == 1:
            axs = [axs]

        for i in range(subplots_rows * subplots_cols):
            if i >= N_subplots:
                axs[counter_rows][counter_cols].remove()
            else:
                ranvar_name, ranvar = list(kwargs.items())[i]

                ax = axs[counter_rows][counter_cols]

                title = foo(ax, ranvar, ranvar_name)

            if counter_cols == subplots_cols - 1:
                counter_rows += 1
                counter_cols = 0
            else:
                counter_cols += 1

        fig.suptitle(title, fontsize=16)

        plt.tight_layout()
        plt.show()

    return inner


@_plot_decorator
def plot_pdf(ax, ranvar, ranvar_name, plot_expected_value: bool = True):
    """
    Plot the 1D probability density function of one RandomVariable object.
    :param plot_expected_value: If True, the expected value is plotted as a vertical line.
    :param kwargs: RandomVariable to be plotted.
        Use the keyword to set the title for the plot.
        Example:
        plotPDF(x1=x1, 2=x2)
        This will make two subplots, one 1D plot for x1 and one plot for x2.
    """
    x_values = np.linspace(np.min(ranvar.samples), np.max(ranvar.samples), 100)
    pdf_values = ranvar.pdf(x_values)

    ax.plot(x_values, pdf_values, label="p(" + ranvar_name + ")")
    ax.fill_between(x_values, pdf_values, alpha=0.1)
    ax.set_xlabel(ranvar_name)
    ax.set_ylabel("p(" + ranvar_name + ")")
    ax.set_title(ranvar_name)
    ax.set_ylim(0, np.max(pdf_values) * 1.2)

    if plot_expected_value:
        ax.axvline(ranvar.expected_value, color="red", label="Expected value")
        ax.text(
            ranvar.expected_value,
            0.98 * ax.get_ylim()[1],
            f" E[{ranvar_name}]",
            color="red",
            verticalalignment="top",
        )

    return "Probability density function"
