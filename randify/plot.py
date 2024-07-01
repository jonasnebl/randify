import matplotlib.pyplot as plt
import numpy as np


def _multiple_plots_loop(plot_one_plot_foo, title, plot_expected_value, **kwargs):
    """
    Helper function for plotting multiple RandomVariables in one matplotlib figure.
    Arranges the plots in a grid
    :param plot_one_plot_foo: Function that plots the probability density function of one RandomVariable.
    :param plot_expected_value: If True, the expected value is plotted as a vertical line.
    :param **kwargs: RandomVariables to be plotted.
    """

    N_subplots = len(kwargs)
    if N_subplots == 0:
        raise ValueError("No RandomVariable provided to plot as keyword argument.")
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
            plot_one_plot_foo(ax, ranvar, ranvar_name, plot_expected_value)

        if counter_cols == subplots_cols - 1:
            counter_rows += 1
            counter_cols = 0
        else:
            counter_cols += 1

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def _plot_one_pdf(ax, ranvar, ranvar_name, plot_expected_value):
    """
    Plot the 1D probability density function of ONE RandomVariable object.
    :param **kwargs: RandomVariable to be plotted.
        Use the keyword to set the title for the plot.
        Example:
        plotPDF(x1=x1, 2=x2)
        This will make two subplots, one 1D plot for x1 and one plot for x2.
    :param plot_expected_value: If True, the expected value is plotted as a vertical line.
    :return: string, Title of the plot
    """
    range_ = np.max(ranvar._samples) - np.min(ranvar._samples)
    x_values = np.linspace(
        np.min(ranvar._samples) - 0.1 * range_, np.max(ranvar._samples) + 0.1 * range_, 100
    )
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


def plot_pdf(plot_expected_value: bool = True, **kwargs):
    """
    Plot multiple 1D probability density functions of RandomVariable objects.
    :param plot_expected_value: If True, the expected value is plotted as a vertical line.
    :param **kwargs: RandomVariables to be plotted.
    Use the keyword to set the title for the plot.
    Example:
        .. code-block:: python
            plotPDF(x1=x1, 2=x2)
        will make two subplots, one plot for x1 and one plot for x2.
    """
    _multiple_plots_loop(
        _plot_one_pdf,
        title="Probability density function",
        plot_expected_value=plot_expected_value,
        **kwargs,
    )
