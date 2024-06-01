from .RandomVariable import RandomVariable
import matplotlib.pyplot as plt
import numpy as np


def x_value_helper(bin_edges):
    """
    Helper function to extend bin_edges with zeros for plotting vertical lines.
    Also calculates the x values for the bin centers instead of the bin edges.
    :param bin_edges: Bin edges from np.histogram or np.histogram2d.
    """
    bin_edges_extended = np.concatenate(
        (
            bin_edges[[0]] - 1e-3 * (np.max(bin_edges) - np.min(bin_edges)),
            bin_edges,
            bin_edges[[-1]] + 1e-3 * (np.max(bin_edges) - np.min(bin_edges)),
        )
    )
    return 0.5 * (bin_edges_extended[:-1] + bin_edges_extended[1:])


def plotPDF(plot_expected_value: bool = True, **kwargs):
    """
    Plot the probability density function of one or two RandomVariable objects.
    :param plot_expected_value: If True, the expected value is plotted as a vertical line.
    :param kwargs: RandomVariable to be plotted.
        Either pass one arg for 1D plot or a tuple of two variables for 2D plot:
        Use the keyword to set the title for the plot.
        Example:
        plotPDF(x1=x1,
                x2_x2=(x2, x3))
        This will make two subplots, one 1D plot for x1 and one 2D plot for x2 and x3.
    """

    N_subplots = len(kwargs)
    subplots_rows = int(np.sqrt(N_subplots))
    subplots_cols = int(np.ceil(N_subplots / subplots_rows))
    counter_rows, counter_cols = 0, 0

    N_bins = 50

    fig, axs = plt.subplots(
        subplots_rows, subplots_cols, figsize=(5 * subplots_cols, 5 * subplots_rows)
    )
    if subplots_rows == 1:
        axs = [axs]
    if subplots_cols == 1:
        axs = [axs]

    fig.suptitle("Probability Density Function")

    for i in range(subplots_rows * subplots_cols):
        if i >= N_subplots:
            axs[counter_rows][counter_cols].remove()
        else:
            arg_name, arg = list(kwargs.items())[i]

            ax = axs[counter_rows][counter_cols]

            if isinstance(arg, RandomVariable):
                hist, bin_edges = np.histogram(arg.samples, bins=N_bins, density=True)

                x_values = x_value_helper(bin_edges)
                hist = np.concatenate(([0], hist, [0]))

                ax.plot(x_values, hist, label="$p(x)$")
                ax.fill_between(x_values, hist, alpha=0.1)
                ax.set_xlabel("$x$")
                ax.set_ylabel("$p(x)$")
                ax.set_title(arg_name)
                ax.set_ylim(0, np.max(hist) * 1.2)

                if plot_expected_value:
                    expected_value = arg.expected_value()
                    ax.axvline(expected_value, color="red", label="Expected value")
                    ax.text(
                        expected_value,
                        0.98 * ax.get_ylim()[1],
                        f" E[{arg_name}]",
                        color="red",
                        verticalalignment="top",
                    )

            elif isinstance(arg, tuple) and len(arg) == 2:
                ax.remove()
                ax = fig.add_subplot(
                    subplots_rows,
                    subplots_cols,
                    counter_rows * subplots_cols + counter_cols + 1,
                    projection="3d",
                )

                H, xedges, yedges = np.histogram2d(
                    arg[0].samples, arg[1].samples, bins=N_bins, density=True
                )

                # add zeros to bin_edges to plot correctly
                x_values = x_value_helper(xedges)
                y_values = x_value_helper(yedges)

                H = np.pad(H, (1, 1), "constant", constant_values=(0, 0))

                X, Y = np.meshgrid(x_values, y_values)
                ax.plot_surface(X, Y, H, cmap="plasma")
                ax.set_xlabel("$x_1$")
                ax.set_ylabel("$x_2$")
                ax.set_zlabel("$p(x_1, x_2)$")
                ax.set_title(arg_name)
                ax.set_zlim(0, np.max(H) * 1.2)

                if plot_expected_value:
                    expected_value_x1 = arg[0].expected_value()
                    expected_value_x2 = arg[1].expected_value()
                    ax.stem(
                        [expected_value_x1],
                        [expected_value_x2],
                        [ax.get_zlim()[1]],
                        linefmt="red",
                        markerfmt="red",
                    )
                    ax.text(
                        expected_value_x1,
                        expected_value_x2,
                        0.98 * ax.get_zlim()[1],
                        f" E[{arg_name[0]}]",
                        color="red",
                        verticalalignment="top",
                    )

            else:
                raise ValueError(
                    "plotPDF() takes only one RandomVariable or tuples"
                    "of two RandomVariable objects as arguments."
                )

        if counter_cols == subplots_cols - 1:
            counter_rows += 1
            counter_cols = 0
        else:
            counter_cols += 1

    plt.tight_layout()
    plt.show()
