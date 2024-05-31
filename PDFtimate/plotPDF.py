import matplotlib.pyplot as plt
import numpy as np


def plotPDF(*args, **kwargs):
    """
    Plot the PDF of one or two RandomVariable objects.
    :param args: 1 or 2 RandomVariable objects
    """

    N_bins = 50

    if len(args) == 1:
        hist, bin_edges = np.histogram(args[0].samples, bins=N_bins, density=True)

        # add zeros to plot correctly
        bin_edges = np.concatenate(
            (bin_edges[[0]] - 0.01 * (np.max(bin_edges) - np.min(bin_edges)), bin_edges)
        )
        hist = np.concatenate(([0], hist, [0]))

        fig, ax = plt.subplots()
        ax.plot(bin_edges, hist, label="$p(x)$")
        ax.fill_between(bin_edges, hist, alpha=0.1)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$p(x)$")
        ax.set_ylim(0, np.max(hist) * 1.1)
    elif len(args) == 2:
        H, xedges, yedges = np.histogram2d(
            args[0].samples, args[1].samples, bins=N_bins, density=True
        )

        # add zeros to bin_edges to plot correctly
        xedges = np.concatenate((xedges[[0]] - 0.01 * (np.max(xedges) - np.min(xedges)), xedges))
        yedges = np.concatenate((yedges[[0]] - 0.01 * (np.max(yedges) - np.min(yedges)), yedges))
        H = np.pad(H, (1, 1), "constant", constant_values=(0, 0))

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        X, Y = np.meshgrid(xedges, yedges)
        ax.plot_surface(X, Y, H, cmap="plasma")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$p(x_1, x_2)$")
        ax.set_zlim(0, np.max(H) * 1.1)
    else:
        raise ValueError("plotPDF() takes 1 or 2 RandomVariables.")
    fig.suptitle("Probability Density Function")

    plt.show()
