import plotnine as pn
import pandas as pd
import numpy as np
from util import save_plot


def plot_accuracy(fn: str, accuracies: list[float]) -> None:
    """Construct and save a basic plotnine line plot of accuracy vs rounds."""
    data = pd.DataFrame(data={"accuracy": accuracies, "round": range(len(accuracies))})
    # Set data and the axes
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="round", y="accuracy"))
        + pn.geom_line(size=1, data=data)
        + pn.ylim(0, 1)
        + pn.xlab("Round")
        + pn.ylab("Accuracy")
        + pn.scale_color_cmap("cividis")
    )
    save_plot(fn, plot)


def plot_complexity(fn: str, complexities: list[float]) -> pn.ggplot:
    """Get a basic plotnine line plot of complexities vs rounds."""
    data = pd.DataFrame(
        data={"complexity": complexities, "round": range(len(complexities))}
    )
    # Set data and the axes
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="round", y="complexity"))
        + pn.geom_line(size=1, data=data)
        + pn.ylim(0, 1)
        + pn.xlab("Round")
        + pn.ylab("Complexity")
        + pn.scale_color_cmap("cividis")
    )
    save_plot(fn, plot)


def plot_tradeoff(
    fn: str, complexities: list, accuracies: list, hamming_rd_bound: list[tuple[float]]
) -> pn.ggplot:
    """Get a basic plotnine point plot of languages in a complexity vs comm_cost 2D plot against the hamming rate distortion bound."""
    rounds = range(len(complexities))
    data = pd.DataFrame(
        data={
            "complexity": complexities,
            "comm_cost": [1 - acc for acc in accuracies],
            "round": rounds,
        }
    )

    bound_data = pd.DataFrame(
        hamming_rd_bound,
        columns=["complexity", "comm_cost"],
    )
    plot = (
        # Set data and the axes
        pn.ggplot(data=data, mapping=pn.aes(x="complexity", y="comm_cost"))
        + pn.scale_y_continuous(limits=[0, 1])
        + pn.geom_point(  # langs
            stroke=0, alpha=1, mapping=pn.aes(color="round")  # might be too faint
        )
        + pn.geom_line(  # bound
            data=bound_data,
        )
        + pn.xlab("Cognitive cost")
        + pn.ylab("Communicative cost")
        + pn.scale_color_cmap("cividis")
    )
    save_plot(fn, plot)


def plot_distribution(fn: str, dist: np.ndarray) -> pn.ggplot:
    """Create a bar plot of a distribution over states, e.g. the communicative need distribution or the ground truth distribution."""
    states = list(range(1, len(dist) + 1))
    data = pd.DataFrame(data={"state": states, "prob": dist.tolist()})
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="state", y="prob"))
        + pn.scale_x_discrete(limits=states)
        + pn.ylim(0, 1)
        + pn.geom_bar(stat="identity")
        + pn.xlab("State")
        + pn.ylab("Probability")
    )
    save_plot(fn, plot)
