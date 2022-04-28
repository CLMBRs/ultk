"""Functions for analyzing and formatting the results of the simplicity/informativeness trade-off."""

import plotnine as pn
import pandas as pd

from altk.language.language import Language


def get_report() -> pd.DataFrame:
    """Runs statistical tests and returns a dataframe containing correlations of optimality with degrees of naturalness and means of optimality for each natural language."""


def get_dataframe(languages: list[Language]) -> pd.DataFrame:
    """Get a pandas DataFrame for a list of languages containing efficient communication data.

    Args:
        - languages: the list of languages for which to get efficient communication dataframe.

    Returns:
        - data: a pandas DataFrame with rows as individual languages, with the columns specifying their
            - communicative cost
            - cognitive complexity
            - satisfaction of the iff universal
            - Language type (natural or artificial)
    """
    data = []
    for lang in languages:
        point = (
            1 - lang.informativity,
            lang.complexity,
            lang.naturalness,
            "natural" if lang.is_natural() else "artificial",
        )
        data.append(point)

    data = pd.DataFrame(
        data=data,
        columns=[
            "comm_cost",
            "complexity",
            "naturalness",
            "Language",
        ],
    )

    # Pandas confused by mixed types int and string, so convert back.
    data[["comm_cost", "complexity", "naturalness"]] = data[
        ["comm_cost", "complexity", "naturalness"]
    ].apply(pd.to_numeric)

    return data


def get_tradeoff_plot(
    languages: list[Language], dominating_languages: list[Language]
) -> pn.ggplot:
    """Create the main plotnine plot for the communicative cost, complexity trade-off for the experiment.

    Returns:
        - plot: a plotnine 2D plot of the trade-off.
    """
    # data = self.get_dataframe(self.get_languages())
    data = get_dataframe(languages)
    pareto_df = get_dataframe(dominating_languages)
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="comm_cost", y="complexity"))
        + pn.scale_x_continuous(limits=[0, 1])
        + pn.geom_jitter(
            stroke=0,
            alpha=1,
            width=0.00,
            height=0.00,
            # mapping=pn.aes(size="Language", shape="Language", fill="Language"),
            mapping=pn.aes(size="Language", shape="Language", color="naturalness"),
        )
        + pn.geom_line(size=1, data=pareto_df)
        + pn.xlab("Communicative cost of languages")
        + pn.ylab("Complexity of languages")
        + pn.scale_color_cmap("cividis")
    )
    return plot
