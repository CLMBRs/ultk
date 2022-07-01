"""Functions for analyzing and formatting the results of the simplicity/informativeness trade-off."""

import plotnine as pn
import pandas as pd

from altk.language.language import Language


def get_report() -> pd.DataFrame:
    """Runs statistical tests and returns a dataframe containing correlations of optimality with degrees of naturalness and means of optimality for each natural language."""


def get_dataframe(languages: list[Language], columns: list[str] = None, subset: list[str] = None, repeats=None) -> pd.DataFrame:
    """Get a pandas DataFrame for a list of languages containing efficient communication data.

    Args:
        languages: the list of languages to map into a dataframe.

        columns: the list of keys to a language's `data` dictionary attribute, which will comprise the columns of the resulting dataframe. By default will use all items of each language's `data` dictionary.

        subset: the columns to subset for duplicates

    Returns:
        - data: a pandas DataFrame with rows as individual languages, with the columns specifying their data. 
    """
    data = pd.DataFrame(
        data=[(lang.data[k] for k in columns) for lang in languages],
        columns=columns,
    )

    # Pandas confused by mixed types int and string, so convert back.
    sample = {k:languages[0].data[k] for k in columns}
    data_types = {
        "string": [k for k in columns if isinstance(sample[k], str)],
        "float": [k for k in columns if isinstance(sample[k], float)],
        "int": [k for k in columns if isinstance(sample[k], int)],
        "bool": [k for k in columns if isinstance(sample[k], int)],
    }
    data[[data_types["float"], data_types["int"]]].apply(pd.to_numeric)

    # drop duplicates without counting
    if repeats == "drop":
        data = data.drop_duplicates(subset=subset)

    # drop but count duplicates
    elif repeats == "count":
        vcs = data.value_counts(subset=subset)
        data = data.drop_duplicates(subset=subset)
        data = data.sort_values(by=subset)
        data["counts"] = vcs.values

    elif repeats is not None:
        raise ValueError(
            f"the argument `repeats` must be either 'drop' or 'count'. Received: {repeats}"
        )

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
