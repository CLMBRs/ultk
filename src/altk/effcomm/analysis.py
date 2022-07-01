"""Functions for analyzing and formatting the results of the simplicity/informativeness trade-off."""

import plotnine as pn
import pandas as pd

from altk.language.language import Language


def get_dataframe(languages: list[Language], columns: list[str] = None, subset: list[str] = ["complexity", "comm_cost"], duplicates: str = "leave") -> pd.DataFrame:
    """Get a pandas DataFrame for a list of languages containing efficient communication data.

    Args:
        languages: the list of languages to map into a dataframe.

        columns: the list of keys to a language's `data` dictionary attribute, which will comprise the columns of the resulting dataframe. By default will use all items of each language's `data` dictionary.

        subset: the columns to subset for duplicates

        duplicates: {"drop", "count", "leave"} whether to drop, count, or do nothing with duplicates. By default is set to "leave" which will leave duplicates in the dataframe.

    Returns:
        - data: a pandas DataFrame with rows as individual languages, with the columns specifying their data. 
    """
    if columns is None:
        columns = list(languages[0].data.keys())

    data = pd.DataFrame(
        data=[tuple(lang.data[k] for k in columns) for lang in languages],
        columns=columns,
    )

    # drop duplicates without counting
    if duplicates == "drop":
        data = data.drop_duplicates(subset=subset)

    # drop but count duplicates
    elif duplicates == "count":
        vcs = data.value_counts(subset=subset)
        data = data.drop_duplicates(subset=subset)
        data = data.sort_values(by=subset)
        data["counts"] = vcs.values

    elif duplicates not in ["drop", "count", "leave"]:
        raise ValueError(
            f"the argument `duplicates` must be either 'drop', 'count', 'leave'. Received: {duplicates}"
        )

    return data    
