"""Functions for analyzing and formatting the results of the simplicity/informativeness trade-off."""

import numpy as np
import pandas as pd
from altk.language.language import Language
from scipy.stats import pearsonr, scoreatpercentile, ttest_1samp
from typing import Any


def get_dataframe(
    languages: list[Language],
    columns: list[str] = None,
    subset: list[str] = ["complexity", "comm_cost"],
    duplicates: str = "leave",
) -> pd.DataFrame:
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
        vcs = data.value_counts(subset=subset, sort=False)
        data = data.drop_duplicates(subset=subset)
        data = data.sort_values(by=subset)
        data["counts"] = vcs.values

    elif duplicates not in ["drop", "count", "leave"]:
        raise ValueError(
            f"the argument `duplicates` must be either 'drop', 'count', 'leave'. Received: {duplicates}"
        )

    return data


def pearson_analysis(
    data, predictor: str, property: str, num_bootstrap_samples=100
) -> dict[str, Any]:
    """Measures pearson correlation coefficient for naturalness with a property.

    Use nonparametric bootstrap for confidence intervals.

    Args:
        data: a DataFrame representing the pool of measured languages

        predictor: a string representing the column to measure pearson r with

        property: a string representing a column to measure pearson r with the predictor column

        num_bootstrap_samples: how many samples to bootstrap from the original data

    Returns:
        a dict of the pearson correlation coefficient for the predictor and the property, and bootstrapped confidence intervals for this coefficient, e.g.
        {
            "rho": (a float between -1 and 1),
            "confidence_intervals": (a pandas Dataframe with the columns [
                'bootstrap_sample_percent', 'low', 'high'
            ])
        }
    """
    min_percent = 0.01  # must be > 2/ len(data)
    intervals = 5
    boots = [int(item * 100) for item in np.geomspace(min_percent, 1.0, intervals)]
    confidence_intervals_df = pd.DataFrame(
        {"bootstrap_sample_percent": boots, "low": None, "high": None}
    )

    r, _ = pearsonr(data[property], data[predictor])
    for i, bootstrap_sample_percent in enumerate(
        np.geomspace(min_percent, 1.0, num=intervals)
    ):
        rhos = []
        for _ in range(num_bootstrap_samples):
            bootstrap_sample = data.sample(
                n=int(bootstrap_sample_percent * len(data)), replace=True
            )
            try:
                rho, _ = pearsonr(
                    bootstrap_sample[property],
                    bootstrap_sample[predictor],
                )
            except ValueError:
                print("MINIMUM SIZE OF DATA: ", int(2 / min_percent))
                print("SIZE OF DATA: ", len(data.index))
            rhos.append(rho)
        interval = scoreatpercentile(rhos, (2.5, 97.5))
        confidence_intervals_df.iloc[
            i, confidence_intervals_df.columns.get_loc("low")
        ] = interval[0]
        confidence_intervals_df.iloc[
            i, confidence_intervals_df.columns.get_loc("high")
        ] = interval[1]

    return {"rho": r, "confidence_intervals": confidence_intervals_df}


def trade_off_means(name: str, df: pd.DataFrame, properties: list) -> pd.DataFrame:
    """Get a dataframe with the mean tradeoff data.

    Args:
        name: a str representing the subset of the population to observe mean properties for, e.g. "natural" or "population".

        df: a pandas DataFrame containing data of a language population to take the means of.

        prperties: the properties to take means of, corresponding to columns of `df`.

    Examples:

    >>> natural_means = trade_off_means("natural_means", natural_data, properties)
    >>> population_means = trade_off_means("population_means", data, properties)
    >>> means_df = pd.concat([natural_means, dlsav_means, population_means]).set_index("name")
    >>> means_df
                        simplicity  complexity  informativity  optimality
        name
        natural_means       0.772222     16.4000       0.746296    0.952280
        population_means    0.681068     22.9631       0.525118    0.832010

    """
    means_dict = {prop: [df[prop].mean()] for prop in properties} | {"name": name}
    means_df = pd.DataFrame(data=means_dict)
    return means_df


def trade_off_ttest(
    sub_population: pd.DataFrame, population_means: dict, properties: list
) -> pd.DataFrame:
    """Get a dataframe with a single-samples t-test results for a subpopulation against the full population.

    This is useful if we want to compare the optimality of natural languages to the full population of languages in an experiment. Because the property of 'being a natural language' is categorical, we use a single-samples T test.

    Args:
        sub_population: a pandas DataFrame representing a subset of the population to take ttests against the full language population for `properties`.

        population_means: a dict containing properties as keys and the mean value of the full language population for that property.

        properties: a list of strings corresponding to columns of the `sub_population` DataFrame and keys of the `population_means` dict.

    Examples:

        >>> df = trade_off_ttest(natural_data, population_means, properties)
        >>> df
                                simplicity  complexity  informativity  optimality
            stat
            t-statistic          4.101937   -4.101937       3.126855    4.031027
            Two-sided p-value    0.014830    0.014830       0.035292    0.015720

    """
    data = {}
    for prop in properties:
        result = ttest_1samp(sub_population[prop], population_means[prop])
        data[prop] = [result.statistic, result.pvalue]

    df = pd.DataFrame(data)
    df["stat"] = ["t-statistic", "Two-sided p-value"]
    return df.set_index("stat")
