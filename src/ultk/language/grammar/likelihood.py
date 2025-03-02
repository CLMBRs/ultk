from typing import Callable, TypeVar, Iterable
from ultk.language.grammar.grammar import GrammaticalExpression
from ultk.language.semantics import Referent
from math import log

T = TypeVar("T")
Dataset = Iterable[tuple[Referent, T]]


def all_or_nothing(data: Dataset, tree: GrammaticalExpression) -> float:
    """Basic all or nothing likelihood, return 1 if all data are correctly predicted, 0 otherwise

    Args:
        data (Dataset): data for likelihood calculation
        tree (GrammaticalExpression): GrammaticalExpression for likelihood calculation

    Returns:
        float: likelihood
    """
    return float(all(tree(datum[0]) == datum[1] for datum in data))


def percent_match(data: Dataset, tree: GrammaticalExpression) -> float:
    """Basic percentage-based likelihood, returns the percent of matches across the output from the tree
    and the expected output from the data

    Args:
        data (Dataset): data for likelihood calculation
        tree (GrammaticalExpression): GrammaticalExpression for likelihood calculation

    Returns:
        float: likelihood
    """
    return sum([tree(datum[0]) == datum[1] for datum in data]) / len(data)


def percent_match_unique(data: Dataset, tree: GrammaticalExpression) -> float:
    """Basic percentage-based likelihood, returns the percent of matches across the output from the tree
    and the expected output from the data. However, if all of the outputs of the tree are the same returns 0.

    Args:
        data (Dataset): data for likelihood calculation
        tree (GrammaticalExpression): GrammaticalExpression for likelihood calculation

    Returns:
        float: likelihood
    """
    first_value = None
    same = True
    total_matches = 0
    for datum in data:
        val = tree(datum[0])
        if first_value is None:
            first_value = val
        elif same and val != first_value:
            same = False
        total_matches += int(val == datum[1])
    if same:
        return 0
    return total_matches / len(data)


def noise_match(
    possible_outputs: int, alpha: float = 0.01
) -> Callable[[Dataset, GrammaticalExpression], float]:
    """Taken from Piantadosi et al. Attempts to discern the probability by believing that the output is correct
    and was passed through a noise function which has an `alpha` chance to corrupt each item in the output list.

    Takes in the number of possible values the output can be and the percent chance of a corruption and returns a
    probability function which `mh_sample` is able to use.

    Specifically for log_mh_sample only.

    See also: https://github.com/piantado/LOTlib3/blob/master/Hypotheses/Likelihoods/BinaryLikelihood.py

    Args:
        possible_ouputs (int): The number of possible values an output is able to be
        alpha (float): The percentage chance that a value will be mutated

    Returns:
        Callable likelihood function:
            Args:
                data (Dataset): Data for likelihood calculation
                tree (GrammaticalExpression): GrammaticalExpression for likelihood calculation
            Returns:
                float: Likelihood in log probability
    """
    correct_chance = log(1 - alpha + alpha / possible_outputs)
    incorrect_chance = log(alpha / possible_outputs)

    def noise_match_probability(data: Dataset, tree: GrammaticalExpression) -> float:
        matches = sum([tree(datum[0]) == datum[1] for datum in data])
        return (len(data) - matches) * (incorrect_chance) + matches * (correct_chance)

    return noise_match_probability


def aggregate_individual_likelihoods(
    likelihood_function: Callable[[tuple[Referent, T], GrammaticalExpression], float],
) -> Callable[[Dataset, GrammaticalExpression], float]:
    """Takes in a likelihood function for an individual datum (in log probability) returns a likelihood function which calls the
    individual probability function and calls it across the dataset, summing it to get the final probability.

    Specifically for log_mh_sample only.

    Args:
        Callable individual likelihood function:
            Args:
                datum (Tuple[Referent, T]): An individual element from the dataset, the first element is the input, the second the output.
                tree (GrammarticalExpression): GrammaticalExpression for likelihood calculation
            Returns:
                float: Likelihood in log probability.

    Returns:
        Callable likelihood function:
            Args:
                data (Dataset): Data for likelihood calculation
                tree (GrammaticalExpression): GrammaticalExpression for likelihood calculation
            Returns:
                float: Likelihood in log probability
    """

    def output(data: Dataset, tree: GrammaticalExpression) -> float:
        output = 0
        for datum in data:
            output += likelihood_function(datum, tree)
        return output

    return output
