from typing import Callable, Any
import pandas as pd

from yaml import load, dump
from typing import Iterable, Union
import pickle

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from ultk.language.grammar import GrammaticalExpression
from ultk.language.language import Expression, Language
from ultk.language.semantics import Meaning, Universe

from learn_quant.grammar import quantifiers_grammar
from learn_quant.meaning import create_universe
from learn_quant.quantifier import QuantifierUniverse


def read_expressions(
    filename: str, universe: Universe = None, return_by_meaning=True
) -> tuple[list[GrammaticalExpression], dict[Meaning, Expression]]:
    """
    Read expressions from a YAML file.

    Args:
        filename (str): The path to the YAML file containing the expressions.
        universe (Universe, optional): The universe object to evaluate the expressions against. Defaults to None.
        return_by_meaning (bool, optional): Whether to return the expressions as a dictionary with meanings as keys. Defaults to True.

    Returns:
        tuple[list[GrammaticalExpression], dict[Meaning, Expression]]: A tuple containing the parsed expressions and, if return_by_meaning is True, a dictionary of expressions by their meanings.
    """
    quantifiers_grammar.add_indices_as_primitives(universe.x_size)
    print(quantifiers_grammar)

    with open(filename, "r") as f:
        expression_list = load(f, Loader=Loader)
    parsed_exprs = [
        quantifiers_grammar.parse(expr_dict["grammatical_expression"])
        for expr_dict in expression_list
    ]
    if universe is not None:
        [expr.evaluate(universe) for expr in parsed_exprs]
    by_meaning = {}
    if return_by_meaning:
        by_meaning = {expr.meaning: expr for expr in parsed_exprs}
    return parsed_exprs, by_meaning


def filter_expressions_by_rules(rules: list, expressions):
    """
    Filters a list of expressions based on a set of rules.

    Args:
        rules (list): A list of rules to filter the expressions.
        expressions: The list of expressions to be filtered.

    Returns:
        list: A filtered list of expressions that match the given rules.
    """
    return list(filter(lambda x: str(x) in rules, expressions))


import os
from pathlib import Path


def read_expressions_from_folder(
    folder: str, return_by_meaning=True
) -> tuple[list[GrammaticalExpression], dict[Meaning, Expression]]:
    """Read expressions from a YAML file in a specified folder.

    Args:
        folder (str): The path to the folder containing the expressions file and the universe file.
        return_by_meaning (bool, optional): Whether to return the expressions as a dictionary with meanings as keys. Defaults to True.

    Returns:
        tuple[list[GrammaticalExpression], dict[Meaning, Expression]]: A tuple containing the parsed expressions, a dictionary of expressions by meaning, and the universe object.

    Raises:
        FileNotFoundError: If the expressions file or the universe file is not found.
    """

    expressions_file = os.path.join(
        folder, "generated_expressions.yml"
    )  # replace 'filename.yaml' with your actual filename
    universe_file = os.path.join(
        folder, "master_universe.pkl"
    )  # replace 'universe.pkl' with your actual universe filename

    with open(universe_file, "rb") as f:
        universe = pickle.load(f)

    quantifiers_grammar.add_indices_as_primitives(universe.x_size)

    with open(expressions_file, "r") as f:
        expression_list = load(f, Loader=Loader)
    parsed_exprs = [
        quantifiers_grammar.parse(expr_dict["grammatical_expression"])
        for expr_dict in expression_list
    ]
    if universe is not None:
        [expr.evaluate(universe) for expr in parsed_exprs]
    by_meaning = {}
    if return_by_meaning:
        by_meaning = {expr.meaning: expr for expr in parsed_exprs}
    return parsed_exprs, by_meaning, universe


def save_quantifiers(
    expressions_by_meaning: dict[GrammaticalExpression, Any],
    out_path: str = "generated_expressions.yml",
):
    """
    Save the quantifiers expressions to a YAML file.

    Args:
        expressions_by_meaning (dict): A dictionary mapping GrammaticalExpression objects to their corresponding meanings.
        out_path (str, optional): The output file path. Defaults to "generated_expressions.yml".
    """

    print("Saving generated expressions to file...")
    print("Output path:", os.getcwd() / Path(out_path))
    with open(out_path, "w+") as outfile:
        dump(
            [
                expressions_by_meaning[meaning].to_dict()
                for meaning in expressions_by_meaning
            ],
            outfile,
            Dumper=Dumper,
        )


def save_inclusive_generation(
    expressions_by_meaning: dict[GrammaticalExpression, Any],
    master_universe: QuantifierUniverse,
    output_dir: str,
    m_size: int,
    x_size: int,
    depth: int,
):
    """
    Save the generated expressions and the master universe to files.

    Args:
        expressions_by_meaning (dict[GrammaticalExpression, Any]): A dictionary mapping GrammaticalExpressions to their meanings.
        master_universe (QuantifierUniverse): The master universe object.
        output_dir (str): The directory where the files will be saved.
        m_size (int): The size of M.
        x_size (int): The size of X.
        depth (int): The depth of the generated expressions.

    Returns:
        None
    """

    output_file = (
        Path(output_dir)
        / Path(
            "inclusive/"
            + "M"
            + str(m_size)
            + "_"
            + "X"
            + str(x_size)
            + "_"
            + str("d" + str(depth))
        )
        / Path("generated_expressions.yml")
    )
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print("Saving generated expressions...")
    save_quantifiers(expressions_by_meaning, output_file)

    # Create a new path for the pickle file
    pickle_output_file = Path(output_file).parent / "master_universe.pkl"

    # Open the file in write binary mode and dump the object
    with open(pickle_output_file, "wb") as f:
        pickle.dump(master_universe, f)

    print("Master universe has been pickled and saved to", pickle_output_file)
