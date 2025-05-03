from learn_quant.util import read_expressions, create_universe
from learn_quant.quantifier import QuantifierUniverse, QuantifierModel
from learn_quant.grammar import get_indices_tag, QuantifierGrammar
from ultk.language.semantics import Universe

from typing import Dict
import numpy as np
from copy import deepcopy
import dill as pkl
from pathlib import Path

import hydra
from omegaconf import DictConfig
import mlflow

# e.g.:
# python -m learn_quant.monotonicity recipe=test_monotonicity
# python -m learn_quant.monotonicity recipe=4_4_5_xi grammar.indices=false


def binary_to_int(arr):
    """Converts a 2-D numpy array of 1s and 0s into integers, assuming each
    row is a binary number.  By convention, left-most column is 1, then 2, and
    so on, up until 2^(arr.shape[1]).

    :param arr: 2D numpy array
    :returns: 1D numpy array, length arr.shape[0], containing integers
    """
    return arr.dot(1 << np.arange(arr.shape[-1]))


def upward_monotonicity_entropy(
    all_models, set_reference_models, quantifier, cfg, flip=False
):
    """Measures degree of upward monotonicity of a quantifiers as
    1 - H(Q | true_pred) / H(Q) where H is (conditional) entropy, and true_pred is the
    variable over models saying whether there's a true _predecessor_ in the
    subset order.

    :param all_models: list of models
    :param quantifier: list of truth values, same len as models
    :returns scalar
    """

    quantifier = quantifier.flatten()

    model_ints = binary_to_int(all_models)
    ref_model_ints = binary_to_int(set_reference_models)

    if np.all(quantifier) or not np.any(quantifier):
        return 1
    # uniform distributions
    p_q_true = sum(quantifier) / len(quantifier)
    p_q_false = 1 - p_q_true
    q_ent = -p_q_true * np.log2(p_q_true) - p_q_false * np.log2(p_q_false)

    # vector of length quantifier, has a 1 if that model has a true
    # predecessor, 0 otherwise
    true_preds = get_true_predecessors(
        all_models, set_reference_models, quantifier, flip
    )

    # TODO: how to handle cases where true_preds is all 0s or all 1s, i.e.
    # where every model does have a true predecessor?  In that case, we have
    # H(Q | pred) = H(Q), so currently would get degree 0
    """
    if np.all(true_preds) or not np.any(true_preds):
        # to avoid divide by zeros / conditioning on zero-prob
        # TODO: does this make sense???
        print("All true_preds are the same")
        return q_ent
    """

    pred_weights = np.vectorize(
        lambda num, r_num: num_preds(model_ints, ref_model_ints, num, r_num, flip)
    )(model_ints, ref_model_ints)

    pred_prob = pred_weights / sum(pred_weights)
    # TODO: should these be weighted by pred_weights, i.e. pred_prob?
    p_pred = sum(true_preds) / len(true_preds)
    p_nopred = 1 - p_pred

    # TODO: make this elegant! solve nan problems
    q_pred = sum(quantifier * true_preds) / len(quantifier)
    q_nopred = sum(quantifier * (1 - true_preds)) / len(quantifier)
    noq_pred = sum((1 - quantifier) * true_preds) / len(quantifier)
    noq_nopred = sum((1 - quantifier) * (1 - true_preds)) / len(quantifier)

    pred_logs = np.log2(
        [noq_pred, q_pred] / p_pred, where=np.array([noq_pred, q_pred] / p_pred) > 0
    )
    pred_logs[pred_logs == -np.inf] = 0
    nopred_logs = np.log2(
        [noq_nopred, q_nopred] / p_nopred,
        where=np.array([noq_nopred, q_nopred] / p_nopred) > 0,
    )
    nopred_logs[nopred_logs == -np.inf] = 0
    ent_pred = -np.nansum(np.array([noq_pred, q_pred]) * pred_logs)
    ent_nopred = -np.nansum(np.array([noq_nopred, q_nopred]) * nopred_logs)
    cond_ent = ent_pred + ent_nopred

    # TODO: return 0 if q_ent == 0 else 1 - (cond_ent / q_ent)
    # TODO: integrate with a logger
    if cfg.measures.monotonicity.debug:
        print("q_ent", q_ent)
        print("p_pred", p_pred)
        print("p_nopred", p_nopred)
        print("q_pred", q_pred)
        print("q_nopred", q_nopred)
        print("noq_pred", noq_pred)
        print("noq_nopred", noq_nopred)
        print("pred_logs", pred_logs)
        print("nopred_logs", nopred_logs)
        print("ent_pred", ent_pred)
        print("ent_nopred", ent_nopred)
        print("cond_ent", cond_ent)
    return 1 - cond_ent / q_ent


def measure_monotonicity(
    all_models,
    set_reference_models_A,
    set_reference_models_B,
    quantifier,
    measure=upward_monotonicity_entropy,
    cfg=None,
):
    """Measures degree of monotonicity, as max of the degree of
    positive/negative monotonicty, for a given quantifier _and its negation_
    (since truth values are symmetric for us).

    :param all_models: list of models
    :param quantifier: list of truth values
    :param measure: method for computing degree of upward monotonicity of a Q
    :return: max of measure applied to all_models and quantifier, plus 1- each
    of those
    """

    def swap_models(all_models):
        # Get the number of rows and columns
        n, m = all_models.shape

        # Ensure that the number of columns is a multiple of 3
        if m % 3 != 0:
            raise ValueError("The number of columns must be a multiple of 3.")

        # Reshape the array to (n, number_of_groups, 3)
        reshaped = all_models.reshape(n, m // 3, 3)

        # Swap the first and second columns of each group
        reshaped[:, :, [0, 1]] = reshaped[:, :, [1, 0]]

        # Reshape back to the original 2D array shape
        swapped = reshaped.reshape(n, m)

        return swapped

    interpretations = [
        # upward monotonicity
        measure(
            all_models,
            set_reference_models_A,
            quantifier,
            cfg,
        ),  # right upward monotonicity
        measure(
            all_models,
            set_reference_models_B,
            quantifier,
            cfg,
        ),  # left upward monotonicity
        # downward monotonicity
        measure(
            all_models, set_reference_models_A, quantifier, cfg, flip=True
        ),  # right downward monotonicity
        measure(
            all_models, set_reference_models_B, quantifier, cfg, flip=True
        ),  # left downward monotonicity
    ]

    # assert measure(all_models, quantifier, cfg,) == measure(1- all_models, quantifier, cfg, flip=True) == measure(flipped_models - both_models, quantifier, cfg, flip=True)

    print("\t", interpretations)

    return interpretations
    # return np.max(np.clip(interpretations, 0.0, 1.0))


def calculate_measure(cfg, measure, expression, universe):
    if measure == "monotonicity":
        print("Calculating montonicity for expression: ", expression.term_expression)
        (
            all_models,
            set_reference_models_A,
            set_reference_models_B,
            quantifiers,
            expression_names,
        ) = get_verified_models([expression], universe)
        monotonicity = measure_monotonicity(
            all_models,
            set_reference_models_A,
            set_reference_models_B,
            quantifiers[0],
            upward_monotonicity_entropy,
            cfg=cfg,
            name=expression_names[0],
        )
        print("Monotonicity: ", monotonicity)
        mlflow.log_metric("monotonicity_entropic", float(monotonicity))
    if measure == "expression_depth":
        expression_depth = calculate_term_expression_depth(expression.term_expression)
        print("Monotonicity: ", monotonicity)
        mlflow.log_param("expression_depth", expression_depth)


def has_same_base_model(ref_model_ints, r_num):
    return ref_model_ints == r_num  # & (ref_model_ints < 7)


def get_preds(num_arr, num, flip=False):
    """Given an array of ints, and an int, get all predecessors of the
    model corresponding to int.
    Returns an array of same shape as num_arr, but with bools
    """
    if not flip:
        return num_arr & num == num_arr
    else:
        return num_arr & num == num


def has_true_pred(num_arr, ref_model_ints, quantifier, num, r_num, flip=False):
    preds = get_preds(num_arr, num, flip)
    # print(num)
    # print(np.stack([quantifier, preds, has_same_base_model(ref_model_ints, r_num)], axis=1))
    return np.any(quantifier * (preds * has_same_base_model(ref_model_ints, r_num)))


def num_preds(num_arr, ref_model_ints, num, r_num, flip=False):
    preds = get_preds(num_arr, num, flip).astype(int)
    return sum(preds & has_same_base_model(ref_model_ints, r_num))


def get_true_predecessors(all_models, set_reference_models, quantifier, flip=False):

    # get integers corresponding to each model
    model_ints = binary_to_int(all_models)
    ref_model_ints = binary_to_int(set_reference_models)

    # vector of length quantifier, has a 1 if that model has a true
    # predecessor, 0 otherwise
    true_preds = np.vectorize(
        lambda num, r_num: has_true_pred(
            model_ints, ref_model_ints, quantifier, num, r_num, flip
        )
    )(model_ints, ref_model_ints).astype(int)

    # Find indices where true_preds equals 1
    """
    if not flip:
        indices = np.where(true_preds == 1)[0]
        if indices.size > 0:
            # Check that for every row in ref_model_ints corresponding to these indices,
            # all values are 1. This assumes ref_model_ints is 2D, with each row a model.
            # np.all(..., axis=1) returns a boolean array for each row; then np.all() on that
            # boolean array checks that every flagged row is entirely ones.
            if np.all(np.all(set_reference_models[indices] == 1, axis=1)):
                true_preds[indices] = 0
    """

    return true_preds


def filter_universe(cfg: DictConfig, uni: Universe):
    """This function filters out quantifier referents with certain digits in their names.
    It is necessary for the entropic measure of degree of monotonicity to work properly.

    Args:
        cfg (DictConfig): Hydra config
        uni (Universe): A universe of referents to filter

    Returns:
        Universe: A filtered universe
    """
    if cfg.measures.monotonicity.universe_filter and any(
        cfg.measures.monotonicity.universe_filter
    ):
        test_referents = tuple(
            ref
            for ref in deepcopy(uni.referents)
            if not any(
                str(digit) in ref.name
                for digit in cfg.measures.monotonicity.universe_filter
            )
        )
        print("Size of filtered universe: ", len(test_referents))
        uni = QuantifierUniverse(
            referents=test_referents,
            m_size=cfg.expressions.universe.m_size,
            x_size=cfg.expressions.universe.x_size,
        )
    return uni


def load_universe(cfg):
    try:
        uni = pkl.load(
            open(
                Path.cwd()
                / Path("learn_quant/outputs")
                / Path(cfg.expressions.target)
                / Path("master_universe.pkl"),
                "rb",
            )
        )
    except FileNotFoundError:
        print("Creating universe")
        uni = create_universe(
            cfg.expressions.universe.m_size, cfg.expressions.universe.x_size
        )

    print("Size of universe: ", len(uni.referents))
    return uni


def load_grammar(cfg):
    if hasattr(cfg.expressions.grammar, "typed_rules"):
        print("Loading rules from module...")
        primitives_grammar = QuantifierGrammar.from_module(
            cfg.expressions.grammar.typed_rules.module_path
        )
        quantifiers_grammar = QuantifierGrammar.from_yaml(cfg.expressions.grammar.path)
        grammar = quantifiers_grammar | primitives_grammar
    else:
        grammar = QuantifierGrammar.from_yaml(cfg.expressions.grammar.path)
    return grammar


def get_verified_models(expressions, uni):
    print("Binarizing referents")
    all_models = uni.binarize_referents(mode="set_vectors_w_padding")
    set_reference_models_A = uni.binarize_referents(mode="A")
    set_reference_models_B = uni.binarize_referents(mode="B")

    # Create quantifiers like in original code
    print("Creating quantifiers")
    quantifiers = np.array(
        [
            [
                expression.meaning.mapping[uni.referents[x]]
                for x in range(len(uni.referents))
            ]
            for expression in expressions
        ],
        dtype=int,
    )
    expression_names = np.array(
        [expression.term_expression for expression in expressions]
    )
    return (
        all_models,
        set_reference_models_A,
        set_reference_models_B,
        quantifiers,
        expression_names,
    )


def calculate_term_expression_depth(expression):
    depth = 0
    max_depth = 0
    for char in expression:
        if char == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ")":
            depth -= 1
    return max_depth


import numpy as np
import pandas as pd


@hydra.main(version_base=None, config_path="conf", config_name="learn")
def main(cfg: DictConfig) -> None:

    if not cfg.measures.monotonicity.create_universe:
        uni = load_universe(cfg)
    else:
        print("Couldn't load the universe. Falling back to creating universe")
        uni = create_universe(
            cfg.measures.monotonicity.universe.m_size,
            cfg.measures.monotonicity.universe.x_size,
        )
    # Filter out referents with certain digits in their names (e.g. 3, 4)
    # EDIT: This is not recommended - M-only idices are intuitively
    # required to be represented to obtain interpretations of monotoncity that match intuition
    # uni = filter_universe(cfg, uni)

    grammar = load_grammar(cfg)

    from omegaconf import OmegaConf

    print(OmegaConf.to_yaml(cfg))

    print("Reading expressions")

    indices_tag = get_indices_tag(indices=cfg.expressions.grammar.indices)
    expressions, _ = read_expressions(
        Path.cwd()
        / Path("learn_quant/outputs")
        / Path(cfg.expressions.target)
        / f"generated_expressions{indices_tag}.yml",
        uni,
        add_indices=cfg.expressions.grammar.indices,
        grammar=grammar,
    )

    print("Finished loading expressions")

    # All models should not use binarize referents, but "get_truth_matrix"
    # Use b only and both?
    (
        all_models,
        set_reference_models_A,
        set_reference_models_B,
        quantifiers,
        expression_names,
    ) = get_verified_models(expressions, uni)

    # Select only quantifiers in the expression list (unless 'all' specified)
    if "all" not in cfg.measures.expressions:
        mask = np.isin(expression_names, cfg.measures.expressions)
        indices = np.where(mask)[0]
        quantifiers = quantifiers[indices]
        expression_names = expression_names[indices]

    print(expression_names)

    print("Measuring monotonicity")

    from tqdm import tqdm

    if "all" in cfg.measures.monotonicity.direction:
        mon_values = np.empty(shape=(len(quantifiers), 4))
        for i in tqdm(range(len(quantifiers))):
            mon_values[i] = measure_monotonicity(
                all_models,
                set_reference_models_A,
                set_reference_models_B,
                quantifiers[i],
                upward_monotonicity_entropy,
                cfg,
            )
        # order_indices = np.argsort(mon_values, axis=0)[::-1]
        print("Monotonicity values len:", len(mon_values))
        print("expression_names len:", len(expression_names))
        print("quantifiers len:", len(quantifiers))

        print(mon_values.shape)
        print(expression_names.shape)
        print(quantifiers.shape)

        df_mon = pd.DataFrame(
            {
                "expression_name": expression_names,
                "right_upward": mon_values[:, 0],
                "left_upward": mon_values[:, 1],
                "right_downward": mon_values[:, 2],
                "left_downward": mon_values[:, 3],
                "degree": np.max(np.clip(mon_values, 0.0, 1.0), axis=1),
            }
        )

        df_mon.to_csv(
            Path.cwd()
            / Path("learn_quant/outputs")
            / Path(cfg.expressions.target)
            / Path("monotonicity_values.csv")
        )


if __name__ == "__main__":
    main()
