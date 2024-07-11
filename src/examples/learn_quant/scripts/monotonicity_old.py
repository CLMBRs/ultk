from functools import lru_cache
import numpy as np
from pprint import pprint
import hydra
from omegaconf import DictConfig, OmegaConf

import random as rnd
from ..quantifier import QuantifierUniverse
from ..grammar import QuantifierGrammar
from ..meaning import create_universe
from ..util import save_quantifiers, save_inclusive_generation
from learn_quant.scripts.generate_expressions import generate_expressions

def upward_monotonicity_extensions(all_models, quantifier):
    """
    Measures degree of upward monotonocity of a quantifier as % of extensions of each true model that are also true.
    :param all_models: list of models
    :param quantifier: list of truth values, same len as all_models
    :return: scalar measure
    """
    if np.all(quantifier) or not np.any(quantifier):
        return 1
    props = []
    #only consider those models for which the quantifier is true (non zero returns indices)
    for i in np.nonzero(quantifier.flatten() == 1)[0]:
        model = all_models[i, :]
        tiled_model = np.tile(model, (len(all_models), 1))
        extends = np.all(tiled_model * all_models == tiled_model, axis=1).flatten()
        #proportion of true extensions of that model for the quantifier
        props.append(np.sum(quantifier[extends])/np.sum(extends))
    return np.mean(props)


def binary_to_int(arr):
    """ Converts a 2-D numpy array of 1s and 0s into integers, assuming each
    row is a binary number.  By convention, left-most column is 1, then 2, and
    so on, up until 2^(arr.shape[1]).

    :param arr: 2D numpy array
    :returns: 1D numpy array, length arr.shape[0], containing integers
    """
    return arr.dot(1 << np.arange(arr.shape[-1]))


def upward_monotonicity_entropy(all_models, quantifier):
    """Measures degree of upward monotonicity of a quantifiers as
    1 - H(Q | true_pred) / H(Q) where H is (conditional) entropy, and true_pred is the
    variable over models saying whether there's a true _predecessor_ in the
    subset order.

    :param all_models: list of models
    :param quantifier: list of truth values, same len as models
    :returns scalar
    """

    quantifier = quantifier.flatten()

    if np.all(quantifier) or not np.any(quantifier):
        return 1
    # uniform distributions
    p_q_true = sum(quantifier) / len(quantifier)
    p_q_false = 1 - p_q_true
    q_ent = -p_q_true*np.log2(p_q_true) - p_q_false*np.log2(p_q_false)

    # get integers corresponding to each model
    model_ints = binary_to_int(all_models)

    def get_preds(num_arr, num):
        """Given an array of ints, and an int, get all predecessors of the
        model corresponding to int.
        Returns an array of same shape as num_arr, but with bools
        """
        return num_arr & num == num_arr

    def num_preds(num_arr, num):
        preds = get_preds(num_arr, num).astype(int)
        return sum(preds)

    def has_true_pred(num_arr, quantifier, num):
        preds = get_preds(num_arr, num)
        return np.any(quantifier * preds)

    # vector of length quantifier, has a 1 if that model has a true
    # predecessor, 0 otherwise
    true_preds = np.vectorize(
        lambda num: has_true_pred(model_ints, quantifier, num)
    )(model_ints).astype(int)

    # TODO: how to handle cases where true_preds is all 0s or all 1s, i.e.
    # where every model does have a true predecessor?  In that case, we have
    # H(Q | pred) = H(Q), so currently would get degree 0
    """
    if np.all(true_preds) or not np.any(true_preds):
        # to avoid divide by zeros / conditioning on zero-prob
        # TODO: does this make sense???
        return q_ent
    """

    pred_weights =  np.vectorize(
        lambda num: num_preds(model_ints, num)
    )(model_ints)


    # print('q:')
    # print(quantifier)
    # print(true_preds)
    pred_prob = pred_weights / sum(pred_weights)
    # print(pred_weights)
    # print(pred_prob)
    # TODO: should these be weighted by pred_weights, i.e. pred_prob?
    p_pred = sum(true_preds) / len(true_preds)
    p_nopred = 1 - p_pred

    # TODO: make this elegant! solve nan problems
    q_pred = sum(quantifier * true_preds) / len(quantifier)
    q_nopred = sum(quantifier * (1 - true_preds)) / len(quantifier)
    noq_pred = sum((1 - quantifier) * true_preds) / len(quantifier)
    noq_nopred = sum((1 - quantifier) * (1 - true_preds)) / len(quantifier)

    pred_logs = np.log2([noq_pred, q_pred] / p_pred)
    pred_logs[pred_logs == -np.inf] = 0
    nopred_logs = np.log2([noq_nopred, q_nopred] / p_nopred)
    nopred_logs[nopred_logs == -np.inf] = 0
    ent_pred = -np.nansum(np.array([noq_pred, q_pred]) * pred_logs)
    ent_nopred = -np.nansum(np.array([noq_nopred, q_nopred]) * nopred_logs)
    cond_ent = ent_pred + ent_nopred
    # print(cond_ent)
    # print(q_ent)

    # return 0 if q_ent == 0 else 1 - (cond_ent / q_ent)
    return 1 - cond_ent / q_ent


@lru_cache(maxsize=None)
def monotonicity_memoized(models_string, quantifier_string):
    """models_string = models.tostring() for 2d int array models;
    quantifier_string = quantifier.tostring() for 1d int array """
    quantifier = np.frombuffer(quantifier_string, dtype=int)
    models = np.frombuffer(models_string, dtype=int).reshape((len(quantifier), -1))
    return measure_monotonicity(models, quantifier)


def measure_monotonicity(all_models, quantifier,
                         measure=upward_monotonicity_entropy):
    """ Measures degree of monotonicty, as max of the degree of
    positive/negative monotonicty, for a given quantifier _and its negation_
    (since truth values are symmetric for us).

    :param all_models: list of models
    :param quantifier: list of truth values
    :param measure: method for computing degree of upward monotonicity of a Q
    :return: max of measure applied to all_models and quantifier, plus 1- each
    of those
    """
    interpretations = [
        measure(all_models, quantifier),
        measure(all_models, 1 - quantifier),
        # downward monotonicity
        measure(1 - all_models, 1 - quantifier),
        measure(1 - all_models, quantifier)]
    return np.max(interpretations)
    # return interpretations


def quantifiers_in_order_of_monotonicity(universe,
                                         quantifiers,
                                         measure=upward_monotonicity_extensions):
    """
    Prints all quantifiers on models of length l in order of monotonicity.
    :param l: Max model size
    :return: None
    """
    models = universe.binarize_referents()
    mon_values = np.empty(shape=(len(quantifiers), ))
    binarized_meanings = np.empty(shape=(len(quantifiers), len(models)))
    expression_names = np.array([str(x) for x in quantifiers.values()])
    for idx, x in enumerate(quantifiers):
        binarized_meaning = x.get_binarized_meaning()
        binarized_meanings[idx] = binarized_meaning
        mon_values[idx] = measure_monotonicity(models, binarized_meaning, upward_monotonicity_entropy)
    order_indices = np.argsort(mon_values, axis=0)
    if not np.issubdtype(order_indices.dtype, np.integer):
        raise ValueError("order_indices should be an array of integers")
    with open("monotonicity_printout.txt", 'w') as f:
        with np.printoptions(threshold=np.inf):
            pprint([(quantifier, mon_value)
                    for quantifier, mon_value
                    in zip(expression_names[order_indices].tolist(), mon_values[order_indices].tolist())], stream=f)
            
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.save = False
    print(OmegaConf.to_yaml(cfg))

    quantifiers_grammar = QuantifierGrammar.from_yaml(cfg.grammar.path)
    universe = create_universe(m_size=cfg.universe.m_size, x_size=cfg.universe.x_size)
    quantifiers = generate_expressions(quantifiers_grammar, cfg, universe=universe)
    quantifiers_in_order_of_monotonicity(universe, quantifiers)


if __name__ == "__main__":
    main()
