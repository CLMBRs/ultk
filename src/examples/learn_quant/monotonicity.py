from ultk.effcomm.informativity import informativity
from ultk.language.grammar import GrammaticalExpression
from ultk.language.language import Language, aggregate_expression_complexity
from ultk.language.semantics import Meaning

from learn_quant.meaning import create_universe
from learn_quant.util import read_expressions
from learn_quant.quantifier import QuantifierUniverse, QuantifierModel
from concepts.contexts import Context

from itertools import product
import math
from scipy.stats import entropy
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import sparse

def switch_direction(bools: list[list[bool]], down = False) -> list[list[bool]]:
        """Switches the value of True to False and vice versa.

        Args:
            bool (list[list[bool]]): Accepts a list of lists of bools 

        Returns:
            list[list[bool]]: A list of lists of bools, flipped unless self.down == True. 
        """
        if down:
            return bools
        else:
            return np.invert(bools)
        
def get_truth_matrix(universe: QuantifierUniverse) -> pd.DataFrame:
    """Get a pandas dataframe that stores the truth vectors of all models in the quantifier universe

    Returns:
        pd.DataFrame: A dataframe of booleans arrays that track the indices of set items of A intersected by B.
    """
    truth_array = []
    for quantifier_model in universe.referents:
        truth_vector = tuple(True if x == '2' else False for x in quantifier_model.name)
        truth_array.append(truth_vector)
    truth_df = pd.concat([pd.DataFrame(universe.get_names()).rename(columns={0: 'names'}), pd.DataFrame(truth_array)], axis=1).set_index("names")
    return truth_df

def calculate_lattice(universe: QuantifierUniverse, down=False) -> Context:
        """ 

        Returns:
            Context: Created from ``objects``, ``properties``, and ``bools`` correspondence.

            objects: Iterable of object label strings.
            properties: Iterable of property label strings.
            bools: Iterable of ``len(objects)`` tuples of ``len(properties)`` booleans.

            See https://github.com/haberchr/concepts/blob/master/concepts/contexts.py

        """
        truth_df = get_truth_matrix(universe)
        objects = truth_df.index.astype(str).tolist()
        properties = [x for x in truth_df.columns]
        bools = list(truth_df.fillna(False).astype(bool).itertuples(index=False, name=None))
        bools = switch_direction(bools, down)
        return Context(objects, properties, bools)

def get_sub_structures(concept_lattice: Context, name: list[str]) -> set[str]:

        """Accepts a list of a singleton name to get a set of substructures of the indexed model, excluding the indexing model name.

        Returns:
            set(str): Tuple of substructures of the model indexed that exist in the universe.

        """

        return set(concept_lattice[name][0]) - set(name)

def has_sub_structure(concept_lattice: Context, name: list[str]) -> bool:

    """Identity function for a model having a substructure in the universe.

    Returns:
        bool: ``True`` if there is one or more substructures in the universe, otherwise ``False``.
    """

    return True if len(set(concept_lattice[name][0]) - set(name)) > 0 else False

def get_sub_structure_in_meaning(concept_lattice, name: list[str], meaning: Meaning) -> set[str]:

    """Identity function for a model having a substructure in a defined Meaning.

    Returns:
        set(str): Tuple of substructures of the model indexed that exist in the selected Meaning.
    """

    names = set(referent.name for referent in meaning.referents)

    return get_sub_structures(concept_lattice, name) & names

def has_sub_structure_in_meaning(concept_lattice, name: list[str], meaning: Meaning) -> bool:

    """Identity function for a model having a substructure in a defined Meaning.

    Returns:
        bool: ``True`` if there is one or more substructures in the universe, otherwise ``False``.
    """

    return True if len(get_sub_structure_in_meaning(concept_lattice, name, meaning)) > 0 else False

def upward_monotonicity_entropy(all_models, quantifier):
    """Measures degree of upward monotonicity of a quantifiers as
    1 - H(Q | true_pred) / H(Q) where H is (conditional) entropy, and true_pred is the
    variable over models saying whether there's a true _predecessor_ in the
    subset order.

    :param all_models: calculated matrix MxM, where a given row is an array of bools that correspond to whether a model at that index is a submodel
    :param quantifier: list of truth values, same len as models
    :returns scalar
    """

    submembership = all_models

    if np.all(quantifier) or not np.any(quantifier):
        return 1
    p_q_true = sum(quantifier) / len(quantifier)
    p_q_false = 1 - p_q_true
    q_ent = -p_q_true*np.log2(p_q_true) - p_q_false*np.log2(p_q_false)

    def get_preds(num_arr, num):
        """Given an array of ints, and an int, get all predecessors of the
        model corresponding to int.
        Returns an array of same shape as num_arr, but with bools
        """
        return num_arr[num,:]

    def has_true_pred(num_arr, y):
        return np.any(y * num_arr)    

    # where necessary?
    true_preds = np.where(np.dot(submembership, quantifier) >= 1, 1, 0)

    p_pred = sum(true_preds) / len(true_preds)
    p_nopred = 1 - p_pred

    # TODO: make this elegant! solve nan problems
    q_pred = sum(quantifier.T * true_preds) / len(quantifier)
    q_nopred = sum(quantifier.T * (1.0 - true_preds)) / len(quantifier)
    noq_pred = sum((1.0 - quantifier.T) * true_preds) / len(quantifier)
    noq_nopred = sum((1.0 - quantifier.T) * (1.0 - true_preds)) / len(quantifier)

    pred_logs = np.log2([noq_pred, q_pred] / p_pred)
    pred_logs[pred_logs == -np.inf] = 0
    nopred_logs = np.log2([noq_nopred, q_nopred] / p_nopred)
    nopred_logs[nopred_logs == -np.inf] = 0
    ent_pred = -np.nansum(np.array([noq_pred, q_pred]) * pred_logs)
    ent_nopred = -np.nansum(np.array([noq_nopred, q_nopred]) * nopred_logs)
    cond_ent = ent_pred + ent_nopred

    # return 0 if q_ent == 0 else 1 - (cond_ent / q_ent)
    return (1.0 - cond_ent / q_ent)[0,0]

def calculate_monotonicity(universe, expressions, down = False):

    metrics = {}
    concept_lattice = calculate_lattice(universe)

    membership = sparse.lil_matrix((len(universe),len(expressions)),dtype=int)
    submembership = sparse.lil_matrix((len(universe),len(universe)),dtype=int)
    model_dictionary = {}
    for model_id, model in enumerate(universe.referents):
        model_dictionary[model.name] = model_id
    for model_id, model in enumerate(universe.referents):
        for expression_id, quantifier_expression in enumerate(expressions):
            metrics[str(quantifier_expression)] = {}
            if model in quantifier_expression.meaning.referents:
                membership[model_id, expression_id] = 1
        sub_ids = list(map(model_dictionary.__getitem__, get_sub_structures(concept_lattice, [model.name])))
        for sub_id in sub_ids:
            submembership[sub_id,model_id] = 1
    
    membership = membership.todense()
    submembership = submembership.todense()

    for expression_id, quantifier_expression in enumerate(expressions):
        print("Calculating monotonicity for: ", quantifier_expression)
        metrics[str(quantifier_expression)]["monotonicity"] = upward_monotonicity_entropy(submembership, membership[:, expression_id])

    return metrics