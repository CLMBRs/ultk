from altk.effcomm.informativity import informativity
from altk.language.grammar import GrammaticalExpression
from altk.language.language import Language, aggregate_expression_complexity
from altk.language.semantics import Meaning

from learn_quant.meaning import create_universe
from learn_quant.util import read_expressions
from concepts.contexts import Context

from itertools import product
import math
from scipy.stats import entropy
import pandas as pd
import numpy as np
from tqdm import tqdm


class MonotonicityMeasurer:
    
    def __init__(self, universe, monotone_set='A', down=False):
        self.down = down
        self.universe = universe
        self.monotone_set = monotone_set
        self.concept_lattice = self._calculate_lattice()
    
    def _get_names(self) -> pd.DataFrame:
        return [referent.name for referent in self.universe.referents]

    def _get_truth_matrix(self) -> pd.DataFrame:
        truth_array = []
        for quantifier_model in self.universe.referents:
            truth_array.append(quantifier_model.get_truth_vector())
        truth_df = pd.concat([pd.DataFrame(self._get_names()).rename(columns={0: 'names'}), pd.DataFrame(truth_array)], axis=1).set_index("names")
        return truth_df

    def _switch_direction(self, bools, down):
        if down:
            return bools
        else:
            return np.invert(bools)

    def _calculate_lattice(self) -> Context:
        truth_df = self._get_truth_matrix()
        objects = truth_df.index.tolist()
        properties = [x for x in truth_df.columns]
        bools = list(truth_df.fillna(False).astype(bool).itertuples(index=False, name=None))
        bools = self._switch_direction(bools, self.down)
        return Context(objects, properties, bools)
    
    def _get_sub_structures(self, name: list[str]):
        return self.concept_lattice[name][0]
    
    def _has_sub_structure(self, name: list[str]):
        return True if len(self.concept_lattice[name][0]) > 1 else False
    
    def _get_sub_structure_in_meaning(self, name: list[str], meaning: Meaning):

        names = set(referent.name for referent in meaning.referents)

        return set(self.concept_lattice[name][0]).intersection(names)
    
    def _has_sub_structure_in_meaning(self, name: list[str], meaning: Meaning):

        return True if len(self._get_sub_structure_in_meaning(name, meaning)) > 0 else False
    
    def __call__(self, expressions):
        metrics = {}
        for idx, _ in enumerate(expressions):
            expression = expressions[idx]
            meaning = expression.meaning
            names_in_quantifier_meaning = set(referent.name for referent in meaning.referents)

            if len(names_in_quantifier_meaning) == len(self.universe):
                continue

            metrics[expression] = {}

            has_sub_structure = []

            for name in tqdm(names_in_quantifier_meaning):

                if self._has_sub_structure_in_meaning([name], meaning) == True:
                    has_sub_structure.append(True)
                else:
                    has_sub_structure.append(False)
             
            is_in_quantifier = [True] * len(names_in_quantifier_meaning)
            is_in_quantifier.extend([False] * (len(self.universe.referents)-len(names_in_quantifier_meaning)))
            has_sub_structure.extend([False] * (len(self.universe.referents)-len(names_in_quantifier_meaning)))

            metrics[expression]["H_x"] = entropy(is_in_quantifier, base=2)
            metrics[expression]["H_x|H_y"] = entropy(is_in_quantifier, has_sub_structure, base=2)
            metrics[expression]["monotonicity"] = 1-(metrics[expression]["H_x|H_y"]/metrics[expression]["H_x"])
        
        self.metrics = metrics