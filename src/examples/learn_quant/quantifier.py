
from typing import Iterable, Union
import numpy as np
import pandas as pd
from altk.language.semantics import Referent
from dataclasses import dataclass

@dataclass(eq=True, frozen=True)
class QuantifierModel(Referent):
    """A quantifier model is a single referent that captures a particular interpretation of a quantifier meaning, which is a set of quantifier referents.

    Every quantifier model is a quadruple <M, A, B>, where M corresponds to all possible quantifier referents for a given communicative situation, A and B are differents sets of quantifier referents that correspond to the items of comparison in quantificational logic.
    """

    name: str = None
    M: frozenset = None
    A: frozenset = None
    B: frozenset = None

    def get_cardinalities(self) -> dict:
        return {"M": len(self.M), 
                "A": len(self.A), 
                "B": len(self.B), 
                }
