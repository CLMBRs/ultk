
from typing import Iterable, Union
import numpy as np
import pandas as pd
from altk.language.semantics import Referent, Universe
from dataclasses import dataclass, field
from concepts.contexts import Context

@dataclass(eq=True, frozen=True)
class QuantifierModel(Referent):
    """A quantifier model is a single referent that captures a particular interpretation of a quantifier meaning, which is a set of quantifier referents.

    Every quantifier model is a quadruple <M, A, B>, where M corresponds to all possible quantifier referents for a given communicative situation, A and B are differents sets of quantifier referents that correspond to the items of comparison in quantificational logic.
    """

    name: str = None
    M: frozenset = field(init=False)
    A: frozenset = field(init=False)
    B: frozenset = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'A', frozenset([i if x in ['0','2'] for i, x in enumerate(self.name)]))
        object.__setattr__(self, 'B', frozenset([i if x in ['1','2'] for i, x in enumerate(self.name)]))
        object.__setattr__(self, 'M', frozenset([i if x in ['0','1','2','3'] for i, x in enumerate(self.name)]))

    def get_cardinalities(self) -> dict:
        return {"M": len(self.M), 
                "A": len(self.A), 
                "B": len(self.B), 
                }


class QuantifierUniverse(Universe):

    def __init__(self, referents: Iterable[QuantifierModel], m_size: int, x_size: int, prior: dict[str, float] = None):
        super().__init__(referents, prior)
        self.m_size = m_size
        self.x_size = x_size

