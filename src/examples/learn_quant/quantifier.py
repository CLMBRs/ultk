
from typing import Iterable, Union
import numpy as np
import pandas as pd
from altk.language.semantics import Referent
from dataclasses import dataclass

@dataclass(eq=False)
class QuantifierModel(Referent):
    """A quantifier model is a single referent that captures a particular interpretation of a quantifier meaning, which is a set of quantifier referents.

    Every quantifier model is a quadruple <M, A, B, X>, where M corresponds to all possible quantifier referents for a given communicative situation, A and B are differents sets of quantifier referents that correspond to the items of comparison in quantificational logic, and X corresponds to the space of all possible referents that are a superset or equal to the referents in M. That is, X is conceptually the set of possible referents in any quantification expression modeled in the universe.

    A scalar index (single referent in a set M, A, B) from [0, X) corresponds to a unique referent in a quantifier model.

    X is conceptually a set but is listed as an integer that denotes the length of an array [0, X).
    """

    name: str = None
    M: set = None
    A: set = None
    B: set = None
    X: int = None

    def __post_init__(self):

        if not self.name and not (self.M and self.A and self.B):
            raise ValueError("Must initialize with either a 'name' of length X or sets for <M, A, B> and an int value for X")
        
        if self.M and self.A and self.B:
            self._update_name()
        elif self.name:
            self._update_sets()
    
    def _update_sets(self, name=None):

        if name:
            self.name = name

        self.X = len(name)

        self.M = set() 
        self.A = set()
        self.B = set()
        for index, i in enumerate(self.name):
            if self.name[index] == "0":
                self.A.add(int(index))
                self.M.add(int(index))
            if self.name[index] == "1":
                self.B.add(int(index))
                self.M.add(int(index))
            if self.name[index] == "2":
                self.A.add(int(index))
                self.B.add(int(index))
                self.M.add(int(index))
            if self.name[index] == "3":
                self.M.add(int(index))
                
    def _update_name(self, **kwargs):

        allowed_keys = {"M", "A", "B", "X"}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        name_seq = []
        for i in range(0, self.X):
            in_both = i in self.A and i in self.B
            in_A = i in self.A and not in_both
            in_B = i in self.B and not in_both
            in_M = i in self.M
            in_neither = (not in_A) and (not in_B) and (not in_both)

            counter = 0
            if in_A:
                name_seq.append("0")
            if in_B:
                name_seq.append("1")
            if in_both:
                name_seq.append("2")
            if in_neither and in_M:
                name_seq.append("3")
            if in_neither and not in_M:
                name_seq.append("4")

        assert len(name_seq) == self.X
            
        self.name = "".join(name_seq)

    def update(self, **kwargs):

        self.__dict__.update((key, kwargs[key])
            for key in ('name', 'M', 'A', 'B', "X") if key in kwargs)
        self._update_name()
        self._update_sets()

    def get_cardinalities(self) -> dict:
        return {"M": len(self.M), 
                "A": len(self.A), 
                "B": len(self.B), 
                "X": self.X}
    
"""
qm = QuantifierModel(M=set([1,2,3,4]), A=set([1,4]), B=set([2,3]))
qm.name

qm2 = QuantifierModel("0030")
qm2.A = None
qm2.A
qm2._update_sets("1111")
qm2._update_sets("0011")

qm3 = QuantifierModel("13022")
qm3
"""