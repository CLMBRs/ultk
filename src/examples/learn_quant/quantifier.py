
from typing import Iterable, Union
import numpy as np
import pandas as pd
from altk.language.semantics import Referent
from dataclasses import dataclass

@dataclass
class QuantifierModel(Referent):
    """A quantifier model is a single referent that captures a particular interpretation of a quantifier meaning, which is a set of quantifier referents."""

    name: str() = None
    M: set() = None
    A: set() = None
    B: set() = None

    def __post_init__(self):

        if not self.name and not (self.M and self.A and self.B):
            raise ValueError("Must initialize with either a name or sets for <M, A, B>")
        
        if self.M and self.A and self.B:
            self._update_name()

        elif self.name:
            self._update_sets()
    
    def _update_sets(self, name=None):

        if name:
            self.name = name

        self.M = set() 
        self.A = set()
        self.B = set()
        for index, i in enumerate(self.name):
            self.M.add(int(index))
            if self.name[index] == "0":
                self.A.add(int(index))
            if self.name[index] == "1":
                self.B.add(int(index))
            if self.name[index] == "2":
                self.A.add(int(index))
                self.B.add(int(index))
                
    def _update_name(self, M=None, A=None, B=None):

        if M:
            self.M = M
        if A:
            self.A = A
        if B:
            self.B = B 

        name_seq = []
        for i in sorted(self.M):
            i_in_both = i in self.A and i in self.B
            i_in_A = i in self.A and not i_in_both
            i_in_B = i in self.B and not i_in_both
            i_in_neither = (not i_in_A) and (not i_in_B)
            if i_in_A:
                name_seq.append("0")
            if i_in_B:
                name_seq.append("1")
            if i_in_both:
                name_seq.append("2")
            if i_in_neither:
                name_seq.append("3")
        self.name = "".join(name_seq)

    def update(self, **kwargs):

        self.__dict__.update((key, kwargs[key])
            for key in ('name', 'M', 'A', 'B') if key in kwargs)
        self._update_name()
        self._update_sets()

    def get_cardinalities(self) -> dict:
        return {"A": len(self.A), "B": len(self.B), "M": len(self.M)}
    
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