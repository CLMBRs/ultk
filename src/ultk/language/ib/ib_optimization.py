
from ultk.language.ib.ib_language import IBLanguage
from ultk.language.ib.ib_structure import IBStructure

import numpy as np

# The original paper "The information bottleneck method" uses x̃ instead of w, x instead of m, and y instead of u

# Calculate all of the normal function results for the meanings
def normal(language: IBLanguage, beta: float) -> np.ndarray:
    pass

# Do an interation of the BA Algorithm
def recalculate_language(language: IBLanguage, beta: float) -> IBLanguage:
    pass

def calculate_optimal(structure: IBStructure, beta: float) -> IBLanguage:
    pass