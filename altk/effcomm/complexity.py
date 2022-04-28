"""Functions to support measuring complexity in efficient communication analyses of languages.
"""

from abc import abstractmethod
from altk.language.language import Language

class ComplexityMeasure:

    """A class for defining how to measure the cognitive complexity of a language.
    """
    def __init__(self):
        raise NotImplementedError()

    def batch_complexity(self, langs: list[Language]) -> list[float]:
        """Measure the complexity of a list of languages."""
        return [self.language_complexity(lang) for lang in langs]

    @abstractmethod
    def language_complexity(self, language: Language) -> float:
        """Measure the complexity of a single language.
        """
        raise NotImplementedError
