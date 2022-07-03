"""Classes for modeling the literal meanings of a language.

    Meanings are modeled as things which map linguistic forms to objects of reference. The linguistic forms and objects of reference can be richly defined, but they are opaque to the meaning.

    In efficient communication analyses, simplicity and informativeness are typically properties of semantic aspects of a language. E.g., a meaning is simple if it is easy to represent, or to compress into some code; a meaning is informative if it is easy for a listener to recover a speaker's intended literal meaning.

    Examples:

        >>> from altk.language.syntax import Form
        >>> from altk.language.language import Expression, Language
        >>> form = Form('blue')
        >>> meaning = Color_Meaning() # some default meaning
        >>> expression = Expression(form, meaning)
        >>> lang = Language([expression])

"""

from typing import Any, Iterable


class Universe:

    """The universe is the set of possible referent objects for a meaning."""

    def __init__(self, objects: Iterable[Any]):
        self.objects = objects

    def __str__(self):
        objects_str = ",\n".join([str(point) for point in self.objects])
        return f"Universe: {objects_str}"

    def __eq__(self, __o: object) -> bool:
        """Returns true if the two universes are the same set."""
        return self.objects == __o.objects
    
    def __len__(self) -> int:
        return len(self.objects)


class Meaning:

    """A meaning picks out objects of the universe.
    
    There are several easy ways of modeling this. 
    
    On one familiar model from (e.g. predicate logic and formal semantics) a semantic value can be set, called a property: the set of objects of the universe satisfying that property. A meaning can be associated with the relevant subset of the universe, or its characteristic function.
    
    On some efficient communication analysis models, we use the concept of meaning to be a more general mapping of forms to objects of reference.

    A meaning is always a subset of the universe, because an expression may itself be underspecified: that is, the expression can be used to express different meanings. Sometimes these different literal meanings are not equally likely, in which it can be helpful to define a meaning as a distribution over objects in the universe.
    """

    def __init__(self, dist: dict[Any, float], universe: Universe) -> None:
        """A meaning is a probability distribution over a set of possible objects of reference. 
        
        Args: 
            dist: a dict with objects of reference as keys, and probabilities as values. The keys must be exactly the objects in `universe`.

            universe: a Universe object that defines the probability space for a meaning.
        """
        if set(dist.keys()) != set(universe.objects): 
            raise ValueError(f"The keys of `dist` must match the Universe exactly; received keys={dist.keys()}, universe={universe.objects}")

        self.dist = dist
        self.universe = universe