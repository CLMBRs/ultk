"""Classes for modeling the meanings of a language.

    Meanings are modeled as things which map linguistic forms to objects of reference. The linguistic forms and objects of reference can be richly defined, but they are in some sense inaccessible at the level of a meaning. We think of meanings as informationally encapsulated instructions for building concepts.

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


class Referent:
    """A referent is an object of communication."""

    def __init__(self, name: str, weight: float = None) -> None:
        self.name = name
        self.weight = weight

    def __str__(self) -> str:
        raise NotImplementedError

    def __hash__(self) -> int:
        raise NotImplementedError


class Universe:

    """The universe is the set of possible referent objects for a meaning."""

    def __init__(self, referents: Iterable[Referent]):
        self.referents = referents

    def __str__(self):
        referents_str = ",\n".join([str(point) for point in self.referents])
        return f"Universe: {referents_str}"

    def __eq__(self, __o: object) -> bool:
        """Returns true if the two universes are the same set."""
        return self.referents == __o.referents

    def __len__(self) -> int:
        return len(self.referents)


class Meaning:

    """A meaning picks out objects of the universe.

    There are several easy ways of modeling this.

    On one familiar model from (e.g. predicate logic and formal semantics) a semantic value can be set, called a property: the set of objects of the universe satisfying that property. A meaning can be associated with the relevant subset of the universe, or its characteristic function.

    On some efficient communication analysis models, we use the concept of meaning to be a more general mapping of forms to objects of reference.

    A meaning is always a subset of the universe, because an expression may itself be underspecified: that is, the expression can be used to express different meanings. Sometimes these different literal meanings are not equally likely, in which it can be helpful to define a meaning explicitly as a distribution.
    """

    def __init__(self, referents: Iterable[Referent], universe: Universe) -> None:
        """A meaning is the set of things it refers to.

        The objects of reference are a subset of the universe of discourse. Sometimes it is natural to construe the meaning as as a probability distribution over the universe, instead of just a binary predicate.

        Args:
            dist: a dict with referents as keys, and probabilities as values. The keys must be exactly the referents in `universe`.

            universe: a Universe object that defines the probability space for a meaning.
        """
        if not set(referents).issubset(set(universe.referents)):
            raise ValueError(
                f"The set of referents for a meaning must be a subset of the universe of discourse."
            )

        self.referents = referents
        self.universe = universe
        self.dist = None

    def construct_distribution(self, weighted=False):
        """Construct the probability distributino associated with the meaning.

        By default, all elements in extension are assigned equal probability. If `weighted` set to true, elements are assigned probability according proportional to their weight attribute.

        Args:
            weighted: a bool representing what weight to assign all elements in the extension of the meaning a probability.
        """
        zeros = {
            ref.name: 0.0 for ref in set(self.universe.referents) - set(self.referents)
        }
        nonzeros = (
            self.weighted_distribution() if weighted else self.referents_uniform()
        )
        self.dist = nonzeros | zeros

    def referents_uniform(self):
        """Construct a probability distribution associated with the meaning such that every referent is equally weighted."""
        return {ref.name: 1 / len(self.referents) for ref in self.referents}

    def weighted_distribution(self):
        """Construct a probability distribution associated with the meaning according to the weights specified by each referent."""
        total_weight = sum([ref.weight for ref in self.referents])
        if not total_weight:
            # If there are no weights, make each referent equally likely.
            # One might also raise an error, complaining that there must be some nonzero weight in the space of referents.
            return self.referents_uniform()
        return {ref.name: ref.weight / total_weight for ref in self.referents}
