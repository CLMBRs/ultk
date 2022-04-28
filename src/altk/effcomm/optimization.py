"""Classes and functions for generating languages that optimize the simplicity/informativeness trade-off, e.g. via an iterative evolutionary algorithm."""

from abc import abstractmethod
import copy
import random
import math
from tqdm import tqdm
from pathos.multiprocessing import ProcessPool
from pygmo import non_dominated_front_2d
from altk.effcomm.complexity import ComplexityMeasure
from altk.effcomm.informativity import InformativityMeasure

from altk.language.language import Expression, Language

##############################################################################
# Mutation
##############################################################################

"""Minimal API for defining mutations used by an Evolutionary_Optimizer."""


class Mutation:
    @abstractmethod
    def precondition(self, language: Language, **kwargs) -> bool:
        """Whether a mutation is allowed to apply to a language."""
        raise NotImplementedError

    @abstractmethod
    def mutate(
        self, language: Language, expressions: list[Expression], **kwargs
    ) -> Language:
        """Mutate the language, possibly using a list of expressions."""
        raise NotImplementedError()


##############################################################################
# Evolutionary Optimizer
##############################################################################


class Evolutionary_Optimizer:
    """Class for approximating the Pareto frontier of languages optimizing the simplicity/informativity trade-off."""

    def __init__(
        self,
        comp_measure: ComplexityMeasure,
        inf_measure: InformativityMeasure,
        expressions: list[Expression],
        mutations: list[Mutation],
        sample_size: int,
        max_mutations: int,
        generations: int,
        lang_size: int,
        processes: int,
    ):
        """Initialize the evolutionary algorithm configurations.

        The measures of complexity and informativity, the expressions, and the mutations are all specific to the particular semantic domain.

        Args:
            - comp_measure:   a Complexity_Measure object

            - inf_measure:    an Informativity_Measure object

            - expressions:    a list of expressions from which to apply mutations to languages.

            - mutations: a list of Mutation objects

            - sample_size:  the size of the population at every generation.

            - max_muatations:   between 1 and this number of mutations will be applied to a subset of the population at the end of each generation.

            - generations:  how many iterations to run the evolutionary algorithm for.

            - lang_size:    between 1 and this number of expressions comprise a language.

            - proceses:     for multiprocessing.ProcessPool, e.g. 6.
        """
        self.comp_measure = comp_measure
        self.inf_measure = inf_measure
        self.expressions = expressions
        self.mutations = mutations

        self.sample_size = sample_size
        self.max_mutations = max_mutations
        self.generations = generations
        self.lang_size = lang_size
        self.processes = processes

        self.dominating_languages = None
        self.explored_languages = None

    def fit(self, seed_population: list[Language]) -> tuple[list[Language]]:
        """Computes the Pareto frontier, a set languages which cannot be both more simple and more informative.

        Uses pygmo's nondominated_front method for computing a population's best solutions to a multi-objective optimization problem.

        Args:
            - seed_population:      a list of languages representing the population at generation 0 of the algorithm.

        Returns:
            - dominating_languages:     a list of the Pareto optimal languages

            - explored_languages:       a list of all the languages explored during the evolutionatry algorithm.
        """
        batch_complexity = self.comp_measure.batch_complexity
        batch_comm_cost = self.inf_measure.batch_communicative_cost

        pool = ProcessPool(nodes=self.processes)  # TODO: remove until you need it

        languages = seed_population
        explored_languages = []

        for gen in tqdm(range(self.generations)):
            # Measure each generation
            # complexity = pool.map(batch_complexity, languages) # for some reason pool hates me
            # comm_cost = pool.map(batch_comm_cost, languages)

            complexity = batch_complexity(languages)
            comm_cost = batch_comm_cost(languages)

            explored_languages.extend(copy.deepcopy(languages))

            # Calculate dominating individuals
            dominating_indices = non_dominated_front_2d(
                list(zip(comm_cost, complexity))
            )
            dominating_languages = [languages[i] for i in dominating_indices]

            # Mutate dominating individuals
            languages = self.sample_mutated(
                dominating_languages, self.sample_size, self.expressions
            )

        return (dominating_languages, explored_languages)

    def sample_mutated(
        self, languages: list[Language], amount: int, expressions: list[Expression]
    ) -> list[Language]:
        """
        Arguments:
            - languages:   dominating languages of a generation
            - amount:      sample_size.
            expressions: the list of expressions
        Returns:
            - mutated_languages: a new population of languages of size=sample_size
        """
        amount -= len(languages)
        amount_per_lang = int(math.floor(amount / len(languages)))
        amount_random = amount % len(languages)

        mutated_languages = []

        for language in languages:
            for i in range(amount_per_lang):
                num_mutations = random.randint(1, self.max_mutations)
                mutated_language = copy.deepcopy(language)
                for j in range(num_mutations):
                    mutated_language = self.mutate(mutated_language, expressions)
                mutated_languages.append(mutated_language)

        # Ensure the number of languages per generation is constant
        for i in range(amount_random):
            language = random.choice(languages)
            mutated_languages.append(self.mutate(language, expressions))

        mutated_languages.extend(languages)

        return mutated_languages

    def mutate(self, language: Language, expressions: list[Expression]) -> Language:
        """Choose a mutation at random to apply to a language."""
        possible_mutations = [
            mutation
            for mutation in self.mutations
            if mutation.precondition(
                language,
                lang_size=self.lang_size,
                inf_measure=self.inf_measure,
            )
        ]
        mutation = random.choice(possible_mutations)
        return mutation.mutate(language, expressions)
