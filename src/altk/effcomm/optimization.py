"""Classes and functions for generating languages that optimize the simplicity/informativeness trade-off, e.g. via an iterative evolutionary algorithm."""

from abc import abstractmethod
import copy
import random
import math
from typing import Callable
from tqdm import tqdm
from pathos.multiprocessing import ProcessPool
from altk.effcomm.tradeoff import batch_measure, pareto_optimal_languages
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
        objectives: dict[str, Callable],
        expressions: list[Expression],
        mutations: list[Mutation],
        sample_size: int,
        max_mutations: int,
        generations: int,
        lang_size: int,
        processes: int,
        x: str = "comm_cost",
        y: str = "complexity",        
    ):
        """Initialize the evolutionary algorithm configurations.

        The measures of complexity and informativity, the expressions, and the mutations are all specific to the particular semantic domain.

        Args:
            - objectives: a dict of the two objectives to optimize for, e.g. simplicity and informativeness, of the form, e.g.
                {
                    "complexity": comp_measure,
                    "comm_cost": lambda l: 1 - inf_measure(l)
                }

            - expressions:    a list of expressions from which to apply mutations to languages.

            - mutations: a list of Mutation objects

            - sample_size:  the size of the population at every generation.

            - max_muatations:   between 1 and this number of mutations will be applied to a subset of the population at the end of each generation.

            - generations:  how many iterations to run the evolutionary algorithm for.

            - lang_size:    between 1 and this number of expressions comprise a language.

            - proceses:     for multiprocessing.ProcessPool, e.g. 6.
        """
        self.objectives = objectives
        self.x = x
        self.y = y
        self.expressions = expressions
        self.mutations = mutations

        self.sample_size = sample_size
        self.max_mutations = max_mutations
        self.generations = generations
        self.lang_size = lang_size
        self.processes = processes

        self.dominating_languages = None
        self.explored_languages = None

    def fit(self, seed_population: list[Language], explore: float = 0.0) -> dict[str, list[Language]]:
        """Computes the Pareto frontier, a set languages which cannot be both more simple and more informative.

        Uses pygmo's nondominated_front method for computing a population's best solutions to a multi-objective optimization problem.

        Args:
            seed_population: a list of languages representing the population at generation 0 of the algorithm.

            explore: a float in [0,1] representing how much to optimize for fitness (optimality wrt pareto front of complexity and comm_cost), and how much to randomly explore.

        Returns:
            a dict of the estimated optimization solutions, as well as points explored along the way; of the form
            {
                "dominating_languages": list of languages as estimated solutions,
                "explored_languages": list of all the languages explored during the evolutionary algorithm.
            }
        """
        languages = seed_population
        explored_languages = []

        for gen in tqdm(range(self.generations)):
            # Measure each generation
            for lang in languages:
                for m in self.objectives:
                    lang.measurements[m] = self.objectives[m](lang)

            explored_languages.extend(copy.deepcopy(languages))

            # Calculate dominating individuals
            dominating_languages = pareto_optimal_languages(languages, self.x, self.y)
            parent_languages = sample_parents(dominating_languages, explored_languages, explore)

            # Mutate dominating individuals
            languages = self.sample_mutated(
                parent_languages, self.sample_size, self.expressions
            )

        return {
            "dominating_languages": dominating_languages,
            "explored_languages": explored_languages,
        }

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
            )
        ]
        mutation = random.choice(possible_mutations)
        return mutation.mutate(language, expressions)


def sample_parents(dominating_languages: list[Language], explored_languages: list[Language], explore: float) -> list[Language]:
    """Use the explore parameter to explore possibly suboptimal areas of the language space."""
    total_fit = len(dominating_languages)
    num_explore = int(explore * total_fit)

    fit_indices = list(range(total_fit))
    random.shuffle(fit_indices)
    fit_indices = fit_indices[:total_fit - num_explore]
    
    parent_languages = [dominating_languages[i] for i in fit_indices] + random.sample(explored_languages, num_explore)

    return parent_languages
