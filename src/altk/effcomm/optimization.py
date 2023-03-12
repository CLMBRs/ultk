"""Classes and functions for generating languages that optimize the simplicity/informativeness trade-off, e.g. via an iterative evolutionary algorithm."""

from abc import abstractmethod
import copy
import random
import math
from typing import Callable
from tqdm import tqdm
from typing import Any
from pathos.multiprocessing import ProcessPool
from altk.effcomm.sampling import rename_id
from altk.effcomm.tradeoff import pareto_optimal_languages
from altk.language.language import Expression, Language

##############################################################################
# Mutation
##############################################################################

"""Minimal API for defining mutations used by an EvolutionaryOptimizer."""


class Mutation:
    @abstractmethod
    @staticmethod
    def precondition(language: Language, **kwargs) -> bool:
        """Whether a mutation is allowed to apply to a language."""
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def mutate(language: Language, expressions: list[Expression], **kwargs) -> Language:
        """Mutate the language, possibly using a list of expressions."""
        raise NotImplementedError()


class RemoveExpression(Mutation):
    @staticmethod
    def precondition(language: Language, **kwargs) -> bool:
        return len(language) > 1

    @staticmethod
    def mutate(language: Language, expressions: list[Expression], **kwargs) -> Language:
        language.expressions.pop(random.randrange(len(language)))
        return language


class AddExpression(Mutation):
    @staticmethod
    def precondition(language: Language, **kwargs) -> bool:
        return True

    @staticmethod
    def mutate(language: Language, expressions: list[Expression], **kwargs) -> Language:
        language.add_expression(random.choice(expressions))
        return language


##############################################################################
# Evolutionary Optimizer
##############################################################################


class EvolutionaryOptimizer:
    """Class for approximating the Pareto frontier of languages optimizing the simplicity/informativity trade-off."""

    def __init__(
        self,
        objectives: dict[str, Callable[[Language], Any]],
        expressions: list[Expression],
        mutations: list[Mutation],
        sample_size: int,
        max_mutations: int,
        generations: int,
        lang_size: int,
        x: str = "comm_cost",
        y: str = "complexity",
    ):
        """Initialize the evolutionary algorithm configurations.

        The measures of complexity and informativity, the expressions, and the mutations are all specific to the particular semantic domain.

        Args:
            objectives: a dict of the two objectives to optimize for, e.g. simplicity and informativeness, of the form, e.g.
                {
                    "complexity": comp_measure,
                    "comm_cost": lambda l: 1 - inf_measure(l)
                }

            expressions:    a list of expressions from which to apply mutations to languages.

            mutations: a list of Mutation objects

            sample_size:  the size of the population at every generation.

            max_muatations:   between 1 and this number of mutations will be applied to a subset of the population at the end of each generation.

            generations:  how many iterations to run the evolutionary algorithm for.

            lang_size:    between 1 and this number of expressions comprise a language.

            proceses:     for multiprocessing.ProcessPool, e.g. 6.
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

        self.dominating_languages = None
        self.explored_languages = None

    def fit(
        self, seed_population: list[Language], id_start: int, explore: float = 0.0
    ) -> dict[str, Any]:
        """Computes the Pareto frontier, a set languages which cannot be both more simple and more informative.

        Uses pygmo's nondominated_front method for computing a population's best solutions to a multi-objective optimization problem.

        Args:
            seed_population: a list of languages representing the population at generation 0 of the algorithm.

            id_start: the number of languages generated in the experiment so far.

            explore: a float in [0,1] representing how much to optimize for fitness (optimality wrt pareto front of complexity and comm_cost), and how much to randomly explore.

        Returns:
            a dict of the estimated optimization solutions, as well as points explored along the way; of the form

                {
                "dominating_languages": list of languages as estimated solutions,
                "explored_languages": list of all the languages explored during the evolutionary algorithm,
                "id_start": updated number of languages generated in the experiment.
                }
        """
        languages = seed_population
        explored_languages = []

        for _ in tqdm(range(self.generations)):
            # Measure each generation
            for lang in languages:
                for m in self.objectives:
                    lang.data[m] = self.objectives[m](lang)

            explored_languages.extend(copy.deepcopy(languages))

            # Calculate dominating individuals
            dominating_languages = pareto_optimal_languages(languages, self.x, self.y)
            parent_result = sample_parents(
                dominating_languages, explored_languages, id_start, explore
            )
            parent_languages = parent_result["languages"]
            id_start = parent_result["id_start"]

            # Mutate dominating individuals
            mutated_result = self.sample_mutated(
                parent_languages, self.sample_size, self.expressions, id_start
            )
            languages = mutated_result["languages"]
            id_start = mutated_result["id_start"]

        return {
            "dominating_languages": dominating_languages,
            "explored_languages": explored_languages,
            "id_start": id_start,
        }

    def sample_mutated(
        self,
        languages: list[Language],
        amount: int,
        expressions: list[Expression],
        id_start: int,
    ) -> dict[str, Any]:
        """
        Arguments:
            languages: dominating languages of a generation

            amount: sample_size.

            expressions: the list of expressions

            id_start: the number of languages generatd in the experiment so far.

        Returns:
            a dict of the new population of languages of size=sample_size, and the updated id_start, of the form
            
                {
                "languages": (list of updated languages)
                "id_start": (updated length of languages)
                }

        """
        amount -= len(languages)
        amount_per_lang = int(math.floor(amount / len(languages)))
        amount_random = amount % len(languages)

        mutated_languages = []

        for language in languages:
            for i in range(amount_per_lang):
                num_mutations = random.randint(1, self.max_mutations)

                mutated_language = copy.deepcopy(language)
                id_start += 1
                mutated_language.data["name"] = rename_id(
                    mutated_language.data["name"], id_start
                )

                for j in range(num_mutations):
                    mutated_language = self.mutate(mutated_language, expressions)
                mutated_languages.append(mutated_language)

        # Ensure the number of languages per generation is constant

        for _ in range(amount_random):
            language = copy.deepcopy(random.choice(languages))
            id_start += 1
            language.data["name"] = rename_id(language.data["name"], id_start)

            mutated_languages.append(self.mutate(language, expressions))

        mutated_languages.extend(languages)

        return {"languages": mutated_languages, "id_start": id_start}

    def mutate(self, language: Language, expressions: list[Expression]) -> Language:
        """Randomly selects a mutation that is allowed to apply and applies it to a language.

        Args:
            language: the Language to mutate

            expressions: the list of all possible expressions. Some mutations need access to this list, so it is part of the mutation api.

        Returns:
            the mutated Language

        """
        possible_mutations = [
            mutation
            for mutation in self.mutations
            if mutation.precondition(
                language, lang_size=self.lang_size, objectives=self.objectives
            )
        ]
        mutation = random.choice(possible_mutations)
        return mutation.mutate(language, expressions)


def sample_parents(
    dominating_languages: list[Language],
    explored_languages: list[Language],
    id_start: int,
    explore: float,
) -> dict[str, Any]:
    """Use the explore parameter to explore possibly suboptimal areas of the language space.

    Args:
        dominating_languages: a list of the languages with current best fitness with respect to the objectives.

        explored_languages: a list of all languages encountered during the evolutionary algorithm.

        id_start: the number of languages generated in the experiment so far.

        explore: a float in `[0,1]` specifying how much to explore possibly suboptimal languages. If set to 0, `parent_languages` is just `dominating_languages`.

    Returns:
        a dict of the languages to serve as the next generation (after possible mutations) and updated id_start, of the form

            {
                "languages": (list of updated languages)
                "id_start": (updated length of languages)
            }
    """
    total_fit = len(dominating_languages)
    num_explore = int(explore * total_fit)

    fit_indices = list(range(total_fit))
    random.shuffle(fit_indices)
    fit_indices = fit_indices[: total_fit - num_explore]

    parent_languages = []
    for i in fit_indices:
        id_start += 1
        lang = dominating_languages[i]
        lang.data["name"] = rename_id(lang.data["name"], id_start)
        parent_languages.append(lang)

    langs_to_explore = random.sample(explored_languages, num_explore)
    for i in range(num_explore):
        id_start += 1
        lang = langs_to_explore[i]
        lang.data["name"] = rename_id(lang.data["name"], id_start)
        parent_languages.append(lang)

    return {"languages": parent_languages, "id_start": id_start}

