from yaml import dump
from time import time
import argparse
from copy import deepcopy
import pickle
import os
import csv
import pathlib
from pathlib import Path

try:
    from yaml import CDumper as Dumper
except ImportError:
    from yaml import Dumper

import hydra
from omegaconf import DictConfig, OmegaConf

from ultk.language.grammar import GrammaticalExpression
from typing import Any

from ..quantifier import QuantifierUniverse
from ..grammar import QuantifierGrammar
from ..meaning import create_universe
from ..util import save_quantifiers, save_inclusive_generation


def enumerate_quantifiers(
    depth: int,
    quantifiers_universe: QuantifierUniverse,
    quantifiers_grammar: QuantifierGrammar,
) -> dict[GrammaticalExpression, Any]:

    expressions_by_meaning = quantifiers_grammar.get_unique_expressions(
        depth,
        max_size=2 ** len(quantifiers_universe),
        unique_key=lambda expr: expr.evaluate(quantifiers_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )

    # filter out the trivial meaning, results in NaNs
    # iterate over keys, since we need to change the dict itself
    for meaning in list(expressions_by_meaning.keys()):
        if len(meaning.referents) == 0:
            del expressions_by_meaning[meaning]

    return expressions_by_meaning


def generate_expressions(quantifiers_grammar: QuantifierGrammar, cfg: DictConfig):

    quantifiers_grammar.add_indices_as_primitives(
        cfg.universe.m_size, cfg.universe.weight
    )
    quantifiers_universe = create_universe(cfg.universe.m_size, cfg.universe.x_size)
    expressions_by_meaning = enumerate_quantifiers(
        cfg.grammar.depth, quantifiers_universe, quantifiers_grammar
    )

    outpath = (
        Path().cwd()
        / Path(cfg.output)
        / Path("M" + str(cfg.universe.m_size))
        / Path("X" + str(cfg.universe.x_size))
        / Path("d" + str(cfg.grammar.depth))
        / Path("generated_expressions.yml")
    )
    if cfg.save:
        save_quantifiers(expressions_by_meaning, outpath)


def generation_time_trial(quantifiers_grammar: QuantifierGrammar, cfg: DictConfig):

    time_log = pathlib.Path(cfg.output) / pathlib.Path(
        "generated_expressions_time_log.csv"
    )

    # Remove old time log
    if os.path.exists(time_log):
        os.remove(time_log)

    # Flush times as they are output to time log
    with open(time_log, "a+", newline="") as f:

        writer = csv.writer(f)
        writer.writerow(
            [
                "m_size",
                "x_size",
                "depth",
                "universe_size",
                "elapsed_time_enumerate",
                "elapsed_time_creation",
            ]
        )

        for m_size in range(1, cfg.universe.m_size + 1):

            print(
                "Generating a universe where x_size={} and m_size={}.".format(
                    cfg.universe.x_size, m_size
                )
            )

            # Ensure that primitives are added to the grammar up to `m_size`
            quantifiers_grammar_at_depth = deepcopy(quantifiers_grammar)
            quantifiers_grammar_at_depth.add_indices_as_primitives(
                cfg.m_size, cfg.universe.weight
            )
            print(quantifiers_grammar_at_depth)

            # Create the universe
            creation_start = time()
            quantifier_universe = create_universe(m_size, cfg.universe.x_size)
            creation_elapsed = time() - creation_start
            print("The size of the universe is {}".format(len(quantifier_universe)))

            # Don't consider depth=1, as depth is not deep enough to generate and expression
            for depth in range(2, cfg.grammar.depth + 1):

                enumeration_start = time()
                print("msize: ", m_size)
                print("depth: ", depth)

                expressions_by_meaning = enumerate_quantifiers(
                    depth, quantifier_universe, quantifiers_grammar_at_depth
                )

                enumerate_elapsed = time() - enumeration_start
                print(enumerate_elapsed)
                print("")

                from pathlib import Path

                outpath = (
                    Path(cfg.output)
                    / Path("M" + str(m_size))
                    / Path("X" + str(cfg.universe.x_size))
                    / Path(str("d" + str(depth)))
                    / Path("generated_expressions.yml")
                )
                Path(outpath).parent.mkdir(parents=True, exist_ok=True)

                print("Saving generated expressions...")
                save_quantifiers(expressions_by_meaning, outpath)

                writer.writerow(
                    [
                        m_size,
                        cfg.universe.x_size,
                        depth,
                        len(quantifier_universe),
                        enumerate_elapsed,
                        creation_elapsed,
                    ]
                )


def generate_inclusive_expressions(quantifiers_grammar, cfg, save=True):
    # Flush times as they are output to time log
    master_universe = None
    depth = cfg.grammar.depth

    # Don't consider m_size=0, as this is the trivial meaning
    for m_size in range(1, cfg.universe.m_size + 1):

        print(
            "Generating a universe where x_size={} and m_size={}.".format(
                cfg.universe.x_size, m_size
            )
        )

        # Ensure that primitives are added to the grammar up to `m_size`
        quantifiers_grammar_at_depth = deepcopy(quantifiers_grammar)
        quantifiers_grammar_at_depth.add_indices_as_primitives(
            m_size, cfg.universe.weight
        )
        print(quantifiers_grammar_at_depth)

        # Create the universe
        creation_start = time()
        quantifier_universe = create_universe(m_size, cfg.universe.x_size)
        creation_elapsed = time() - creation_start
        print("The size of the universe is {}".format(len(quantifier_universe)))

        if not master_universe:
            master_universe = quantifier_universe
        else:
            master_universe += quantifier_universe

        enumeration_start = time()
        print("maximum_msize: ", m_size)
        print("depth: ", depth)

        expressions_by_meaning = enumerate_quantifiers(
            depth, master_universe, quantifiers_grammar_at_depth
        )

        enumerate_elapsed = time() - enumeration_start
        print(enumerate_elapsed)
        print("")
        if save:
            save_inclusive_generation(
                expressions_by_meaning,
                master_universe,
                cfg.output,
                m_size,
                cfg.universe.x_size,
                cfg.grammar.depth,
            )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    quantifiers_grammar = QuantifierGrammar.from_yaml(cfg.grammar.path)

    if cfg.universe.inclusive_universes:
        # Generate expressions for universes up to size M
        generate_inclusive_expressions(quantifiers_grammar, cfg)
    elif "time_trial" in cfg.mode:
        # Time the generation of expressions for universes up to size M and depths up to D at fixed X
        generation_time_trial(quantifiers_grammar, cfg)
    elif "generate" in cfg.mode:
        # Generate expressions for a single universe at size M
        generate_expressions(quantifiers_grammar, cfg)


if __name__ == "__main__":
    main()
