"""Enumerate C-RASP programs in order of size, and report the minimal programs for some named languages.

From the `src/examples` directory, e.g.:
    python -m crasp.scripts.enumerate_programs --max_length 6 --max_cost 14
"""

import argparse
import time

from ultk.language.grammar.enumeration import SemanticEnumerator

from ..grammar import build_crasp_grammar
from ..languages import NAMED_LANGUAGES
from ..meaning import build_string_universe
from ..representations import REPRESENTATIONS


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max_length", type=int, default=6, help="maximum length of strings"
    )
    parser.add_argument(
        "--max_cost", type=int, default=13, help="maximum size of programs"
    )
    parser.add_argument(
        "--representation",
        choices=["meaning", *REPRESENTATIONS],
        default="packed",
        help="how meanings are represented during enumeration: as `Meaning`s (slowest), "
        "or with `BasicCRASPRepresentation` or `PackedCRASPRepresentation`",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    grammar = build_crasp_grammar(("a", "b"))
    universe = build_string_universe(("a", "b"), args.max_length)
    representation = (
        None
        if args.representation == "meaning"
        else REPRESENTATIONS[args.representation](universe)
    )
    enumerator = SemanticEnumerator(grammar, universe, representation=representation)

    start = time.time()
    for cost in range(1, args.max_cost + 1):
        enumerator.run(cost)
        counts = {str(lhs): len(enumerator.get_states(lhs)) for lhs in (bool, "P", "C")}
        print(
            f"size {cost:3d}: {counts}, total time {time.time() - start:.1f}s",
            flush=True,
        )
        if enumerator.is_saturated:
            print("Saturated: no larger program expresses a new meaning.")
            break

    expressions = enumerator.get_expressions()
    print(
        f"\n{len(expressions)} distinct languages (restricted to strings of length <= {args.max_length}) "
        f"expressible by programs of size <= {args.max_cost}\n"
    )
    meanings_by_mapping = {meaning.mapping: meaning for meaning in expressions}
    for name, in_language in NAMED_LANGUAGES.items():
        mapping = tuple(in_language(referent.name) for referent in universe.referents)
        if mapping in meanings_by_mapping:
            expression = expressions[meanings_by_mapping[mapping]]
            print(f"{name}: size {len(expression)}: {expression}")
        else:
            print(f"{name}: no program of size <= {args.max_cost}")
