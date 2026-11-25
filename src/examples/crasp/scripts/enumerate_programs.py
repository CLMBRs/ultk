"""Enumerate C-RASP programs in order of size, and report the minimal programs for some example languages.

From the `src/examples` directory, e.g.:
    python -m crasp.scripts.enumerate_programs --max_length 6 --max_cost 14
"""

import argparse
import re
import time

from ultk.language.grammar.enumeration import SemanticEnumerator

from ..grammar import build_crasp_grammar
from ..meaning import build_string_universe
from ..representation import CRASPRepresentation

# example languages from Yang & Chiang (2024), over the alphabet {a, b}, as regular expressions or predicates
EXAMPLE_LANGUAGES = {
    "a*b*": lambda string: re.fullmatch("a*b*", string) is not None,
    "a*b*a*": lambda string: re.fullmatch("a*b*a*", string) is not None,
    "Dyck-1 (a = open, b = close)": lambda string: all(
        string[:end].count("a") >= string[:end].count("b")
        for end in range(1, len(string) + 1)
    )
    and string.count("a") == string.count("b"),
    "a^n b^n": lambda string: re.fullmatch("a*b*", string) is not None
    and string.count("a") == string.count("b"),
    "more a's than b's": lambda string: string.count("a") > string.count("b"),
}


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max_length", type=int, default=6, help="maximum length of strings"
    )
    parser.add_argument(
        "--max_cost", type=int, default=13, help="maximum size of programs"
    )
    parser.add_argument(
        "--default_representation",
        action="store_true",
        help="represent meanings as `Meaning`s (slower), instead of with `CRASPRepresentation`",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    grammar = build_crasp_grammar(("a", "b"))
    universe = build_string_universe(("a", "b"), args.max_length)
    representation = (
        None if args.default_representation else CRASPRepresentation(universe)
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
    for name, in_language in EXAMPLE_LANGUAGES.items():
        mapping = tuple(in_language(referent.name) for referent in universe.referents)
        if mapping in meanings_by_mapping:
            expression = expressions[meanings_by_mapping[mapping]]
            print(f"{name}: size {len(expression)}: {expression}")
        else:
            print(f"{name}: no program of size <= {args.max_cost}")
