"""Find the minimal programs for the languages in `languages.py`, among programs saved by `benchmark`.

The programs were enumerated over all strings up to some length L (inferred from the saved file).  For each named
language, this finds the first (so, smallest) saved program that defines it on those strings.  Optionally:

- `--verify_length V` checks each program found on all strings up to length V > L.  A program that is correct up to
  length V is a minimal program for the language on strings up to length V too: any program correct up to length V is
  also correct up to length L, so none can be smaller.
- `--merge_length M` reports pairs of programs that define the same language on strings up to length M < L, but not on
  strings up to length L, together with the shortest string that tells them apart.

From the `src/examples` directory, e.g.:
    python -m crasp.scripts.find_named_languages --programs crasp/outputs/programs_12.jsonl --verify_length 20
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from ultk.language.grammar.grammar import Grammar, GrammaticalExpression
from ultk.language.semantics import Universe

from ..grammar import build_crasp_grammar
from ..languages import NAMED_LANGUAGES
from ..meaning import build_string_universe
from ..representations import PackedCRASPRepresentation


class ProgramEvaluator:
    """Evaluates whole programs over a universe of strings with `PackedCRASPRepresentation`, which is much faster than
    `GrammaticalExpression.evaluate` over large universes."""

    def __init__(self, grammar: Grammar, universe: Universe):
        self.grammar = grammar
        self.universe = universe
        self.representation = PackedCRASPRepresentation(universe)
        self.lengths = np.array([len(referent.name) for referent in universe.referents])

    def accepts(self, program: str) -> np.ndarray:
        """A boolean array saying which strings of the universe the program accepts."""

        def value(node: GrammaticalExpression):
            rule = self.grammar._rules_by_name[node.rule_name]
            if node.children is None:
                return self.representation.evaluate_terminal(rule)
            return self.representation.apply_rule(
                rule, [value(child) for child in node.children]
            )

        meaning = self.representation.to_meaning(
            value(self.grammar.parse(program)), bool
        )
        return np.array(meaning.mapping)

    def in_language(self, predicate) -> np.ndarray:
        return np.array(
            [predicate(referent.name) for referent in self.universe.referents]
        )

    def accepted_per_length(self, accepted: np.ndarray) -> tuple[int, ...]:
        return tuple(
            np.bincount(self.lengths[accepted], minlength=self.lengths.max() + 1)[
                1:
            ].tolist()
        )


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--programs",
        required=True,
        help="programs saved by `benchmark --programs_output`",
    )
    parser.add_argument(
        "--verify_length",
        type=int,
        default=None,
        help="check the programs found on all strings up to this length",
    )
    parser.add_argument(
        "--merge_length",
        type=int,
        default=None,
        help="report programs that only differ on strings longer than this",
    )
    parser.add_argument("--output", default=None, help="file for the results, as JSON")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    with open(args.programs) as f:
        programs = [json.loads(line) for line in f]
    max_length = len(programs[0]["accepted_per_length"])
    max_size = programs[-1]["size"]
    print(
        f"{len(programs)} programs of size {programs[0]['size']} to {max_size}, "
        f"enumerated over strings up to length {max_length}\n"
    )
    grammar = build_crasp_grammar(("a", "b"))
    evaluator = ProgramEvaluator(grammar, build_string_universe(("a", "b"), max_length))
    verifier = (
        ProgramEvaluator(grammar, build_string_universe(("a", "b"), args.verify_length))
        if args.verify_length
        else None
    )

    # the language a program defines has the same number of strings of each length as the program accepts,
    # so only programs with the right counts need to be evaluated
    by_signature = defaultdict(list)
    for entry in programs:
        by_signature[tuple(entry["accepted_per_length"])].append(entry)

    results: dict[str, dict] = {}
    for name, predicate in NAMED_LANGUAGES.items():
        target = evaluator.in_language(predicate)
        found = next(
            (
                entry
                for entry in by_signature.get(evaluator.accepted_per_length(target), [])
                if np.array_equal(evaluator.accepts(entry["program"]), target)
            ),
            None,
        )
        result: dict = {"size": None, "program": None}
        if found is not None:
            result = {"size": found["size"], "program": found["program"]}
            if verifier is not None:
                result["correct_up_to_verify_length"] = bool(
                    np.array_equal(
                        verifier.accepts(found["program"]),
                        verifier.in_language(predicate),
                    )
                )
        results[name] = result
        size = result["size"] if found else "-"
        verified = ""
        if "correct_up_to_verify_length" in result:
            verified = (
                f" [correct up to length {args.verify_length}]"
                if result["correct_up_to_verify_length"]
                else f" [WRONG on some string up to length {args.verify_length}]"
            )
        program = result["program"] or f"no program of size <= {max_size}"
        print(f"{name:46s} {size:>3}  {program}{verified}", flush=True)

    merges: list[dict] = []
    if args.merge_length:
        short = ProgramEvaluator(
            grammar, build_string_universe(("a", "b"), args.merge_length)
        )
        groups = defaultdict(list)
        for entry in programs:
            groups[short.accepts(entry["program"]).tobytes()].append(entry)
        merged = [group for group in groups.values() if len(group) > 1]
        print(
            f"\n{len(programs)} languages on strings up to length {max_length} are only {len(groups)} "
            f"on strings up to length {args.merge_length}; for example:"
        )
        names = [referent.name for referent in evaluator.universe.referents]
        for group in sorted(
            merged, key=lambda group: (group[1]["size"], group[0]["size"])
        )[:5]:
            first, second = group[0], group[1]
            differ = np.flatnonzero(
                evaluator.accepts(first["program"])
                != evaluator.accepts(second["program"])
            )
            # strings are ordered by length, so the first difference is on a shortest string
            witness = names[differ[0]]
            merges.append(
                {
                    "programs": [first["program"], second["program"]],
                    "shortest_difference": witness,
                }
            )
            print(
                f"  size {first['size']}: {first['program']}\n"
                f"  size {second['size']}: {second['program']}\n"
                f"  first differ on {witness!r}\n"
            )

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"named_languages": results, "merges": merges}, f, indent=1)
