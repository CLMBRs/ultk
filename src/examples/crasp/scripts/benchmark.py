"""Benchmark enumerating C-RASP programs over all strings up to some length.

For each program size, writes one line of JSON with the number of distinct meanings of each type found so far, the number
of candidate meanings computed at that size, the time taken, and the peak memory used.  Optionally, also saves every
program found (with how many strings of each length it accepts), for `find_named_languages`.

Enumeration stops after `--max_cost`, after a size that took longer than `--level_time_budget` seconds, or when it runs out
of memory (which can be capped with e.g. `ulimit -v`).  See `BENCHMARKS.md` for results.

From the `src/examples` directory, e.g.:
    python -m crasp.scripts.benchmark --max_length 12 --max_cost 14 --programs_output crasp/outputs/programs_12.jsonl
"""

import argparse
import json
import resource
import sys
import time
from collections.abc import Hashable, Sequence
from typing import Any, TextIO

from ultk.language.grammar.enumeration import MeaningRepresentation, SemanticEnumerator
from ultk.language.grammar.grammar import Rule
from ultk.language.semantics import Meaning

from ..grammar import build_crasp_grammar
from ..meaning import build_string_universe
from ..representations import REPRESENTATIONS
from ..values import Booleans, Counts


class CountingRepresentation(MeaningRepresentation):
    """Wraps another representation, counting how many candidate meanings it computes."""

    def __init__(self, inner: MeaningRepresentation):
        self.inner = inner
        self.count = 0

    def evaluate_terminal(self, rule: Rule) -> Any:
        self.count += 1
        return self.inner.evaluate_terminal(rule)

    def apply_rule(self, rule: Rule, children: Sequence[Any]) -> Any:
        self.count += 1
        return self.inner.apply_rule(rule, children)

    def to_meaning(self, value: Any, lhs: Any) -> Meaning:
        return self.inner.to_meaning(value, lhs)


def program_string(
    enumerator: SemanticEnumerator,
    lhs: Any,
    meaning: Hashable,
    cache: dict[tuple[Any, Hashable], str],
) -> str:
    """Get the minimal program for a meaning as a string, by following back-pointers.

    This avoids `SemanticEnumerator.get_expressions`, which also converts the meaning of every sub-program to a
    `Meaning`; over large universes, that is much slower than the enumeration itself.
    """
    if (lhs, meaning) not in cache:
        state = enumerator.get_state(meaning, lhs)
        if state.children is None or state.rule.rhs is None:
            cache[(lhs, meaning)] = state.rule.name
        else:
            arguments = ", ".join(
                program_string(enumerator, child_lhs, child, cache)
                for child_lhs, child in zip(state.rule.rhs, state.children)
            )
            cache[(lhs, meaning)] = f"{state.rule.name}({arguments})"
    return cache[(lhs, meaning)]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--max_length", type=int, default=12, help="maximum length of strings"
    )
    parser.add_argument(
        "--max_cost", type=int, default=14, help="maximum size of programs"
    )
    parser.add_argument(
        "--representation",
        choices=["meaning", *REPRESENTATIONS],
        default="packed",
        help="how meanings are represented during enumeration",
    )
    parser.add_argument(
        "--level_time_budget",
        type=float,
        default=float("inf"),
        help="stop after a size that took longer than this many seconds",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="file for per-size results (default: stdout)",
    )
    parser.add_argument(
        "--programs_output",
        type=str,
        default=None,
        help="file in which to save every program found, one JSON line each",
    )
    return parser.parse_args()


def peak_memory_gb() -> float:
    # ru_maxrss is in kilobytes on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


if __name__ == "__main__":
    args = get_args()
    output: TextIO = open(args.output, "w") if args.output else sys.stdout
    programs_output = open(args.programs_output, "w") if args.programs_output else None

    start = time.time()
    grammar = build_crasp_grammar(("a", "b"))
    universe = build_string_universe(("a", "b"), args.max_length)
    enumerator = SemanticEnumerator(
        grammar,
        universe,
        representation=(
            None
            if args.representation == "meaning"
            else REPRESENTATIONS[args.representation](universe)
        ),
    )
    counter = CountingRepresentation(enumerator.representation)
    enumerator.representation = counter
    setup = {
        "max_length": args.max_length,
        "representation": args.representation,
        "strings": len(universe),
        "setup_seconds": round(time.time() - start, 2),
    }
    print(json.dumps(setup), file=output, flush=True)

    lengths = [len(referent.name) for referent in universe.referents]
    names: dict[tuple[Any, Hashable], str] = {}
    enumeration_seconds = 0.0
    for cost in range(1, args.max_cost + 1):
        candidates_before, level_start = counter.count, time.time()
        enumerator.run(cost)
        level_seconds = time.time() - level_start
        enumeration_seconds += level_seconds
        result = {
            "size": cost,
            "languages": len(enumerator.get_states(bool)),
            "P": len(enumerator.get_states(Booleans)),
            "C": len(enumerator.get_states(Counts)),
            "candidates": counter.count - candidates_before,
            "level_seconds": round(level_seconds, 2),
            "total_seconds": round(enumeration_seconds, 2),
            "peak_memory_gb": round(peak_memory_gb(), 2),
        }
        print(json.dumps(result), file=output, flush=True)

        # saving programs is not timed
        if programs_output is not None:
            for meaning, state in enumerator.get_states(bool).items():
                if state.cost == cost:
                    accepted = counter.to_meaning(meaning, bool).mapping
                    per_length = [0] * args.max_length
                    for length, is_accepted in zip(lengths, accepted):
                        per_length[length - 1] += is_accepted
                    entry = {
                        "size": cost,
                        "program": program_string(enumerator, bool, meaning, names),
                        "accepted_per_length": per_length,
                    }
                    programs_output.write(json.dumps(entry) + "\n")
            programs_output.flush()

        if enumerator.is_saturated:
            break
        if level_seconds > args.level_time_budget:
            print(
                json.dumps(
                    {"stopped": f"size {cost} took more than {args.level_time_budget}s"}
                ),
                file=output,
                flush=True,
            )
            break
