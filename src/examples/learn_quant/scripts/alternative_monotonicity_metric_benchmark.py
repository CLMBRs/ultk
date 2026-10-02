"""Benchmark alternative monotonicity metrics on a transparent M4/X4 universe."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import alternative_monotonicity_metrics as metrics


@dataclass(frozen=True)
class Meaning:
    name: str
    family: str
    formula: str
    predicate: metrics.Predicate


def meanings(n: int) -> list[Meaning]:
    full = frozenset(range(n))
    return [
        Meaning("all A are B", "textbook", "A <= B", lambda a, b: a <= b),
        Meaning(
            "some A are B",
            "textbook",
            "|A & B| >= 1",
            lambda a, b: bool(a & b),
        ),
        Meaning(
            "no A are B",
            "textbook",
            "|A & B| = 0",
            lambda a, b: not (a & b),
        ),
        Meaning(
            "not all A are B",
            "textbook",
            "A not<= B",
            lambda a, b: not a <= b,
        ),
        Meaning(
            "most A are B",
            "proportional",
            "|A & B| > |A - B|",
            lambda a, b: len(a & b) > len(a - b),
        ),
        Meaning(
            "at least two overlap",
            "overlap threshold",
            "|A & B| >= 2",
            lambda a, b: len(a & b) >= 2,
        ),
        Meaning(
            "at most one overlaps",
            "overlap threshold",
            "|A & B| <= 1",
            lambda a, b: len(a & b) <= 1,
        ),
        Meaning(
            "exactly two overlap",
            "overlap band",
            "|A & B| = 2",
            lambda a, b: len(a & b) == 2,
        ),
        Meaning(
            "between one and three overlap",
            "overlap band",
            "1 <= |A & B| <= 3",
            lambda a, b: 1 <= len(a & b) <= 3,
        ),
        Meaning(
            "even overlap",
            "alternating",
            "|A & B| mod 2 = 0",
            lambda a, b: len(a & b) % 2 == 0,
        ),
        Meaning(
            "B has at least two elements",
            "cardinality threshold",
            "|B| >= 2",
            lambda _a, b: len(b) >= 2,
        ),
        Meaning(
            "B threshold with one exception",
            "near-monotone",
            "|B| >= 2 except (A=empty, B=M)",
            lambda a, b: len(b) >= 2 and not (not a and b == full),
        ),
        Meaning(
            "B has one to three elements",
            "cardinality band",
            "1 <= |B| <= 3",
            lambda _a, b: 1 <= len(b) <= 3,
        ),
        Meaning(
            "B has even cardinality",
            "alternating",
            "|B| mod 2 = 0",
            lambda _a, b: len(b) % 2 == 0,
        ),
        Meaning("A equals B", "set relation", "A = B", lambda a, b: a == b),
        Meaning("A differs from B", "set relation", "A != B", lambda a, b: a != b),
        Meaning(
            "A and B are comparable",
            "set relation",
            "A <= B or B <= A",
            lambda a, b: a <= b or b <= a,
        ),
        Meaning(
            "A and B are incomparable",
            "set relation",
            "not (A <= B or B <= A)",
            lambda a, b: not (a <= b or b <= a),
        ),
        Meaning(
            "both empty or both nonempty",
            "boundary",
            "(A != empty) iff (B != empty)",
            lambda a, b: bool(a) == bool(b),
        ),
        Meaning("tautology", "endpoint", "true", lambda _a, _b: True),
        Meaning("contradiction", "endpoint", "false", lambda _a, _b: False),
    ]


def calculate(n: int = 4) -> pd.DataFrame:
    universe = metrics.FiniteSetUniverse.create(n)
    structures = {}
    for direction in metrics.DIRECTIONS:
        structures[direction] = {
            "relation": universe.relation(direction),
            "strict": universe.strict_relation(direction),
            "edge": universe.relation(direction, immediate=True),
            "distance": universe.movement_distance(direction),
            "chains": universe.maximal_chains(direction),
            "contexts": universe.context_ids(direction),
        }

    rows = []
    for meaning in meanings(n):
        q = universe.truth_values(meaning.predicate)
        for direction in metrics.DIRECTIONS:
            structure = structures[direction]
            relation = structure["relation"]
            strict = structure["strict"]
            edge = structure["edge"]
            edit_count, edit_score = metrics.nearest_monotone_edit(q, edge)
            threshold_accuracy, threshold_description = (
                metrics.best_simple_threshold_accuracy(q, universe, direction)
            )
            context_mean, context_sd, context_coverage = (
                metrics.context_preservation_profile(q, strict, structure["contexts"])
            )
            robustness_curve = metrics.distance_robustness_curve(
                q, strict, structure["distance"], n
            )
            defined_robustness = [
                score for score in robustness_curve.values() if not np.isnan(score)
            ]
            rows.append(
                {
                    "meaning": meaning.name,
                    "family": meaning.family,
                    "formula": meaning.formula,
                    "direction": direction,
                    "truth_prevalence": q.mean(),
                    "exact": metrics.exact_monotonicity(q, strict),
                    "majorant_entropy": metrics.majorant_entropy_score(q, relation),
                    "two_sided_min": metrics.two_sided_min_score(q, relation),
                    "pairwise_preservation": (
                        metrics.pairwise_preservation_score(q, strict)
                    ),
                    "edge_preservation": metrics.pairwise_preservation_score(q, edge),
                    "unconditional_pair": metrics.unconditional_pair_score(q, strict),
                    "nearest_monotone_edit": edit_score,
                    "best_cardinality_threshold_repair": (
                        metrics.best_cardinality_threshold_repair_score(
                            q, universe, direction
                        )
                    ),
                    "edit_count": edit_count,
                    "closure_inflation": metrics.closure_inflation_score(q, relation),
                    "closure_precision": metrics.closure_precision_score(q, relation),
                    "chain_switch": metrics.switch_simplicity_score(
                        q, structure["chains"]
                    ),
                    "chain_inversion": metrics.chain_inversion_score(
                        q, structure["chains"]
                    ),
                    "equal_distance_robustness": (
                        float(np.mean(defined_robustness))
                        if defined_robustness
                        else 1.0
                    ),
                    **{
                        f"robustness_k{distance}": score
                        for distance, score in robustness_curve.items()
                    },
                    "derivative_sign": metrics.derivative_sign_score(q, edge),
                    "context_mean": context_mean,
                    "context_sd": context_sd,
                    "context_coverage": context_coverage,
                    "best_simple_threshold": threshold_accuracy,
                    "best_threshold_description": threshold_description,
                    "exception_code_proxy": metrics.exception_code_score(
                        edit_count, universe.size
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    result = calculate()
    output = ROOT / "analysis/alternative_monotonicity_metric_benchmark.csv"
    result.to_csv(output, index=False)
    print(f"Wrote {output} ({len(result)} directional rows)")


if __name__ == "__main__":
    main()
