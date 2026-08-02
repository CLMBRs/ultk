"""Benchmark monotonicity measures on familiar set-theoretic quantifiers.

The benchmark meanings and their expected exact directions are specified before
scores are calculated. Exhaustive comparable-pair checks verify those semantic
expectations independently of the entropy measure.

Run with the archived ``altk`` environment because the M6/X6 universe pickle
uses its original class layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_ALTK_ARCHIVE = Path.home() / "Documents/UWLing/altk/src/examples"
DIRECTIONS = ("RU", "LU", "RD", "LD")
Predicate = Callable[[frozenset, frozenset], bool]


@dataclass(frozen=True)
class BenchmarkMeaning:
    category: str
    name: str
    formula: str
    predicate: Predicate
    expected_directions: tuple[str, ...]


def benchmark_meanings() -> list[BenchmarkMeaning]:
    """Return a theory-first suite of familiar quantifier meanings."""

    return [
        BenchmarkMeaning(
            "Aristotelian",
            "all A are B",
            "A <= B",
            lambda a, b: a <= b,
            ("RU", "LD"),
        ),
        BenchmarkMeaning(
            "Aristotelian",
            "not all A are B",
            "A not<= B",
            lambda a, b: not a <= b,
            ("LU", "RD"),
        ),
        BenchmarkMeaning(
            "Aristotelian",
            "no A are B",
            "A intersect B = empty",
            lambda a, b: not (a & b),
            ("RD", "LD"),
        ),
        BenchmarkMeaning(
            "Aristotelian",
            "some A are B",
            "A intersect B != empty",
            lambda a, b: bool(a & b),
            ("RU", "LU"),
        ),
        BenchmarkMeaning(
            "Aristotelian",
            "all B are A",
            "B <= A",
            lambda a, b: b <= a,
            ("LU", "RD"),
        ),
        BenchmarkMeaning(
            "Aristotelian",
            "some A are not B",
            "A - B != empty",
            lambda a, b: bool(a - b),
            ("LU", "RD"),
        ),
        BenchmarkMeaning(
            "Set relation",
            "A proper-subset B",
            "A < B",
            lambda a, b: a < b,
            ("RU", "LD"),
        ),
        BenchmarkMeaning(
            "Set relation",
            "A proper-superset B",
            "A > B",
            lambda a, b: a > b,
            ("LU", "RD"),
        ),
        BenchmarkMeaning(
            "Set relation",
            "A equals B",
            "A = B",
            lambda a, b: a == b,
            (),
        ),
        BenchmarkMeaning(
            "Set relation",
            "A differs from B",
            "A != B",
            lambda a, b: a != b,
            (),
        ),
        BenchmarkMeaning(
            "Set relation",
            "A and B are comparable",
            "A <= B or B <= A",
            lambda a, b: a <= b or b <= a,
            (),
        ),
        BenchmarkMeaning(
            "Set relation",
            "A and B are incomparable",
            "not (A <= B or B <= A)",
            lambda a, b: not (a <= b or b <= a),
            (),
        ),
        BenchmarkMeaning(
            "Cardinality",
            "more A than B",
            "|A| > |B|",
            lambda a, b: len(a) > len(b),
            ("LU", "RD"),
        ),
        BenchmarkMeaning(
            "Cardinality",
            "fewer A than B",
            "|A| < |B|",
            lambda a, b: len(a) < len(b),
            ("RU", "LD"),
        ),
        BenchmarkMeaning(
            "Cardinality",
            "at least as many A as B",
            "|A| >= |B|",
            lambda a, b: len(a) >= len(b),
            ("LU", "RD"),
        ),
        BenchmarkMeaning(
            "Cardinality",
            "at most as many A as B",
            "|A| <= |B|",
            lambda a, b: len(a) <= len(b),
            ("RU", "LD"),
        ),
        BenchmarkMeaning(
            "Cardinality",
            "same number of A and B",
            "|A| = |B|",
            lambda a, b: len(a) == len(b),
            (),
        ),
        BenchmarkMeaning(
            "Cardinality",
            "different numbers of A and B",
            "|A| != |B|",
            lambda a, b: len(a) != len(b),
            (),
        ),
        BenchmarkMeaning(
            "Overlap count",
            "at least two overlap",
            "|A intersect B| >= 2",
            lambda a, b: len(a & b) >= 2,
            ("RU", "LU"),
        ),
        BenchmarkMeaning(
            "Overlap count",
            "at most one overlaps",
            "|A intersect B| <= 1",
            lambda a, b: len(a & b) <= 1,
            ("RD", "LD"),
        ),
        BenchmarkMeaning(
            "Overlap count",
            "exactly one overlaps",
            "|A intersect B| = 1",
            lambda a, b: len(a & b) == 1,
            (),
        ),
        BenchmarkMeaning(
            "Overlap count",
            "exactly three overlap",
            "|A intersect B| = 3",
            lambda a, b: len(a & b) == 3,
            (),
        ),
        BenchmarkMeaning(
            "Overlap count",
            "exactly five overlap",
            "|A intersect B| = 5",
            lambda a, b: len(a & b) == 5,
            (),
        ),
        BenchmarkMeaning(
            "Proportional",
            "most A are B",
            "|A intersect B| > |A - B|",
            lambda a, b: len(a & b) > len(a - b),
            ("RU",),
        ),
        BenchmarkMeaning(
            "Proportional",
            "at least half of A are B",
            "|A intersect B| >= |A - B|",
            lambda a, b: len(a & b) >= len(a - b),
            ("RU",),
        ),
        BenchmarkMeaning(
            "Proportional",
            "less than half of A are B",
            "|A intersect B| < |A - B|",
            lambda a, b: len(a & b) < len(a - b),
            ("RD",),
        ),
        BenchmarkMeaning(
            "Proportional",
            "at most half of A are B",
            "|A intersect B| <= |A - B|",
            lambda a, b: len(a & b) <= len(a - b),
            ("RD",),
        ),
        BenchmarkMeaning(
            "Boundary",
            "both A and B are nonempty",
            "A != empty and B != empty",
            lambda a, b: bool(a) and bool(b),
            ("RU", "LU"),
        ),
        BenchmarkMeaning(
            "Boundary",
            "A or B is empty",
            "A = empty or B = empty",
            lambda a, b: not a or not b,
            ("RD", "LD"),
        ),
        BenchmarkMeaning(
            "Boundary",
            "exactly one of A and B is nonempty",
            "(A != empty) xor (B != empty)",
            lambda a, b: bool(a) ^ bool(b),
            (),
        ),
        BenchmarkMeaning(
            "Boundary",
            "A and B are both empty or both nonempty",
            "(A != empty) iff (B != empty)",
            lambda a, b: bool(a) == bool(b),
            (),
        ),
        BenchmarkMeaning(
            "Boolean combination",
            "all or no A are B",
            "A <= B or A intersect B = empty",
            lambda a, b: a <= b or not (a & b),
            ("LD",),
        ),
        BenchmarkMeaning(
            "Logical endpoint",
            "tautology",
            "true",
            lambda _a, _b: True,
            DIRECTIONS,
        ),
        BenchmarkMeaning(
            "Logical endpoint",
            "contradiction",
            "false",
            lambda _a, _b: False,
            DIRECTIONS,
        ),
    ]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_archive(archive: Path):
    sys.path.insert(0, str(archive))
    os.chdir(archive)
    import dill as pkl
    from ultk.util.frozendict import FrozenDict

    FrozenDict.__setitem__ = dict.__setitem__
    with (archive / "learn_quant/outputs/M6/X6/d3/master_universe.pkl").open(
        "rb"
    ) as handle:
        return pkl.load(handle)


def exact_directions(
    quantifier: np.ndarray, right_relation: np.ndarray, left_relation: np.ndarray
) -> dict[str, bool]:
    q = np.asarray(quantifier, dtype=bool)
    relations = (right_relation, left_relation)
    result = {}
    for direction, relation in zip(("RU", "LU"), relations):
        result[direction] = not np.any(relation & q[:, None] & ~q[None, :])
    for direction, relation in zip(("RD", "LD"), relations):
        result[direction] = not np.any(relation & ~q[:, None] & q[None, :])
    return result


def preservation_scores(
    quantifier: np.ndarray, right_relation: np.ndarray, left_relation: np.ndarray
) -> dict[str, float]:
    """Return conditional truth-preservation rates over proper comparable pairs."""

    q = np.asarray(quantifier, dtype=bool)
    proper_mask = ~np.eye(len(q), dtype=bool)
    relations = (right_relation & proper_mask, left_relation & proper_mask)
    result = {}
    for direction, relation in zip(("RU", "LU"), relations):
        eligible = relation & q[:, None]
        violations = eligible & ~q[None, :]
        result[direction] = (
            1.0 if not eligible.any() else 1 - violations.sum() / eligible.sum()
        )
    for direction, relation in zip(("RD", "LD"), relations):
        eligible = relation & q[None, :]
        violations = eligible & ~q[:, None]
        result[direction] = (
            1.0 if not eligible.any() else 1 - violations.sum() / eligible.sum()
        )
    return result


def calculate_benchmark(archive: Path, repo_root: Path) -> pd.DataFrame:
    universe = load_archive(archive)
    measures = load_module("benchmark_measures", repo_root / "measures.py")
    variants = load_module(
        "benchmark_monotonicity_variants", repo_root / "monotonicity_variants.py"
    )
    all_models = universe.binarize_referents(mode="set_vectors_w_padding")
    reference_a = universe.binarize_referents(mode="A")
    reference_b = universe.binarize_referents(mode="B")
    right_relation = variants.comparable_pair_relation(all_models, reference_a)
    left_relation = variants.comparable_pair_relation(all_models, reference_b)
    cfg = SimpleNamespace(
        measures=SimpleNamespace(monotonicity=SimpleNamespace(debug=False))
    )

    rows = []
    for meaning in benchmark_meanings():
        quantifier = np.fromiter(
            (
                meaning.predicate(referent.A, referent.B)
                for referent in universe.referents
            ),
            dtype=int,
            count=len(universe.referents),
        )
        verified = exact_directions(quantifier, right_relation, left_relation)
        expected = {
            direction: direction in meaning.expected_directions
            for direction in DIRECTIONS
        }
        if expected != verified:
            raise AssertionError(
                f"Semantic expectation failed for {meaning.name}: "
                f"expected={expected}, exhaustive={verified}"
            )
        retention = preservation_scores(quantifier, right_relation, left_relation)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            references = (reference_a, reference_b)
            not_quantifier = 1 - quantifier
            up_closure = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, quantifier, cfg, False
                    )
                    for reference in references
                ]
            )
            down_closure = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, quantifier, cfg, True
                    )
                    for reference in references
                ]
            )
            up_interior = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, not_quantifier, cfg, True
                    )
                    for reference in references
                ]
            )
            down_interior = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, not_quantifier, cfg, False
                    )
                    for reference in references
                ]
            )
            majorant = np.concatenate([up_closure, down_closure])
            minimum = np.concatenate(
                [
                    np.minimum(up_closure, up_interior),
                    np.minimum(down_closure, down_interior),
                ]
            )
            mean = np.concatenate(
                [
                    np.mean([up_closure, up_interior], axis=0),
                    np.mean([down_closure, down_interior], axis=0),
                ]
            )

        row = {
            "category": meaning.category,
            "meaning": meaning.name,
            "formula": meaning.formula,
            "expected_exact_directions": ",".join(meaning.expected_directions)
            or "none",
            "truth_prevalence": quantifier.mean(),
        }
        for index, direction in enumerate(DIRECTIONS):
            row[f"expected_{direction}"] = expected[direction]
            row[f"min_{direction}"] = minimum[index]
            row[f"majorant_{direction}"] = majorant[index]
            row[f"mean_{direction}"] = mean[index]
            row[f"preservation_{direction}"] = retention[direction]
        nonexact_scores = [
            minimum[index]
            for index, direction in enumerate(DIRECTIONS)
            if not expected[direction]
        ]
        row["min_degree"] = np.max(minimum)
        row["max_nonexact_min_score"] = (
            np.max(nonexact_scores) if nonexact_scores else 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--altk-archive", type=Path, default=DEFAULT_ALTK_ARCHIVE)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "analysis/semantic_monotonicity_benchmark.csv",
    )
    args = parser.parse_args()
    result = calculate_benchmark(args.altk_archive.resolve(), repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(result)} meanings)")
    print(
        "Exact directions scored 1:",
        all(
            np.isclose(row[f"min_{direction}"], 1.0)
            for _, row in result.iterrows()
            for direction in DIRECTIONS
            if row[f"expected_{direction}"]
        ),
    )
    print(
        "Largest non-exact two-sided-min score:",
        f"{result['max_nonexact_min_score'].max():.3f}",
    )


if __name__ == "__main__":
    main()
