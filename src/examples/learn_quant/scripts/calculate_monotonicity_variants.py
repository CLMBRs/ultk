"""Calculate explicit monotonicity-measure variants for the 2,000 expressions.

This script evaluates the four one-sided entropy primitives (up/down closure
and up/down interior), derives the manuscript, complement-dual, and two-sided
variants, and adds a direct comparable-pair violation control.

Run with the archived ``altk`` environment because the expression pool pickle
contains classes from that checkout:

    /path/to/altk/bin/python scripts/calculate_monotonicity_variants.py
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


DIRS = ["right_upward", "left_upward", "right_downward", "left_downward"]
DEFAULT_ALTK_ARCHIVE = Path.home() / "Documents/UWLing/altk/src/examples"
POOL_REL = Path("learn_quant/outputs/M4/X4/d5")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_archive(archive: Path):
    sys.path.insert(0, str(archive))
    os.chdir(archive)

    import dill as pkl
    from ultk.util.frozendict import FrozenDict

    FrozenDict.__setitem__ = dict.__setitem__
    base = archive / POOL_REL
    with open(base / "master_universe.pkl", "rb") as handle:
        universe = pkl.load(handle)
    with open(base / "generated_expressions_xidx.pkl", "rb") as handle:
        pool = pkl.load(handle)
    return universe, {
        expression.term_expression: expression for expression in pool.values()
    }


def load_module(name: str, source: Path):
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_scores(universe, by_term: dict, terms: list[str]) -> pd.DataFrame:
    measures = load_module("learn_quant._variant_measures", repo_root() / "measures.py")
    variants = load_module(
        "learn_quant._monotonicity_variants",
        repo_root() / "monotonicity_variants.py",
    )
    all_models = universe.binarize_referents(mode="set_vectors_w_padding")
    reference_a = universe.binarize_referents(mode="A")
    reference_b = universe.binarize_referents(mode="B")
    references = (reference_a, reference_b)
    relations = [
        variants.comparable_pair_relation(all_models, reference)
        & ~np.eye(len(all_models), dtype=bool)
        for reference in references
    ]
    cfg = SimpleNamespace(
        measures=SimpleNamespace(monotonicity=SimpleNamespace(debug=False))
    )

    rows = []
    started = time.time()
    for index, term in enumerate(terms):
        expression = by_term[term]
        q = np.fromiter(
            (expression.meaning.mapping[ref] for ref in universe.referents),
            dtype=int,
            count=len(universe.referents),
        )
        not_q = 1 - q
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            up_closure = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, q, cfg, False
                    )
                    for reference in references
                ]
            )
            down_closure = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, q, cfg, True
                    )
                    for reference in references
                ]
            )
            up_interior = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, not_q, cfg, True
                    )
                    for reference in references
                ]
            )
            down_interior = np.array(
                [
                    measures.upward_monotonicity_entropy(
                        all_models, reference, not_q, cfg, False
                    )
                    for reference in references
                ]
            )

        direct_up = []
        direct_down = []
        q_bool = q.astype(bool)
        for relation in relations:
            direct_up.append(
                1
                - (relation & q_bool[:, None] & ~q_bool[None, :]).sum() / relation.sum()
            )
            direct_down.append(
                1
                - (relation & ~q_bool[:, None] & q_bool[None, :]).sum() / relation.sum()
            )

        primitive = {
            "up_closure": up_closure,
            "up_interior": up_interior,
            "down_closure": down_closure,
            "down_interior": down_interior,
        }
        metric_values = {
            "majorant": np.concatenate([up_closure, down_closure]),
            "complement_dual": np.concatenate([up_closure, down_interior]),
            "two_sided_mean": np.concatenate(
                [
                    np.mean([up_closure, up_interior], axis=0),
                    np.mean([down_closure, down_interior], axis=0),
                ]
            ),
            "two_sided_min": np.concatenate(
                [
                    np.minimum(up_closure, up_interior),
                    np.minimum(down_closure, down_interior),
                ]
            ),
            "two_sided_max": np.concatenate(
                [
                    np.maximum(up_closure, up_interior),
                    np.maximum(down_closure, down_interior),
                ]
            ),
            "violation_rate": np.array([*direct_up, *direct_down]),
        }
        row = {"expression": term}
        for primitive_name, values in primitive.items():
            for argument, value in zip(("right", "left"), values):
                row[f"{primitive_name}_{argument}"] = value
        for metric_name, values in metric_values.items():
            for direction, value in zip(DIRS, values):
                row[f"{metric_name}_{direction}"] = value
            row[f"{metric_name}_degree"] = np.clip(values, 0, 1).max()
        rows.append(row)

        if (index + 1) % 250 == 0 or index + 1 == len(terms):
            print(
                f"  {index + 1:4d}/{len(terms)} expressions "
                f"({time.time() - started:.1f}s elapsed)"
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--altk-archive", type=Path, default=DEFAULT_ALTK_ARCHIVE)
    parser.add_argument(
        "--sample",
        type=Path,
        default=repo_root() / "expressions_sample_2k.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "analysis/monotonicity_measure_variants_2k.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sample = pd.read_csv(args.sample.resolve())
    terms = sample["term_expression"].tolist()
    if args.limit is not None:
        terms = terms[: args.limit]
    universe, by_term = load_archive(args.altk_archive.resolve())
    missing = [term for term in terms if term not in by_term]
    if missing:
        raise ValueError(f"{len(missing)} sampled expressions are absent from the pool")

    print(f"calculating six variants for {len(terms)} expressions")
    scores = calculate_scores(universe, by_term, terms)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({scores.shape[0]} rows, {scores.shape[1]} columns)")


if __name__ == "__main__":
    main()
