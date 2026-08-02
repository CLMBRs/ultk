"""Calculate complement-dual monotonicity scores for the 2k sample.

The manuscript-era run table was generated with downward scores computed via
``flip=True`` (a true-successor feature). This alternative instead uses
the upward, predecessor-based computation on the complemented truth vector:

    down(Q) = up(not Q)

This enforces complement-pair mirror symmetry, but it also changes the
downward operator from a least monotone majorant to a greatest monotone
minorant. It is therefore an alternative metric, not a neutral bug fix. The
historical ``corrected`` filenames are retained so existing analyses continue
to run. This script preserves manuscript-era values in ``*_original`` columns.

Run with the archived ``altk`` environment because the expression-pool pickle
contains classes from that checkout:

    /path/to/altk/bin/python scripts/recalculate_corrected_monotonicity.py
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


DIRS = ["right_upward", "left_upward", "right_downward", "left_downward"]
DEFAULT_ALTK_ARCHIVE = Path.home() / "Documents/UWLing/altk/src/examples"
POOL_REL = Path("learn_quant/outputs/M4/X4/d5")
METRIC_VERSION = "complement-symmetric-entropy-v1"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_archive(archive: Path):
    """Load archived objects before importing the alternative metric source.

    The pickles require the archived ``learn_quant``/``ultk`` class layout. Once
    those modules are loaded, the current repository's ``measures.py`` can be
    executed under a private module name while reusing the archived data types.
    """

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


def load_alternative_measures():
    source = repo_root() / "measures.py"
    spec = importlib.util.spec_from_file_location(
        "learn_quant._complement_symmetric_measures", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load alternative metric source from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_scores(
    universe,
    by_term: dict,
    sample_terms: list[str],
    measures,
) -> pd.DataFrame:
    missing = [term for term in sample_terms if term not in by_term]
    if missing:
        raise ValueError(f"{len(missing)} sampled expressions are absent from the pool")

    all_models = universe.binarize_referents(mode="set_vectors_w_padding")
    reference_a = universe.binarize_referents(mode="A")
    reference_b = universe.binarize_referents(mode="B")
    refs = universe.referents
    cfg = SimpleNamespace(
        measures=SimpleNamespace(monotonicity=SimpleNamespace(debug=False))
    )

    scores = np.empty((len(sample_terms), 4), dtype=float)
    started = time.time()
    for index, term in enumerate(sample_terms):
        expression = by_term[term]
        quantifier = np.fromiter(
            (expression.meaning.mapping[ref] for ref in refs),
            dtype=int,
            count=len(refs),
        )
        # measure_monotonicity prints each score vector; suppress 2,000 routine lines.
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore", RuntimeWarning)
            scores[index] = measures.measure_monotonicity(
                all_models,
                reference_a,
                reference_b,
                quantifier,
                measures.upward_monotonicity_entropy,
                cfg,
            )
        if (index + 1) % 250 == 0 or index + 1 == len(sample_terms):
            print(
                f"  {index + 1:4d}/{len(sample_terms)} expressions "
                f"({time.time() - started:.1f}s elapsed)"
            )

    corrected = pd.DataFrame(scores, columns=DIRS)
    corrected.insert(0, "expression", sample_terms)
    corrected["degree"] = np.clip(scores, 0.0, 1.0).max(axis=1)
    corrected["metric_version"] = METRIC_VERSION
    return corrected


def add_original_values(
    corrected: pd.DataFrame, run_table: pd.DataFrame
) -> pd.DataFrame:
    original = (
        run_table.dropna(subset=["expression"])
        .groupby("expression")[DIRS + ["degree"]]
        .first()
        .add_suffix("_original")
        .reset_index()
    )
    compared = corrected.merge(
        original, on="expression", how="left", validate="one_to_one"
    )
    for column in DIRS + ["degree"]:
        compared[f"{column}_delta"] = compared[column] - compared[f"{column}_original"]
    return compared


def build_corrected_run_table(
    run_table: pd.DataFrame, corrected: pd.DataFrame
) -> pd.DataFrame:
    renamed = run_table.rename(
        columns={column: f"{column}_original" for column in DIRS + ["degree"]}
    )
    replacement = corrected[["expression", *DIRS, "degree", "metric_version"]]
    merged = renamed.merge(
        replacement, on="expression", how="left", validate="many_to_one"
    )
    missing = int(merged["degree"].isna().sum())
    if missing:
        raise ValueError(f"{missing} run rows did not receive corrected scores")
    return merged


def verify_complement_symmetry(
    corrected: pd.DataFrame, by_term: dict, universe
) -> None:
    refs = universe.referents
    vectors = {}
    for term in corrected["expression"]:
        expression = by_term[term]
        vector = np.fromiter(
            (expression.meaning.mapping[ref] for ref in refs),
            dtype=bool,
            count=len(refs),
        )
        vectors[vector.tobytes()] = term

    indexed = corrected.set_index("expression")
    errors = []
    seen = set()
    for vector_bytes, term in vectors.items():
        vector = np.frombuffer(vector_bytes, dtype=bool)
        complement = vectors.get((~vector).tobytes())
        if complement is None:
            continue
        pair = tuple(sorted((term, complement)))
        if pair in seen:
            continue
        seen.add(pair)
        q = indexed.loc[term]
        nq = indexed.loc[complement]
        errors.extend(
            [
                abs(q["right_upward"] - nq["right_downward"]),
                abs(q["left_upward"] - nq["left_downward"]),
                abs(q["right_downward"] - nq["right_upward"]),
                abs(q["left_downward"] - nq["left_upward"]),
            ]
        )

    maximum = max(errors, default=0.0)
    print(f"complement pairs checked: {len(seen)}")
    print(f"maximum mirror error:     {maximum:.2e}")
    if maximum > 1e-12:
        raise AssertionError(f"Complement-dual metric mirror error is {maximum}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--altk-archive", type=Path, default=DEFAULT_ALTK_ARCHIVE)
    parser.add_argument(
        "--sample",
        type=Path,
        default=repo_root() / "expressions_sample_2k.csv",
    )
    parser.add_argument(
        "--run-csv",
        type=Path,
        default=repo_root() / "outputs/combined_runs_AOC_monotonicity_updated.csv",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=repo_root() / "outputs/monotonicity_values_corrected_2k.csv",
    )
    parser.add_argument(
        "--runs-output",
        type=Path,
        default=repo_root() / "outputs/combined_runs_AOC_monotonicity_corrected.csv",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Development-only prefix limit; omit for the full 2,000 expressions.",
    )
    args = parser.parse_args()

    universe, by_term = load_archive(args.altk_archive.resolve())
    measures = load_alternative_measures()
    sample = pd.read_csv(args.sample)
    sample_terms = sample["term_expression"].tolist()
    if args.limit is not None:
        sample_terms = sample_terms[: args.limit]
    print(f"calculating complement-dual metric for {len(sample_terms)} expressions")

    corrected = calculate_scores(universe, by_term, sample_terms, measures)
    run_table = pd.read_csv(args.run_csv)
    compared = add_original_values(corrected, run_table)
    verify_complement_symmetry(corrected, by_term, universe)

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    compared.to_csv(args.scores_output, index=False)
    print(f"wrote {args.scores_output} ({len(compared)} expression rows)")

    if args.limit is None:
        corrected_runs = build_corrected_run_table(run_table, corrected)
        corrected_runs.to_csv(args.runs_output, index=False)
        print(f"wrote {args.runs_output} ({len(corrected_runs)} run rows)")
    else:
        print("prefix run: skipped merged run table")

    changed = compared["degree_delta"].abs() > 1e-12
    print(
        f"degree changed for {changed.sum()}/{len(compared)} expressions; "
        f"mean delta={compared['degree_delta'].mean():+.4f}; "
        f"max |delta|={compared['degree_delta'].abs().max():.4f}"
    )


if __name__ == "__main__":
    main()
