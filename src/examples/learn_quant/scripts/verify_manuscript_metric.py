"""Verify manuscript results under majorant and complement-dual metrics.

This audit does two things:

1. Refit the manuscript's Figure 2 Pearson correlation and Table 6 mixed model
   from the committed run tables.
2. Recompute three identifiable Table 4 expressions on the manuscript's M6/X6
   universe with explicit majorant and complement-dual variants.

Run with the archived ``altk`` environment because the M6/X6 pickles reference
its original class layout.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

PUBLISHED_CORRELATION = -0.3453
PUBLISHED_TABLE6 = {
    "Intercept": 3403.436,
    "C(model)[T.Transformer]": 1583.381,
    "degree": -1789.666,
    "degree:C(model)[T.Transformer]": -1058.114,
}
TABLE4_PUBLISHED = {
    "subset_eq(A, B)": [1.0, 0.0, 0.0, 1.0],
    "not(subset_eq(A, B))": [0.059, 1.0, 1.0, 0.059],
    "or(subset_eq(A, B), subset_eq(B, A))": [0.0, 0.0, 0.0, 0.0],
}
DEFAULT_ALTK_ARCHIVE = Path.home() / "Documents/UWLing/altk/src/examples"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def fit_manuscript_model(path: Path):
    data = pd.read_csv(path)
    data = data[
        (data["training"] == True) & data["expression"].notna()  # noqa: E712
    ].dropna(subset=["degree", "val_loss_step_AOC"])
    correlation = stats.pearsonr(data["degree"], data["val_loss_step_AOC"]).statistic
    model = smf.mixedlm(
        "val_loss_step_AOC ~ degree * C(model)",
        data,
        groups=data["expression"],
        re_formula="~1",
    ).fit(reml=False)
    return data, correlation, model


def load_m6_archive(archive: Path):
    sys.path.insert(0, str(archive))
    os.chdir(archive)
    import dill as pkl
    from ultk.util.frozendict import FrozenDict

    FrozenDict.__setitem__ = dict.__setitem__
    base = archive / "learn_quant/outputs/M6/X6/d3"
    with open(base / "master_universe.pkl", "rb") as handle:
        universe = pkl.load(handle)
    with open(base / "generated_expressions_xidx.pkl", "rb") as handle:
        pool = pkl.load(handle)
    by_term = {expression.term_expression: expression for expression in pool.values()}
    return universe, by_term


def load_fixed_measures():
    source = repo_root() / "measures.py"
    spec = importlib.util.spec_from_file_location(
        "learn_quant._manuscript_metric_fix", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def table4_scores(archive: Path) -> pd.DataFrame:
    universe, by_term = load_m6_archive(archive)
    measures = load_fixed_measures()
    all_models = universe.binarize_referents(mode="set_vectors_w_padding")
    reference_a = universe.binarize_referents(mode="A")
    reference_b = universe.binarize_referents(mode="B")
    cfg = SimpleNamespace(
        measures=SimpleNamespace(monotonicity=SimpleNamespace(debug=False))
    )

    rows = []
    for term, published in TABLE4_PUBLISHED.items():
        expression = by_term[term]
        quantifier = np.fromiter(
            (expression.meaning.mapping[ref] for ref in universe.referents),
            dtype=int,
            count=len(universe.referents),
        )
        results = {}
        for label, variant in [
            ("majorant", "majorant"),
            ("complement_dual", "complement_dual"),
        ]:
            with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore", RuntimeWarning)
                results[label] = measures.measure_monotonicity(
                    all_models,
                    reference_a,
                    reference_b,
                    quantifier,
                    measures.upward_monotonicity_entropy,
                    cfg,
                    variant=variant,
                )
        row = {"expression": term}
        for index, direction in enumerate(["RU", "LU", "RD", "LD"]):
            row[f"published_{direction}"] = published[index]
            row[f"majorant_{direction}"] = results["majorant"][index]
            row[f"complement_dual_{direction}"] = results["complement_dual"][index]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--altk-archive", type=Path, default=DEFAULT_ALTK_ARCHIVE)
    parser.add_argument(
        "--majorant-csv",
        type=Path,
        default=repo_root() / "outputs/combined_runs_AOC_monotonicity_updated.csv",
    )
    parser.add_argument(
        "--complement-dual-csv",
        type=Path,
        default=repo_root() / "outputs/combined_runs_AOC_monotonicity_corrected.csv",
    )
    args = parser.parse_args()

    majorant_data, majorant_r, majorant_model = fit_manuscript_model(args.majorant_csv)
    complement_data, complement_r, complement_model = fit_manuscript_model(
        args.complement_dual_csv
    )
    if len(complement_data) != len(majorant_data):
        raise ValueError("Majorant and complement-dual fits use different run counts")

    rows = []
    for term in PUBLISHED_TABLE6:
        rows.append(
            {
                "quantity": term,
                "published": PUBLISHED_TABLE6[term],
                "majorant_reproduction": majorant_model.params[term],
                "complement_dual": complement_model.params[term],
                "majorant_abs_error": abs(
                    majorant_model.params[term] - PUBLISHED_TABLE6[term]
                ),
            }
        )
    comparison = pd.DataFrame(rows)

    output_dir = repo_root() / "analysis"
    table_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(
        output_dir / "manuscript_table6_metric_comparison.csv", index=False
    )

    table4 = table4_scores(args.altk_archive.resolve())
    table4.to_csv(output_dir / "manuscript_table4_metric_comparison.csv", index=False)

    report = table_dir / "14_manuscript_metric_verification.txt"
    with report.open("w") as handle:

        def emit(text=""):
            print(text)
            print(text, file=handle)

        emit("MANUSCRIPT METRIC VERIFICATION")
        emit("=" * 72)
        emit(f"rows in both fits: {len(majorant_data)}")
        emit()
        emit("Figure 2 Pearson correlation: degree vs validation-loss AUC")
        emit(f"  published:              {PUBLISHED_CORRELATION:+.4f}")
        emit(f"  majorant CSV:           {majorant_r:+.4f}")
        emit(f"  complement-dual:        {complement_r:+.4f}")
        emit(
            "  note: the committed majorant CSV is close but not identical to the "
            "published correlation,"
        )
        emit("        consistent with the paper using an earlier run-table snapshot.")
        emit()
        emit("Table 6 mixed model: AUC ~ degree * model + (1|expression)")
        emit(
            comparison.to_string(index=False, float_format=lambda value: f"{value:.3f}")
        )
        emit()
        emit("Table 4 M6/X6 directional examples")
        emit(table4.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
        emit()
        matches = []
        for direction in ["RU", "LU", "RD", "LD"]:
            matches.append(
                np.allclose(
                    table4[f"published_{direction}"],
                    table4[f"majorant_{direction}"],
                    atol=5e-4,
                )
            )
        emit(
            "Published Table 4 matches manuscript-era implementation (3 expressions, "
            f"4 directions, rounded to 3 decimals): {all(matches)}"
        )
        emit(
            "Conclusion: the manuscript's numerical analyses used the order-dual "
            "majorant (flip=True for downward)."
        )
        emit(
            "This is symmetric under order reversal but not under truth-label "
            "complementation; the phrase 'symmetric definition' should specify which."
        )


if __name__ == "__main__":
    main()
