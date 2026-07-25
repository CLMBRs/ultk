"""Reproduce the learning figures from local data (no MLflow server required).

This script is path-independent: every path is resolved relative to this file,
so it runs from anywhere without editing hard-coded absolute paths.

It produces the following figures under ``learn_quant/figures/``:

0. ``paper_figure1.png`` -- an EXACT reproduction of Figure 1 in the SALT35
   manuscript ("Aggregate training validation loss and degree of monotonicity
   for each quantifier"). Built with plotnine from cell ``#VSC-33341683`` of
   ``notebooks/graph_expressions copy.ipynb``: ``degree`` (Monotonicity) vs
   ``val_loss_step_AOC`` (Validation Loss AUC), coloured by model, with a red
   dashed linear fit. Uses plotnine's default salmon/cyan palette on a white
   background, matching the published PDF.

1. ``monotonicity_vs_training_step.png`` -- the flagship figure from
   ``notebooks/plot_experiments.ipynb``: final monotonicity (``degree``) of each
   run vs. the training step at which the run converged (a learning-speed /
   difficulty axis). Built from the aggregated CSV (fast, offline).

2. ``depth_vs_learning.png`` -- reproduction of the "expression depth vs
   learning" relationship (depth predicts learning difficulty in the manuscript
   regressions ``val_loss_step_AOC ~ ... + expression_depth`` and the
   training-success-rate-by-depth aggregation in ``statistical_model.ipynb``).

   NOTE ON NAMING: the column ``val_loss_step_AOC`` = ``SUM(val_loss_step)`` is
   the integral of the validation loss over training -- i.e. the **area UNDER
   the validation-loss curve (AUC)**. The ``AOC`` in the column name is a
   historical misnomer; all figure labels say "Validation Loss AUC". Higher AUC
   = more accumulated loss = harder to learn (hence a positive slope vs length
   and a negative slope vs monotonicity).

3. ``length_vs_learning.png`` -- the requested VARIANT: instead of parenthesis
   nesting depth, complexity is measured as the *length* of the expression, i.e.
   the number of leaf nodes (atoms such as ``A``, ``B`` and integer indices).
   Leaf count has far finer resolution than depth (which is essentially fixed at
   4 in this dataset), so it exposes the learning trend more clearly.

Usage::

    python -m learn_quant.scripts.reproduce_figures
    python scripts/reproduce_figures.py --outdir /tmp/figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Path handling (everything relative to this file)
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    """Return the ``learn_quant`` package root (parent of the scripts folder)."""
    return Path(__file__).resolve().parent.parent


def rel(path):
    """Render a path relative to the learn_quant package root when possible,
    so logged/tee'd output is machine-independent."""
    try:
        return Path(path).relative_to(repo_root())
    except ValueError:
        return Path(path)


# --------------------------------------------------------------------------- #
# Expression parsing: count leaves (atoms) instead of measuring nesting depth
# --------------------------------------------------------------------------- #
def _split_top_level_args(inside: str) -> list[str]:
    """Split comma-separated arguments while respecting nested parentheses."""
    args: list[str] = []
    current = ""
    depth = 0
    for char in inside:
        if char == "(":
            depth += 1
            current += char
        elif char == ")":
            depth -= 1
            current += char
        elif char == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += char
    if current.strip():
        args.append(current.strip())
    return args


def count_leaves(expr: str) -> int:
    """Number of leaf nodes (terminals/atoms) in an expression string.

    Leaves are operands with no arguments, e.g. ``A``, ``B`` and integer
    indices. Operators such as ``and``/``cardinality``/``subset_eq`` are
    internal nodes. This is the "length" of the expression.
    """
    expr = expr.strip()
    open_idx = expr.find("(")
    if open_idx == -1:
        return 1  # bare atom (A, B, or an index)
    inside = expr[open_idx + 1 : expr.rfind(")")]
    return sum(count_leaves(arg) for arg in _split_top_level_args(inside))


def count_nodes(expr: str) -> int:
    """Total number of nodes (operators + operands) -- an alternative size."""
    expr = expr.strip()
    open_idx = expr.find("(")
    if open_idx == -1:
        return 1
    inside = expr[open_idx + 1 : expr.rfind(")")]
    return 1 + sum(count_nodes(arg) for arg in _split_top_level_args(inside))


def count_functions(expr: str) -> int:
    """Number of function/operator applications in an expression string.

    Every non-leaf node is a function application in this grammar (``and``,
    ``or``, ``cardinality``, ``subset_eq``, ``union``, ...), so the function
    count equals ``count_nodes - count_leaves``. This is distinct from *depth*
    (max parenthesis nesting): it counts how many operations there are in total,
    not how deeply they nest.
    """
    expr = expr.strip()
    open_idx = expr.find("(")
    if open_idx == -1:
        return 0  # a bare atom applies no function
    inside = expr[open_idx + 1 : expr.rfind(")")]
    return 1 + sum(count_functions(arg) for arg in _split_top_level_args(inside))



def plot_monotonicity_vs_step_from_csv(df: pd.DataFrame, outpath: Path) -> bool:
    """Same flagship figure but built from the aggregated CSV (fast, offline).

    Uses ``first_step`` (the training step at which the run converged) as the
    training-step axis -- the same conceptual quantity the notebook read from
    ``global_step`` in mlruns, but available without an iCloud/mlruns walk.

    Monotonicity is ``degree`` (max over the four directional senses, clipped to
    [0,1]) -- the theory-standard measure the paper's Figure 1 uses. The former
    ``monotonicity_entropic`` column was inconsistent with the directional data
    and is no longer used.
    """
    if not {"degree", "first_step"}.issubset(df.columns):
        print("  [skip] CSV missing degree/first_step")
        return False
    data = df.dropna(subset=["degree", "first_step"]).copy()
    if data.empty:
        print("  [skip] no converged rows in CSV")
        return False

    x = data["first_step"].to_numpy(dtype=float)
    y = data["degree"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.15, color="#0073C2", s=18, edgecolors="none")
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, "r--", lw=2, label="linear fit")
    r = np.corrcoef(x, y)[0, 1]
    ax.legend(title=f"r = {r:.2f}", loc="best")
    ax.set_title("Expression Monotonicity vs Training Step", fontweight="bold")
    ax.set_xlabel("Training Step (step at convergence)", fontweight="bold")
    ax.set_ylabel("Monotonicity", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  saved {rel(outpath)}  ({len(data)} runs)")
    return True


# --------------------------------------------------------------------------- #
# Figure 0: EXACT reproduction of the manuscript's Figure 1 (plotnine)
# --------------------------------------------------------------------------- #
def plot_paper_figure1(df: pd.DataFrame, outpath: Path) -> bool:
    """Recreate SALT35 Figure 1 exactly, using plotnine.

    Source: cell ``#VSC-33341683`` in ``notebooks/graph_expressions copy.ipynb``.
    Monotonicity (``degree``) vs Validation Loss AUC (``val_loss_step_AOC``),
    coloured by model, with a red dashed linear fit. The published PDF uses
    plotnine's default salmon/cyan hue palette on a white background (not the
    later blue/yellow restyle committed in the notebook), so no manual colour
    scale is applied here.
    """
    try:
        from plotnine import (
            aes,
            element_line,
            element_rect,
            element_text,
            geom_jitter,
            ggplot,
            guide_legend,
            guides,
            labs,
            stat_smooth,
            theme,
            theme_minimal,
        )
    except ImportError:
        print("  [skip] plotnine not installed; cannot build exact paper figure")
        return False

    if not {"val_loss_step_AOC", "degree", "model"}.issubset(df.columns):
        print("  [skip] CSV missing val_loss_step_AOC/degree/model")
        return False

    plotdf = df.dropna(subset=["val_loss_step_AOC", "degree"]).copy()
    if plotdf.empty:
        print("  [skip] no rows for paper figure")
        return False

    plot = (
        # Point params (alpha=1, size=2) match the committed manuscript cell
        # (cell 82 of ``graph_expressions copy.ipynb``); the published PDF uses
        # the default salmon/cyan palette + white background + red dashed fit.
        ggplot(plotdf, aes(x="val_loss_step_AOC", y="degree", color="model"))
        + geom_jitter(alpha=1, width=10, height=0.01, size=2)
        + stat_smooth(method="lm", color="red", linetype="dashed", se=False, size=1.2)
        + theme_minimal(base_size=14)
        + guides(color=guide_legend(override_aes={"size": 4, "alpha": 0.9}))
        + theme(
            figure_size=(12, 8),
            plot_background=element_rect(fill="white", color="white"),
            panel_background=element_rect(fill="white", color="white"),
            panel_grid_major=element_line(color="gray", size=0.5, linetype="dashed"),
            panel_grid_minor=element_line(color="lightgray", size=0.25, alpha=0.15),
            axis_title_x=element_text(size=22, color="black"),
            axis_title_y=element_text(size=22, color="black"),
            axis_text=element_text(size=14, color="black"),
            axis_line=element_line(color="black", size=0.8),
            axis_ticks=element_line(color="black"),
            legend_title=element_text(size=18, color="black"),
            legend_text=element_text(size=16, color="black"),
            legend_position=(0.85, 0.8),
            legend_direction="vertical",
            legend_key_size=25,
        )
        + labs(x="Validation Loss AUC", y="Monotonicity", color="Model")
    )
    plot.save(outpath, dpi=600, verbose=False)
    print(f"  saved {rel(outpath)}  ({len(plotdf)} points)")
    return True


def plot_length_vs_auc(
    df: pd.DataFrame,
    outpath: Path,
    ycol: str = "leaf_count",
    ylabel: str = "Expression Length (leaf nodes)",
) -> bool:
    """Twin of the paper's Figure 1, but with an expression-size measure on y.

    Same style as ``plot_paper_figure1`` (Validation Loss AUC on x, coloured by
    model, red dashed linear fit), but the y-axis is a size/complexity measure
    (``leaf_count`` = number of atoms, or ``func_count`` = number of function
    applications) instead of monotonicity. Shows how expression size relates to
    learning difficulty.
    """
    try:
        from plotnine import (
            aes,
            element_line,
            element_rect,
            element_text,
            geom_jitter,
            ggplot,
            guide_legend,
            guides,
            labs,
            stat_smooth,
            theme,
            theme_minimal,
        )
    except ImportError:
        print("  [skip] plotnine not installed; cannot build length figure")
        return False

    if ycol not in df.columns:
        print(f"  [skip] column {ycol} not present")
        return False

    plotdf = df.dropna(subset=["val_loss_step_AOC", ycol]).copy()
    if plotdf.empty:
        print("  [skip] no rows for length figure")
        return False

    plot = (
        ggplot(plotdf, aes(x="val_loss_step_AOC", y=ycol, color="model"))
        + geom_jitter(alpha=0.5, width=10, height=0.15, size=1.5)
        + stat_smooth(method="lm", color="red", linetype="dashed", se=False, size=1.2)
        + theme_minimal(base_size=14)
        + guides(color=guide_legend(override_aes={"size": 4, "alpha": 0.9}))
        + theme(
            figure_size=(12, 8),
            plot_background=element_rect(fill="white", color="white"),
            panel_background=element_rect(fill="white", color="white"),
            panel_grid_major=element_line(color="gray", size=0.5, linetype="dashed"),
            panel_grid_minor=element_line(color="lightgray", size=0.25, alpha=0.15),
            axis_title_x=element_text(size=22, color="black"),
            axis_title_y=element_text(size=22, color="black"),
            axis_text=element_text(size=14, color="black"),
            axis_line=element_line(color="black", size=0.8),
            axis_ticks=element_line(color="black"),
            legend_title=element_text(size=18, color="black"),
            legend_text=element_text(size=16, color="black"),
            legend_position=(0.85, 0.8),
            legend_direction="vertical",
            legend_key_size=25,
        )
        + labs(
            x="Validation Loss AUC",
            y=ylabel,
            color="Model",
        )
    )
    plot.save(outpath, dpi=600, verbose=False)
    print(f"  saved {rel(outpath)}  ({len(plotdf)} points)")
    return True


# --------------------------------------------------------------------------- #
# Figures 2 & 3: complexity (depth OR length) vs learning
# --------------------------------------------------------------------------- #
LEARNING_MEASURES = [
    ("first_step", "Steps to converge\n(learning speed)"),
    ("val_loss_step_AOC", "Validation Loss AUC\n(learning difficulty)"),
]


def plot_complexity_vs_learning(
    df: pd.DataFrame, xcol: str, xlabel: str, outpath: Path
) -> None:
    """Two-panel figure: each learning measure vs a complexity measure.

    Points are jittered on x, coloured by model; a linear fit and Pearson r are
    drawn, and binned means show the trend.
    """
    trained = df[df["training"] == True]  # noqa: E712  (works for bool or "True")

    n_panels = len(LEARNING_MEASURES)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    rng = np.random.default_rng(0)
    model_colors = {"LSTM": "#E69F00", "Transformer": "#56B4E9"}

    for ax, (ycol, ylabel) in zip(axes, LEARNING_MEASURES):
        data = trained.dropna(subset=[xcol, ycol]).copy()
        if data.empty:
            ax.set_visible(False)
            continue

        x = data[xcol].to_numpy(dtype=float)
        y = data[ycol].to_numpy(dtype=float)
        jitter = rng.uniform(-0.18, 0.18, size=len(x))

        if "model" in data.columns and data["model"].notna().any():
            for model_name, sub in data.groupby("model"):
                idx = data["model"] == model_name
                ax.scatter(
                    x[idx.to_numpy()] + jitter[idx.to_numpy()],
                    y[idx.to_numpy()],
                    alpha=0.25,
                    s=14,
                    edgecolors="none",
                    color=model_colors.get(str(model_name), None),
                    label=str(model_name),
                )
        else:
            ax.scatter(x + jitter, y, alpha=0.25, s=14, edgecolors="none")

        # Linear fit + correlation.
        if len(np.unique(x)) >= 2:
            slope, intercept = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 100)
            ax.plot(xs, slope * xs + intercept, "k--", lw=2, label="linear fit")
            r = np.corrcoef(x, y)[0, 1]
            r_note = f"r = {r:.2f}"
        else:
            r_note = ""

        # Binned means (one point per integer complexity value).
        means = data.groupby(xcol)[ycol].mean()
        ax.plot(
            means.index.to_numpy(dtype=float),
            means.to_numpy(),
            "o-",
            color="red",
            lw=2,
            ms=6,
            label="mean per value",
        )

        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(title=r_note, loc="best", fontsize=8)

    fig.suptitle(outpath.stem.replace("_", " ").title(), fontweight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  saved {rel(outpath)}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    root = repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "outputs" / "combined_runs_AOC_monotonicity_updated.csv",
    )
    parser.add_argument("--outdir", type=Path, default=root / "figures")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load the aggregated CSV once (used by all figures).
    if not args.csv.is_file():
        print(f"[error] CSV not found at {args.csv}")
        return
    df = pd.read_csv(args.csv)
    df["leaf_count"] = df["expression"].apply(count_leaves)
    df["node_count"] = df["expression"].apply(count_nodes)
    df["func_count"] = df["expression"].apply(count_functions)

    # --- Figure 0: exact reproduction of the manuscript's Figure 1 ----------
    print("Figure 0: exact paper Figure 1 (Monotonicity vs Validation Loss AUC)")
    plot_paper_figure1(df, args.outdir / "paper_figure1.png")

    # --- Figure 0b: twin of Figure 1 with expression LENGTH on the y-axis ----
    print("Figure 0b: length (leaf count) vs Validation Loss AUC")
    plot_length_vs_auc(df, args.outdir / "length_vs_auc.png")

    # --- Figure 0c: twin of Figure 1 with FUNCTION COUNT on the y-axis -------
    print("Figure 0c: function count vs Validation Loss AUC")
    plot_length_vs_auc(
        df,
        args.outdir / "functions_vs_auc.png",
        ycol="func_count",
        ylabel="Expression Length (function applications)",
    )

    # --- Figure 1: monotonicity (degree) vs training step -------------------
    # Always built from the CSV, which carries `degree`. The former
    # `--from-mlruns` path read the per-run `monotonicity_entropic` metric, which
    # has no `degree` equivalent logged, so it is no longer used for this figure.
    print("Figure 1: monotonicity (degree) vs training step")
    fig1_path = args.outdir / "monotonicity_vs_training_step.png"
    plot_monotonicity_vs_step_from_csv(df, fig1_path)

    # --- Figures 2 & 3: complexity vs learning (from aggregated CSV) ---------
    print("Figures 2 & 3: complexity vs learning")
    plot_complexity_vs_learning(
        df,
        xcol="expression_depth",
        xlabel="Expression depth\n(parenthesis nesting)",
        outpath=args.outdir / "depth_vs_learning.png",
    )
    plot_complexity_vs_learning(
        df,
        xcol="leaf_count",
        xlabel="Expression length\n(number of leaf nodes / atoms)",
        outpath=args.outdir / "length_vs_learning.png",
    )
    plot_complexity_vs_learning(
        df,
        xcol="func_count",
        xlabel="Expression length\n(number of function applications)",
        outpath=args.outdir / "functions_vs_learning.png",
    )

    print(f"\nDone. Figures written to {rel(args.outdir)}")


if __name__ == "__main__":
    main()
