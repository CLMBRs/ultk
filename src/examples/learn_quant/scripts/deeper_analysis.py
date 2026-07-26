"""Deeper analyses of the quantifier-learning data (verified == Postgres).

Runs three investigations on
``outputs/combined_runs_AOC_monotonicity_updated.csv`` (proven identical to the
live MLflow Postgres DB):

1. Joint model + partial correlations -- does monotonicity carry learning signal
   *independent* of expression length, and vice versa?
2. Directional monotonicity -- are upward-monotone quantifiers easier to learn
   than downward-monotone ones? Left vs right?
3. Per-operator difficulty -- which grammar operators (index, difference, ...)
   drive learning difficulty, beyond raw length?

Outputs printed tables plus figures under ``figures/``:
    partial_corr_heatmap.png
    directional_monotonicity.png
    operator_effects.png

Every section's printed statistical output (correlation tables, OLS/ridge
coefficient tables, nested-model comparisons) is also written verbatim to
``analysis/tables/<section>.txt`` so the regression results are reproducible
artifacts rather than stdout ephemera.

Run:
    python scripts/deeper_analysis.py
"""

from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from reproduce_figures import count_functions, count_leaves

OPERATORS = [
    "and",
    "or",
    "not",
    "cardinality",
    "subset_eq",
    "equals",
    "greater_than",
    "union",
    "intersection",
    "difference",
    "index",
]

# Learning-difficulty targets. Higher = harder.
#   val_loss_step_AOC : area under validation-loss curve (all trained runs)
#   first_step        : step at which val loss crossed 0.05 (converged runs only)
TARGETS = ["val_loss_step_AOC", "first_step"]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def rel(path):
    """Render a path relative to the learn_quant package root when possible,
    so logged/tee'd output is machine-independent."""
    try:
        return Path(path).relative_to(repo_root())
    except ValueError:
        return Path(path)


def load() -> pd.DataFrame:
    df = pd.read_csv(
        repo_root() / "outputs" / "combined_runs_AOC_monotonicity_updated.csv"
    )
    df = df[df["expression"].notna()].copy()
    df["leaf_count"] = df["expression"].apply(count_leaves)
    df["func_count"] = df["expression"].apply(count_functions)
    # operator counts (how many times each operator appears in the expression)
    for op in OPERATORS:
        pat = re.compile(rf"\b{op}\(")
        df[f"op_{op}"] = df["expression"].apply(lambda e, p=pat: len(p.findall(e)))
    return df


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


# --------------------------------------------------------------------------- #
# 1. Joint model + partial correlations
# --------------------------------------------------------------------------- #
def partial_corr(
    df: pd.DataFrame, x: str, y: str, controls: list[str]
) -> tuple[float, float, int]:
    """Partial correlation of x and y controlling for `controls`.

    Regress x on controls and y on controls, correlate the residuals.
    """
    sub = df[[x, y, *controls]].dropna()
    if len(sub) < 10:
        return np.nan, np.nan, len(sub)
    C = sm.add_constant(sub[controls].astype(float))
    rx = sub[x].astype(float) - sm.OLS(sub[x].astype(float), C).fit().predict(C)
    ry = sub[y].astype(float) - sm.OLS(sub[y].astype(float), C).fit().predict(C)
    r, p = stats.pearsonr(rx, ry)
    return r, p, len(sub)


def analysis_joint(df: pd.DataFrame, outdir: Path) -> None:
    print("\n" + "=" * 72)
    print("1. JOINT MODEL + PARTIAL CORRELATIONS")
    print("=" * 72)

    trained = df[df["training"] == True].copy()  # noqa: E712

    # --- raw vs partial correlations against AUC ---
    # Monotonicity = `degree` (max over the four directional senses, clipped to
    # [0,1]) -- the theory-standard degree of monotonicity and the measure the
    # paper's Figure 1 plots. (The old `monotonicity_entropic` mean-of-directions
    # column was inconsistent with the directional data and is no longer used.)
    print("\nCorrelations with val_loss_step_AOC (learning difficulty):")
    print(
        f"{'predictor':<16}{'raw r':>10}{'partial r':>12}  (partial controls for the other)"
    )
    for x, ctrl in [
        ("degree", ["func_count"]),
        ("func_count", ["degree"]),
        ("leaf_count", ["degree"]),
    ]:
        sub = trained[[x, "val_loss_step_AOC"]].dropna()
        raw = stats.pearsonr(sub[x], sub["val_loss_step_AOC"])[0]
        pr, pp, n = partial_corr(trained, x, "val_loss_step_AOC", ctrl)
        print(f"{x:<16}{raw:>10.3f}{pr:>12.3f}   (n={n}, p={pp:.1e})")

    # --- joint OLS: standardized coefficients ---
    print(
        "\nJoint OLS  (standardized):  val_loss_step_AOC ~ degree + func_count + C(model)"
    )
    d = trained.dropna(subset=["val_loss_step_AOC", "degree", "func_count"]).copy()
    d["aoc_z"] = _z(d["val_loss_step_AOC"])
    d["mono_z"] = _z(d["degree"])
    d["func_z"] = _z(d["func_count"])
    m = smf.ols("aoc_z ~ mono_z + func_z + C(model)", data=d).fit()
    print(m.summary().tables[1])
    print(f"  R^2 = {m.rsquared:.3f}   n = {int(m.nobs)}")

    # --- partial-correlation heatmap among key vars ---
    keyvars = ["val_loss_step_AOC", "first_step", "degree", "func_count", "leaf_count"]
    corr = trained[keyvars].corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keyvars)))
    ax.set_yticks(range(len(keyvars)))
    labels = ["AUC", "first_step", "monotonicity\n(degree)", "func_count", "leaf_count"]
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(keyvars)):
        for j in range(len(keyvars)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if abs(corr.iloc[i, j]) > 0.5 else "black",
                fontsize=10,
            )
    ax.set_title("Pearson correlations (trained runs)", fontweight="bold")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outdir / "partial_corr_heatmap.png", dpi=300)
    plt.close(fig)
    print(f"\n  saved {rel(outdir / 'partial_corr_heatmap.png')}")


# --------------------------------------------------------------------------- #
# 4. Directional monotonicity
# --------------------------------------------------------------------------- #
def analysis_directional(df: pd.DataFrame, outdir: Path) -> None:
    print("\n" + "=" * 72)
    print("4. DIRECTIONAL MONOTONICITY")
    print("=" * 72)

    trained = df[df["training"] == True].copy()  # noqa: E712
    dirs = ["right_upward", "left_upward", "right_downward", "left_downward"]
    n_auc = int(trained["val_loss_step_AOC"].notna().sum())
    print(f"\n(n = {n_auc} trained runs with AUC)")

    print("\nCorrelation of each directional monotonicity with val_loss_step_AOC:")
    print("(negative = that kind of monotonicity is associated with EASIER learning)")
    for c in dirs:
        sub = trained[[c, "val_loss_step_AOC"]].dropna()
        r, p = stats.pearsonr(sub[c], sub["val_loss_step_AOC"])
        pr, pp, n = partial_corr(trained, c, "val_loss_step_AOC", ["func_count"])
        print(f"  {c:<16} raw r={r:+.3f}  partial(func) r={pr:+.3f}   (n={n})")

    # upward vs downward aggregate (max of the two directions)
    trained["upward"] = trained[["right_upward", "left_upward"]].max(axis=1)
    trained["downward"] = trained[["right_downward", "left_downward"]].max(axis=1)
    trained["rightward"] = trained[["right_upward", "right_downward"]].max(axis=1)
    trained["leftward"] = trained[["left_upward", "left_downward"]].max(axis=1)

    print("\nJoint OLS: val_loss_step_AOC ~ upward + downward + func_count + C(model)")
    d = trained.dropna(subset=["val_loss_step_AOC", "func_count"]).copy()
    m = smf.ols(
        "val_loss_step_AOC ~ upward + downward + func_count + C(model)", data=d
    ).fit()
    print(
        f"  n = {int(m.nobs)}   R^2 = {m.rsquared:.3f}   adj R^2 = {m.rsquared_adj:.3f}"
    )
    print(m.summary().tables[1])

    # figure: mean AUC for high vs low in each direction
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # panel A: AUC vs each directional degree (binned means)
    ax = axes[0]
    colors = {
        "right_upward": "#1b9e77",
        "left_upward": "#66c2a5",
        "right_downward": "#d95f02",
        "left_downward": "#fc8d62",
    }
    for c in dirs:
        sub = trained.dropna(subset=[c, "val_loss_step_AOC"])
        bins = pd.cut(sub[c], np.linspace(0, 1, 6))
        means = sub.groupby(bins, observed=True)["val_loss_step_AOC"].mean()
        centers = [iv.mid for iv in means.index]
        ax.plot(centers, means.values, "o-", color=colors[c], label=c, lw=2)
    ax.set_xlabel("Directional monotonicity degree", fontweight="bold")
    ax.set_ylabel("Mean val_loss_step_AOC", fontweight="bold")
    ax.set_title("Learning difficulty vs monotonicity direction", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=9)

    # panel B: upward vs downward, split by model
    ax = axes[1]
    trained["up_hi"] = (trained["upward"] > 0.5).map(
        {True: "high upward", False: "low upward"}
    )
    trained["down_hi"] = (trained["downward"] > 0.5).map(
        {True: "high downward", False: "low downward"}
    )
    grp = (
        trained.dropna(subset=["val_loss_step_AOC"])
        .groupby(["up_hi", "down_hi"], observed=True)["val_loss_step_AOC"]
        .mean()
        .unstack()
    )
    grp.plot(kind="bar", ax=ax, color=["#8da0cb", "#e78ac3"])
    ax.set_ylabel("Mean val_loss_step_AOC", fontweight="bold")
    ax.set_title("Upward vs downward monotonicity", fontweight="bold")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outdir / "directional_monotonicity.png", dpi=300)
    plt.close(fig)
    print(f"\n  saved {rel(outdir / 'directional_monotonicity.png')}")


# --------------------------------------------------------------------------- #
# 6. Per-operator difficulty
# --------------------------------------------------------------------------- #
def analysis_operators(df: pd.DataFrame, outdir: Path) -> None:
    print("\n" + "=" * 72)
    print("6. PER-OPERATOR DIFFICULTY")
    print("=" * 72)

    from sklearn.linear_model import Ridge

    trained = df[df["training"] == True].copy()  # noqa: E712
    opcols = [f"op_{op}" for op in OPERATORS]

    print("\nOperator prevalence (mean count per expression):")
    for op in OPERATORS:
        print(f"  {op:<14} {trained[f'op_{op}'].mean():.2f}")

    # The operator counts are exactly collinear (grammar constraints, e.g.
    # #and + #or = #bool-predicates - 1), so plain OLS is rank-deficient and
    # returns nonsense coefficients. Use RIDGE regression (L2) on standardized
    # predictors -- it distributes weight stably across collinear columns -- and
    # bootstrap for confidence intervals.
    d = trained.dropna(subset=["val_loss_step_AOC"]).copy()
    d["model_T"] = (d["model"] == "Transformer").astype(float)

    feat = opcols + ["model_T"]
    X = d[feat].to_numpy(float)
    y = d["val_loss_step_AOC"].to_numpy(float)

    # standardize predictors so coefficients are comparable (per +1 SD)
    Xm, Xs = X.mean(0), X.std(0)
    Xs[Xs == 0] = 1.0
    Xz = (X - Xm) / Xs

    alpha = 10.0
    base = Ridge(alpha=alpha).fit(Xz, y)

    # bootstrap CIs
    rng = np.random.default_rng(0)
    B = 300
    boot = np.zeros((B, Xz.shape[1]))
    n = len(y)
    for b in range(B):
        idx = rng.integers(0, n, n)
        boot[b] = Ridge(alpha=alpha).fit(Xz[idx], y[idx]).coef_
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    print(f"\nRidge (alpha={alpha}) on standardized operator counts + model.")
    print("Coefficient = Δ val_loss_step_AOC per +1 SD of that operator's count:")
    coef = pd.Series(base.coef_, index=feat)
    order = coef[opcols].sort_values().index
    print(f"{'operator':<16}{'coef':>10}{'95% CI':>22}")
    for c in order:
        i = feat.index(c)
        star = "*" if (lo[i] > 0) or (hi[i] < 0) else " "
        print(
            f"{c.replace('op_',''):<16}{coef[c]:>10.1f}   [{lo[i]:8.1f},{hi[i]:8.1f}] {star}"
        )

    # variance explained: length-only vs operator-identity (via CV-free R^2)
    from sklearn.metrics import r2_score

    m_ops = r2_score(y, base.predict(Xz))
    Xlen = d[["func_count", "model_T"]].to_numpy(float)
    Xlenz = (Xlen - Xlen.mean(0)) / Xlen.std(0)
    m_len = r2_score(y, Ridge(alpha=alpha).fit(Xlenz, y).predict(Xlenz))
    print(f"\n  length-only (func_count+model) R^2 = {m_len:.3f}")
    print(f"  operator-identity R^2              = {m_ops:.3f}")
    print(f"  improvement from operator identity : +{m_ops - m_len:.3f}")

    # figure: operator coefficients (SD units) with bootstrap CIs
    op_only = [c for c in order]
    y_pos = np.arange(len(op_only))
    cvals = [coef[c] for c in op_only]
    clo = [coef[c] - lo[feat.index(c)] for c in op_only]
    chi = [hi[feat.index(c)] - coef[c] for c in op_only]
    names = [c.replace("op_", "") for c in op_only]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#c0392b" if coef[c] > 0 else "#2471a3" for c in op_only]
    ax.errorbar(
        cvals, y_pos, xerr=[clo, chi], fmt="none", ecolor="gray", capsize=3, zorder=1
    )
    ax.scatter(cvals, y_pos, color=colors, zorder=2, s=45)
    ax.axvline(0, color="black", linestyle="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel(
        "Δ val_loss_step_AOC per +1 SD of operator count\n(red = harder, blue = easier; Ridge, bootstrap 95% CI)",
        fontweight="bold",
    )
    ax.set_title("Per-operator contribution to learning difficulty", fontweight="bold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "operator_effects.png", dpi=300)
    plt.close(fig)
    print(f"\n  saved {rel(outdir / 'operator_effects.png')}")


# --------------------------------------------------------------------------- #
# 7. Nested-model comparison: does monotonicity add to length (and vice versa)?
# --------------------------------------------------------------------------- #
def analysis_nested(df: pd.DataFrame, outdir: Path) -> None:
    """Hierarchical regression: which predictor is stronger, and do they add?

    Predicting learning difficulty (Validation Loss AUC) from expression length
    and monotonicity, always controlling for model type. We fit a nest of models
    and quantify the *unique* contribution of each predictor:

        base : AUC ~ C(model)
        L    : AUC ~ length + C(model)
        M    : AUC ~ monotonicity + C(model)
        LM   : AUC ~ length + monotonicity + C(model)

    Reported for each addition:
      - Delta R^2  (incremental variance explained)
      - nested F-test p-value (statsmodels anova_lm)
      - AIC / BIC (lower = better)
    Plus a commonality analysis partitioning explained variance into
    length-unique, monotonicity-unique, and shared.
    """
    from statsmodels.stats.anova import anova_lm

    print("\n" + "=" * 72)
    print("7. NESTED-MODEL COMPARISON  (does adding monotonicity improve fit?)")
    print("=" * 72)

    # length = func_count (equivalent to leaf_count, r=0.94); monotonicity = degree.
    # 'degree' is the measure the paper's Figure 1 plots on its y-axis.
    d = (
        df[df["training"] == True]
        .dropna(subset=["val_loss_step_AOC", "func_count", "degree"])  # noqa: E712
        .copy()
    )
    d["length_z"] = _z(d["func_count"])
    d["mono_z"] = _z(d["degree"])
    d["aoc"] = d["val_loss_step_AOC"]

    base = smf.ols("aoc ~ C(model)", data=d).fit()
    L = smf.ols("aoc ~ length_z + C(model)", data=d).fit()
    M = smf.ols("aoc ~ mono_z + C(model)", data=d).fit()
    LM = smf.ols("aoc ~ length_z + mono_z + C(model)", data=d).fit()

    def line(name, m):
        print(
            f"  {name:<28} R^2={m.rsquared:6.3f}   AIC={m.aic:9.0f}   BIC={m.bic:9.0f}"
        )

    print(f"\nModels (n={int(LM.nobs)}), controlling for model type:")
    line("base: model only", base)
    line("L:    + length", L)
    line("M:    + monotonicity", M)
    line("LM:   + length + monotonicity", LM)

    # Incremental tests (nested F).
    print("\nIncremental contribution (nested F-tests):")
    fLM_from_L = anova_lm(L, LM)
    fLM_from_M = anova_lm(M, LM)
    dR2_addM = LM.rsquared - L.rsquared
    dR2_addL = LM.rsquared - M.rsquared
    print(
        f"  add MONOTONICITY on top of length:  ΔR^2=+{dR2_addM:.3f}  "
        f"F={fLM_from_L['F'][1]:.1f}  p={fLM_from_L['Pr(>F)'][1]:.2e}"
    )
    print(
        f"  add LENGTH on top of monotonicity:  ΔR^2=+{dR2_addL:.3f}  "
        f"F={fLM_from_M['F'][1]:.1f}  p={fLM_from_M['Pr(>F)'][1]:.2e}"
    )

    # Standardized effect sizes in the full model (compare |beta|).
    print("\nStandardized coefficients in the full model LM (|beta| = strength):")
    b = LM.params
    print(f"  length (per +1 SD)        : {b['length_z']:+8.1f}")
    print(f"  monotonicity (per +1 SD)  : {b['mono_z']:+8.1f}")
    stronger = "monotonicity" if abs(b["mono_z"]) > abs(b["length_z"]) else "length"
    ratio = abs(b["mono_z"]) / abs(b["length_z"])
    print(
        f"  => {stronger} is the stronger predictor "
        f"(|beta| ratio mono/length = {ratio:.2f})"
    )

    # Commonality analysis (variance beyond the model-type baseline).
    r2_base = base.rsquared
    total = LM.rsquared - r2_base  # explained by L & M jointly
    unique_L = LM.rsquared - M.rsquared  # length's unique part
    unique_M = LM.rsquared - L.rsquared  # monotonicity's unique part
    common = total - unique_L - unique_M  # shared
    print("\nCommonality analysis (variance beyond model-type baseline):")
    print(f"  total explained by length+monotonicity : {total:.3f}")
    print(f"    unique to length                     : {unique_L:.3f}")
    print(f"    unique to monotonicity               : {unique_M:.3f}")
    print(f"    shared (common)                      : {common:.3f}")

    # Figure: (a) incremental R^2 waterfall, (b) commonality partition.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    names = ["model\nonly", "+length", "+monotonicity\n(full)", "+length\n(full)"]
    r2s = [base.rsquared, L.rsquared, LM.rsquared, LM.rsquared]
    ax.bar(range(4), r2s, color=["#999999", "#4C72B0", "#55A868", "#55A868"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylabel("R² (Validation Loss AUC)", fontweight="bold")
    ax.set_title("Cumulative variance explained", fontweight="bold")
    ax.annotate(
        f"+{dR2_addL:.3f}\n(add length)",
        xy=(2.5, LM.rsquared),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    parts = [unique_L, common, unique_M]
    labels = [
        f"length only\n{unique_L:.3f}",
        f"shared\n{common:.3f}",
        f"monotonicity only\n{unique_M:.3f}",
    ]
    colors = ["#4C72B0", "#C7C7C7", "#55A868"]
    left = 0
    for p, lab, c in zip(parts, labels, colors):
        ax.barh(0, p, left=left, color=c, edgecolor="white")
        if p > 0.003:
            ax.text(left + p / 2, 0, lab, ha="center", va="center", fontsize=9)
        left += p
    ax.set_xlim(0, total * 1.02)
    ax.set_yticks([])
    ax.set_xlabel("R² beyond model-type baseline", fontweight="bold")
    ax.set_title("Unique vs shared contribution", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "nested_model_comparison.png", dpi=300)
    plt.close(fig)
    print(f"\n  saved {rel(outdir / 'nested_model_comparison.png')}")


# --------------------------------------------------------------------------- #
# 8. Predict MONOTONICITY from complexity vs learnability -- which explains it?
# --------------------------------------------------------------------------- #
def analysis_predict_monotonicity(df: pd.DataFrame, outdir: Path) -> None:
    """Flip the outcome: predict monotonicity from length and learnability (AUC).

    Question: between an expression's *complexity* (length) and its *learnability*
    (Validation Loss AUC), which better explains its monotonicity?

    Monotonicity (`degree`) and length are per-expression constants; AUC is
    per-run. To avoid pseudoreplication we aggregate to ONE ROW PER EXPRESSION
    (mean AUC across its trained runs), then fit a nest of models:

        base : degree ~ 1                 (intercept only; R^2 = 0)
        L    : degree ~ length
        A    : degree ~ AUC
        LA   : degree ~ length + AUC

    Reported: incremental R^2, nested F-tests, standardized coefficients, and a
    commonality partition (length-unique / AUC-unique / shared).

    Caveat: this is descriptive association, not causal. Monotonicity is an
    intrinsic property of the expression; AUC is an *outcome* of training, so the
    natural causal arrow is monotonicity -> learnability. This analysis asks only
    which covaries more strongly with monotonicity.
    """
    from statsmodels.stats.anova import anova_lm

    print("\n" + "=" * 72)
    print(
        "8. PREDICTING MONOTONICITY  (complexity vs learnability -- which explains it?)"
    )
    print("=" * 72)

    trained = df[df["training"] == True].dropna(  # noqa: E712
        subset=["val_loss_step_AOC", "degree", "func_count"]
    )
    # one row per expression: monotonicity & length are constant; average AUC
    agg = (
        trained.groupby("expression")
        .agg(
            degree=("degree", "first"),
            func_count=("func_count", "first"),
            leaf_count=("leaf_count", "first"),
            auc=("val_loss_step_AOC", "mean"),
        )
        .reset_index()
    )
    agg["length_z"] = _z(agg["func_count"])
    agg["auc_z"] = _z(agg["auc"])
    n = len(agg)
    print(f"\nAggregated to {n} unique expressions (mean AUC across trained runs).")

    print("\nSimple correlations with monotonicity (degree):")
    for name, col in [
        ("length (func_count)", "func_count"),
        ("learnability (mean AUC)", "auc"),
    ]:
        r, p = stats.pearsonr(agg[col], agg["degree"])
        print(f"  {name:<26} r={r:+.3f}   (p={p:.2e})")

    base = smf.ols("degree ~ 1", data=agg).fit()
    L = smf.ols("degree ~ length_z", data=agg).fit()
    A = smf.ols("degree ~ auc_z", data=agg).fit()
    LA = smf.ols("degree ~ length_z + auc_z", data=agg).fit()

    def line(name, m):
        print(f"  {name:<26} R^2={m.rsquared:6.3f}   AIC={m.aic:9.0f}")

    print("\nNested models (outcome = monotonicity):")
    line("base: intercept only", base)
    line("L:    + length", L)
    line("A:    + learnability (AUC)", A)
    line("LA:   + length + AUC", LA)

    print("\nIncremental contribution (nested F-tests):")
    f_addA = anova_lm(L, LA)
    f_addL = anova_lm(A, LA)
    dR2_addA = LA.rsquared - L.rsquared
    dR2_addL = LA.rsquared - A.rsquared
    print(
        f"  add LEARNABILITY on top of length:  ΔR^2=+{dR2_addA:.3f}  "
        f"F={f_addA['F'][1]:.1f}  p={f_addA['Pr(>F)'][1]:.2e}"
    )
    print(
        f"  add LENGTH on top of learnability:  ΔR^2=+{dR2_addL:.3f}  "
        f"F={f_addL['F'][1]:.1f}  p={f_addL['Pr(>F)'][1]:.2e}"
    )

    print("\nStandardized coefficients in the full model (|beta| = strength):")
    b = LA.params
    print(f"  length (per +1 SD)        : {b['length_z']:+7.3f}")
    print(f"  learnability (per +1 SD)  : {b['auc_z']:+7.3f}")
    stronger = (
        "learnability (AUC)" if abs(b["auc_z"]) > abs(b["length_z"]) else "length"
    )
    ratio = abs(b["auc_z"]) / abs(b["length_z"]) if b["length_z"] else float("inf")
    print(
        f"  => {stronger} is the stronger predictor of monotonicity "
        f"(|beta| ratio AUC/length = {ratio:.2f})"
    )

    total = LA.rsquared
    unique_L = LA.rsquared - A.rsquared
    unique_A = LA.rsquared - L.rsquared
    common = total - unique_L - unique_A
    print("\nCommonality analysis (variance in monotonicity explained):")
    print(f"  total explained by length+AUC : {total:.3f}")
    print(f"    unique to length            : {unique_L:.3f}")
    print(f"    unique to learnability      : {unique_A:.3f}")
    print(f"    shared                      : {common:.3f}")

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    names = ["intercept", "+length", "+AUC", "full\n(L+AUC)"]
    r2s = [base.rsquared, L.rsquared, A.rsquared, LA.rsquared]
    ax.bar(range(4), r2s, color=["#999999", "#4C72B0", "#C44E52", "#8172B3"])
    ax.set_xticks(range(4))
    ax.set_xticklabels(names)
    ax.set_ylabel("R² (monotonicity)", fontweight="bold")
    ax.set_title("Explaining monotonicity", fontweight="bold")
    for i, v in enumerate(r2s):
        ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    parts = [unique_L, common, unique_A]
    labels = [
        f"length only\n{unique_L:.3f}",
        f"shared\n{common:.3f}",
        f"learnability only\n{unique_A:.3f}",
    ]
    colors = ["#4C72B0", "#C7C7C7", "#C44E52"]
    left = 0
    for p, lab, c in zip(parts, labels, colors):
        ax.barh(0, p, left=left, color=c, edgecolor="white")
        if p > 0.003:
            ax.text(left + p / 2, 0, lab, ha="center", va="center", fontsize=9)
        left += p
    ax.set_xlim(0, max(total * 1.02, 1e-3))
    ax.set_yticks([])
    ax.set_xlabel("R² in monotonicity", fontweight="bold")
    ax.set_title("Unique vs shared contribution", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "predict_monotonicity.png", dpi=300)
    plt.close(fig)
    print(f"\n  saved {rel(outdir / 'predict_monotonicity.png')}")


class _Tee(io.TextIOBase):
    """Write-through stream: everything printed goes to all wrapped streams."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        for st in self._streams:
            st.write(s)
        return len(s)

    def flush(self) -> None:
        for st in self._streams:
            st.flush()


def main() -> None:
    outdir = repo_root() / "figures"
    outdir.mkdir(exist_ok=True)
    tables_dir = repo_root() / "analysis" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = load()
    print(
        f"Loaded {len(df)} rows; {int((df['training'] == True).sum())} trained."
    )  # noqa: E712

    # Each section's printed output (correlations, OLS/ridge tables, nested-model
    # stats) is teed verbatim into analysis/tables/<section>.txt.
    sections = [
        ("1_joint_partial_correlations", analysis_joint),
        ("4_directional_monotonicity", analysis_directional),
        ("6_operator_effects", analysis_operators),
        ("7_nested_model_comparison", analysis_nested),
        ("8_predict_monotonicity", analysis_predict_monotonicity),
    ]
    for name, fn in sections:
        path = tables_dir / f"{name}.txt"
        with open(path, "w") as f:
            with redirect_stdout(_Tee(sys.stdout, f)):
                fn(df, outdir)
        print(f"  table saved to {rel(path)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
