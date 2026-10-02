"""Review extensions: the additional figures & tables from the 2026-07 reproduction review.

Companion to ``deeper_analysis.py``. Produces, from the same verified CSV:

Figures (written to ``figures/``):
    nested_model_redraw.png        -- clarified version of nested_model_comparison.png:
                                      one bar per DISTINCT model (base / +length / +mono /
                                      full), so the full model appears once, with the two
                                      addition paths (nested F-tests) drawn as arrows.
    directional_panelB_redraw.png  -- clarified version of panel B of
                                      directional_monotonicity.png: downward status on x,
                                      human-readable labels, cell means and n printed.
    operator_by_architecture.png   -- per-operator Ridge coefficients fit SEPARATELY per
                                      architecture, in three panels: (a) absolute AUC
                                      units, (b) Transformer-LSTM difference with
                                      bootstrap CIs, (c) RESCALED within-architecture
                                      (coefficient / that architecture's AUC SD), which
                                      removes the Transformer's larger overall loss scale
                                      and shows whether the difficulty *profile* differs.

Printed tables (also teed to ``analysis/tables/9x_*.txt``):
    directional diagnostics        -- distribution of the four directional degrees,
                                      up/down aggregates, AUC by 2x2 category,
                                      per-architecture split + interaction OLS.
    operator prevalence            -- %% of expressions attested, total & mean counts.
    polarity counterbalance test   -- does `difference` drive the downward surplus?
                                      (it does not); the `not`-count gradient.

Run:
    python scripts/review_extensions.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from scipy import stats

from deeper_analysis import OPERATORS, _Tee, _z, load, repo_root, rel

DIRS = ["right_upward", "left_upward", "right_downward", "left_downward"]


def _add_updown(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["upward"] = frame[["right_upward", "left_upward"]].max(axis=1)
    frame["downward"] = frame[["right_downward", "left_downward"]].max(axis=1)
    return frame


# --------------------------------------------------------------------------- #
# Clarified nested-model figure (all four distinct models shown once)
# --------------------------------------------------------------------------- #
def nested_model_redraw(df: pd.DataFrame, outdir: Path) -> None:
    """Redraw of nested_model_comparison.png with one bar per distinct model.

    The original left panel showed the full model twice (once per addition
    order) and omitted the monotonicity-only model, which made it unreadable.
    Length = func_count (z-scored), monotonicity = degree (z-scored).
    """
    from statsmodels.stats.anova import anova_lm

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
    p_addM = anova_lm(L, LM)["Pr(>F)"][1]
    p_addL = anova_lm(M, LM)["Pr(>F)"][1]

    r2 = [base.rsquared, L.rsquared, M.rsquared, LM.rsquared]
    uL, uM = LM.rsquared - M.rsquared, LM.rsquared - L.rsquared
    common = (LM.rsquared - base.rsquared) - uL - uM

    print(
        f"n={int(LM.nobs)}  R^2: base={r2[0]:.3f}  L={r2[1]:.3f}  M={r2[2]:.3f}  LM={r2[3]:.3f}"
    )
    print(
        f"add mono on L: dR2=+{uM:.3f} (p={p_addM:.1e});  add length on M: dR2=+{uL:.3f} (p={p_addL:.1e})"
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    names = [
        "model type\nonly",
        "+ length\n(func_count, L)",
        "+ monotonicity\n(degree, M)",
        "length + mono\n(full, LM)",
    ]
    ax.bar(range(4), r2, color=["#999999", "#4C72B0", "#55A868", "#8172B3"], width=0.62)
    for i, v in enumerate(r2):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.annotate(
        "",
        xy=(2.85, r2[3] + 0.006),
        xytext=(1.15, r2[1] + 0.006),
        arrowprops=dict(
            arrowstyle="->", color="#2e6b40", lw=1.6, connectionstyle="arc3,rad=-0.3"
        ),
    )
    ax.text(
        1.62,
        r2[3] + 0.016,
        f"L → LM: add monotonicity\nΔR² = +{uM:.3f}   (F-test p = {p_addM:.0e})",
        ha="center",
        fontsize=9,
        color="#2e6b40",
    )
    ax.annotate(
        "",
        xy=(3.0, r2[3] - 0.02),
        xytext=(2.35, r2[2] - 0.02),
        arrowprops=dict(
            arrowstyle="->", color="#31517e", lw=1.6, connectionstyle="arc3,rad=0.25"
        ),
    )
    ax.text(
        2.72,
        r2[1] + 0.011,
        f"M → LM: add length\nΔR² = +{uL:.3f}\n(p = {p_addL:.0e})",
        ha="center",
        fontsize=9,
        color="#31517e",
    )
    ax.set_xticks(range(4))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("R²  (Validation Loss AUC)", fontweight="bold")
    ax.set_ylim(0, r2[3] + 0.046)
    ax.set_title(
        "Nested OLS models — length = func_count (z-scored)", fontweight="bold"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax = axes[1]
    parts = [uL, common, uM]
    labels = [
        f"unique to length\n(func_count)\n{uL:.3f}",
        f"shared\n{common:.3f}",
        f"unique to monotonicity\n(degree)\n{uM:.3f}",
    ]
    left = 0
    for p, lab, c in zip(parts, labels, ["#4C72B0", "#C7C7C7", "#55A868"]):
        ax.barh(0, p, left=left, color=c, edgecolor="white", height=0.5)
        ax.text(left + p / 2, 0, lab, ha="center", va="center", fontsize=9.5)
        left += p
    ax.set_xlim(0, (r2[3] - r2[0]) * 1.04)
    ax.set_yticks([])
    ax.set_xlabel(
        f"R² beyond the model-type baseline (total = {r2[3] - r2[0]:.3f})",
        fontweight="bold",
    )
    ax.set_title("Commonality partition of the full model", fontweight="bold")

    fig.tight_layout()
    fig.savefig(outdir / "nested_model_redraw.png", dpi=200)
    plt.close(fig)
    print(f"  saved {rel(outdir / 'nested_model_redraw.png')}")


# --------------------------------------------------------------------------- #
# Clarified directional panel B (the 2x2 high/low split)
# --------------------------------------------------------------------------- #
def directional_panelB_redraw(df: pd.DataFrame, outdir: Path) -> None:
    """Readable version of directional_monotonicity.png panel B.

    Downward status (the effect that matters) goes on the x-axis; labels are in
    words; cell means and cell sizes are printed on the bars.
    """
    tr = _add_updown(df[df["training"] == True]).dropna(  # noqa: E712
        subset=["val_loss_step_AOC"]
    )
    up_hi = tr["upward"] > 0.5
    down_hi = tr["downward"] > 0.5
    cell = lambda u, dn: tr.loc[
        (up_hi == u) & (down_hi == dn), "val_loss_step_AOC"
    ]  # noqa: E731
    means = {(u, dn): cell(u, dn).mean() for u in (True, False) for dn in (True, False)}
    ns = {(u, dn): len(cell(u, dn)) for u in (True, False) for dn in (True, False)}
    for k in means:
        print(
            f"  upward>{0.5}={str(k[0]):5s} downward>0.5={str(k[1]):5s}  mean AUC={means[k]:7.0f}  n={ns[k]}"
        )

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    x = np.array([0, 1])  # downward high / low
    w = 0.34
    v_lo_up = [means[(False, True)], means[(False, False)]]
    v_hi_up = [means[(True, True)], means[(True, False)]]
    n_lo_up = [ns[(False, True)], ns[(False, False)]]
    n_hi_up = [ns[(True, True)], ns[(True, False)]]
    b1 = ax.bar(x - w / 2, v_lo_up, w, color="#4C72B0", label="upward degree ≤ 0.5")
    b2 = ax.bar(x + w / 2, v_hi_up, w, color="#9ecae1", label="upward degree > 0.5")
    for bars, nn in [(b1, n_lo_up), (b2, n_hi_up)]:
        for rect, n in zip(bars, nn):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 40,
                f"{rect.get_height():,.0f}",
                ha="center",
                fontsize=11,
                fontweight="bold",
            )
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() / 2,
                f"n = {n:,}",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "DOWNWARD-monotone runs\n(downward degree > 0.5)",
            "not downward-monotone\n(downward degree ≤ 0.5)",
        ],
        fontsize=11,
    )
    ax.set_ylabel(
        "Mean Validation Loss AUC   (lower = easier to learn)", fontweight="bold"
    )
    ax.legend(title="Upward monotonicity", fontsize=10, title_fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.set_title(
        "Downward monotonicity drives ease of learning; upward adds little",
        fontweight="bold",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(outdir / "directional_panelB_redraw.png", dpi=200)
    plt.close(fig)
    print(f"  saved {rel(outdir / 'directional_panelB_redraw.png')}")


# --------------------------------------------------------------------------- #
# Directional diagnostics (distributions, categories, per-architecture)
# --------------------------------------------------------------------------- #
def directional_diagnostics(df: pd.DataFrame) -> None:
    """Distribution of directional degrees + per-architecture asymmetry check."""
    u = _add_updown(df.drop_duplicates("expression").dropna(subset=DIRS + ["degree"]))
    print(f"per-expression directional degrees (n={len(u)} unique expressions):")
    print(
        f"{'degree':<16}{'mean':>7}{'sd':>7}{'median':>8}{'% == 0':>8}{'% >= .99':>10}"
    )
    for c in DIRS + ["upward", "downward", "degree"]:
        print(
            f"{c:<16}{u[c].mean():>7.3f}{u[c].std():>7.3f}{u[c].median():>8.3f}"
            f"{(u[c] <= 0.001).mean() * 100:>7.1f}%{(u[c] >= 0.99).mean() * 100:>9.1f}%"
        )
    print(
        f"\ncorr(upward, downward) = {stats.pearsonr(u['upward'], u['downward'])[0]:+.3f}"
    )
    print(
        f"degree attained by downward sense: {(u['downward'] >= u['upward']).mean() * 100:.1f}% of expressions"
    )

    tr = _add_updown(df[df["training"] == True]).dropna(  # noqa: E712
        subset=["val_loss_step_AOC", "func_count"]
    )
    print("\nAUC correlations by architecture:")
    for m, g in tr.groupby("model"):
        ru = stats.pearsonr(g["upward"], g["val_loss_step_AOC"])[0]
        rd = stats.pearsonr(g["downward"], g["val_loss_step_AOC"])[0]
        print(f"  {m:<12} n={len(g):5d}  r(upward)={ru:+.3f}  r(downward)={rd:+.3f}")
    mi = smf.ols(
        "val_loss_step_AOC ~ (upward + downward) * C(model) + func_count", data=tr
    ).fit()
    for t in [
        "upward",
        "downward",
        "upward:C(model)[T.Transformer]",
        "downward:C(model)[T.Transformer]",
    ]:
        print(f"  {t:<38} beta={mi.params[t]:+8.0f}  p={mi.pvalues[t]:.2e}")
    print(
        "=> the downward advantage holds in both architectures and is larger for the Transformer."
    )


# --------------------------------------------------------------------------- #
# Operator prevalence
# --------------------------------------------------------------------------- #
def operator_prevalence(df: pd.DataFrame) -> None:
    """%% of expressions attested + cumulative counts, per operator."""
    uniq = df.drop_duplicates("expression")
    tr = df[df["training"] == True]  # noqa: E712
    print(
        f"{'operator':<14}{'% expr attested':>16}{'total (uniq expr)':>18}"
        f"{'mean/expr':>11}{'total (trained runs)':>21}"
    )
    for op in OPERATORS:
        cu = uniq[f"op_{op}"]
        ct = tr[f"op_{op}"]
        print(
            f"{op:<14}{(cu > 0).mean() * 100:>15.1f}%{cu.sum():>18d}"
            f"{cu.mean():>11.2f}{ct.sum():>21d}"
        )


# --------------------------------------------------------------------------- #
# Polarity counterbalance: is the downward surplus driven by `difference`?
# --------------------------------------------------------------------------- #
def polarity_counterbalance(df: pd.DataFrame) -> None:
    """Test the 'difference lacks an inverse' hypothesis; show the `not` gradient."""
    u = _add_updown(df.drop_duplicates("expression").dropna(subset=DIRS))
    gap = u["downward"] - u["upward"]
    print(f"{'op':<14}{'r(up)':>8}{'r(down)':>9}{'r(down-up gap)':>15}")
    for op in OPERATORS:
        c = u[f"op_{op}"]
        print(
            f"{op:<14}{stats.pearsonr(c, u['upward'])[0]:>+8.3f}"
            f"{stats.pearsonr(c, u['downward'])[0]:>+9.3f}"
            f"{stats.pearsonr(c, gap)[0]:>+15.3f}"
        )
    print(f"\noverall mean gap (down-up): {gap.mean():.3f}")
    m0 = u["op_difference"] == 0
    print(
        f"gap among difference==0 (n={m0.sum()}): {gap[m0].mean():.3f}"
        "   -> difference does NOT drive the surplus"
    )
    print("\nmean degrees by `not` count (negation is the polarity flipper):")
    print(
        u.groupby(u["op_not"].clip(upper=2))[["upward", "downward"]]
        .mean()
        .round(3)
        .to_string()
    )


# --------------------------------------------------------------------------- #
# Per-operator difficulty by architecture (absolute, difference, RESCALED)
# --------------------------------------------------------------------------- #
def operator_by_architecture(df: pd.DataFrame, outdir: Path, n_boot: int = 500) -> None:
    """Fit the operator Ridge separately per architecture; three panels.

    Panel (a): coefficients in absolute AUC units (shared standardization of X,
    bootstrap 95%% CIs). Panel (b): Transformer - LSTM difference. Panel (c):
    the same coefficients RESCALED by each architecture's own AUC standard
    deviation. The Transformer accumulates more loss overall (larger AUC mean
    and SD), which inflates its absolute coefficients; dividing by the
    within-architecture SD removes that scale so panel (c) compares the
    difficulty *profiles* directly.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    tr = (
        df[df["training"] == True].dropna(subset=["val_loss_step_AOC"]).copy()
    )  # noqa: E712
    opcols = [f"op_{o}" for o in OPERATORS]
    X_all = tr[opcols].to_numpy(float)
    Xm, Xs = X_all.mean(0), X_all.std(0)
    Xs[Xs == 0] = 1.0
    alpha = 10.0
    rng = np.random.default_rng(0)

    res = {}
    for arch, g in tr.groupby("model"):
        X = (g[opcols].to_numpy(float) - Xm) / Xs
        y = g["val_loss_step_AOC"].to_numpy(float)
        base = Ridge(alpha=alpha).fit(X, y)
        boot = np.zeros((n_boot, len(opcols)))
        for b in range(n_boot):
            idx = rng.integers(0, len(y), len(y))
            boot[b] = Ridge(alpha=alpha).fit(X[idx], y[idx]).coef_
        res[arch] = dict(
            coef=base.coef_,
            boot=boot,
            n=len(y),
            r2=r2_score(y, base.predict(X)),
            ymean=y.mean(),
            ysd=y.std(),
        )

    L, T = res["LSTM"], res["Transformer"]
    diff = T["boot"] - L["boot"]
    dlo, dhi = np.percentile(diff, [2.5, 97.5], axis=0)
    prof_r = np.corrcoef(L["coef"], T["coef"])[0, 1]

    print(
        f"LSTM        n={L['n']}  mean AUC={L['ymean']:7.0f}  SD={L['ysd']:7.0f}  R^2={L['r2']:.3f}"
    )
    print(
        f"Transformer n={T['n']}  mean AUC={T['ymean']:7.0f}  SD={T['ysd']:7.0f}  R^2={T['r2']:.3f}"
    )
    print(f"coefficient-profile correlation: r = {prof_r:.3f}")
    print(
        f"\n{'operator':<14}{'LSTM':>8}{'Tfmr':>8}{'diff':>8}{'diff 95% CI':>19}{'sig':>4}"
        f"{'LSTM/SD':>9}{'Tfmr/SD':>9}"
    )
    order = np.argsort(T["coef"])
    for i in order:
        sig = "*" if (dlo[i] > 0 or dhi[i] < 0) else ""
        print(
            f"{OPERATORS[i]:<14}{L['coef'][i]:>8.0f}{T['coef'][i]:>8.0f}"
            f"{T['coef'][i] - L['coef'][i]:>8.0f}   [{dlo[i]:>6.0f},{dhi[i]:>6.0f}]{sig:>4}"
            f"{L['coef'][i] / L['ysd']:>9.3f}{T['coef'][i] / T['ysd']:>9.3f}"
        )

    # --- figure -------------------------------------------------------------
    y_pos = np.arange(len(OPERATORS))
    off = 0.18
    fig, axes = plt.subplots(
        1, 3, figsize=(17, 6), gridspec_kw={"width_ratios": [1.15, 0.9, 1.0]}
    )

    ax = axes[0]
    for k, (r, col, lab) in enumerate(
        [(L, "#E69F00", "LSTM"), (T, "#56B4E9", "Transformer")]
    ):
        c = np.array(r["coef"])[order]
        lo, hi = np.percentile(r["boot"], [2.5, 97.5], axis=0)
        lo, hi = lo[order], hi[order]
        yy = y_pos + (off if k else -off)
        ax.errorbar(
            c,
            yy,
            xerr=[c - lo, hi - c],
            fmt="o",
            color=col,
            ecolor=col,
            capsize=3,
            ms=6,
            lw=1.4,
            label=lab,
        )
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([OPERATORS[i] for i in order])
    ax.set_xlabel(
        "Δ AUC per +1 SD of operator count\n(absolute units — inflated by the\n"
        "Transformer's larger overall loss)",
        fontweight="bold",
    )
    ax.set_title("(a) Absolute coefficients", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", ls="--", alpha=0.4)

    ax = axes[1]
    c = (T["coef"] - L["coef"])[order]
    lo, hi = dlo[order], dhi[order]
    sig = (lo > 0) | (hi < 0)
    cols = np.where(sig, np.where(c > 0, "#c0392b", "#2471a3"), "#9aa0a6")
    ax.errorbar(
        c, y_pos, xerr=[c - lo, hi - c], fmt="none", ecolor="gray", capsize=3, zorder=1
    )
    ax.scatter(c, y_pos, color=cols, s=55, zorder=2)
    for ci, yi, s in zip(c, y_pos, sig):
        if s:
            ax.text(
                ci, yi + 0.32, f"{ci:+.0f}", ha="center", fontsize=9, fontweight="bold"
            )
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel(
        "Transformer − LSTM (absolute)\ncolour = 95% CI excludes 0", fontweight="bold"
    )
    ax.set_title("(b) Difference", fontweight="bold")
    ax.grid(True, axis="x", ls="--", alpha=0.4)

    ax = axes[2]
    for k, (r, col, lab) in enumerate(
        [(L, "#E69F00", "LSTM"), (T, "#56B4E9", "Transformer")]
    ):
        c = (np.array(r["coef"]) / r["ysd"])[order]
        lo, hi = np.percentile(r["boot"] / r["ysd"], [2.5, 97.5], axis=0)
        lo, hi = lo[order], hi[order]
        yy = y_pos + (off if k else -off)
        ax.errorbar(
            c,
            yy,
            xerr=[c - lo, hi - c],
            fmt="o",
            color=col,
            ecolor=col,
            capsize=3,
            ms=6,
            lw=1.4,
            label=lab,
        )
    ax.axvline(0, color="black", ls="--", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel(
        "Δ AUC in WITHIN-architecture SD units\n(coefficient ÷ that architecture's AUC SD:\n"
        "removes the overall-scale difference)",
        fontweight="bold",
    )
    ax.set_title("(c) Rescaled — profiles nearly coincide", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", ls="--", alpha=0.4)

    fig.suptitle(
        f"Per-operator difficulty by architecture — same profile (r = {prof_r:.3f}), "
        f"amplified by the Transformer's larger loss scale "
        f"(AUC SD {T['ysd']:.0f} vs {L['ysd']:.0f})",
        fontweight="bold",
        y=1.00,
    )
    fig.tight_layout()
    fig.savefig(outdir / "operator_by_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {rel(outdir / 'operator_by_architecture.png')}")


# --------------------------------------------------------------------------- #
# Script-generated versions of the two formerly ad-hoc figures (2026-07-22)
# --------------------------------------------------------------------------- #
def paper_figure1_redl(df: pd.DataFrame, outdir: Path) -> None:
    """Variant of paper_figure1 with the fit line drawn across the full range."""
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
        print("  [skip] plotnine not installed")
        return
    plotdf = df.dropna(subset=["val_loss_step_AOC", "degree"]).copy()
    plot = (
        ggplot(plotdf, aes(x="val_loss_step_AOC", y="degree", color="model"))
        + geom_jitter(alpha=1, width=10, height=0.01, size=2)
        + stat_smooth(
            method="lm",
            color="red",
            linetype="dashed",
            se=False,
            size=1.2,
            fullrange=True,
        )
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
    plot.save(outdir / "paper_figure1_redl.png", dpi=600, verbose=False)
    print(f"  saved {rel(outdir / 'paper_figure1_redl.png')}  ({len(plotdf)} points)")


def figure_vs_mixedmodel(df: pd.DataFrame, outdir: Path) -> None:
    """paper_figure1 scatter with mixed-effects fixed-effect fits overlaid.

    The two solid lines are the mixed model's fixed effects (AUC as a function
    of monotonicity, per architecture; random intercept per expression) drawn
    on the figure's axes (x = predicted AUC, y = degree). The black dashed line
    is the figure's own OLS of degree on AUC. Their different slopes illustrate
    that the figure's r and the regression's beta answer different questions.
    """
    d = (
        df[df["training"] == True]
        .dropna(subset=["val_loss_step_AOC", "degree"])  # noqa: E712
        .copy()
    )
    m = smf.mixedlm(
        "val_loss_step_AOC ~ degree * C(model)",
        d,
        groups=d["expression"],
        re_formula="~1",
    ).fit(reml=False)
    b = m.params
    grid = np.linspace(0, 1, 50)
    pred = {
        "LSTM": b["Intercept"] + b["degree"] * grid,
        "Transformer": (
            b["Intercept"]
            + b["C(model)[T.Transformer]"]
            + (b["degree"] + b["degree:C(model)[T.Transformer]"]) * grid
        ),
    }
    print("mixed model AUC ~ degree * model + (1|expression):")
    for t in ["degree", "C(model)[T.Transformer]", "degree:C(model)[T.Transformer]"]:
        print(f"  {t:<34} beta={b[t]:+9.1f}  p={m.pvalues[t]:.2e}")

    x = d["val_loss_step_AOC"].to_numpy(float)
    y = d["degree"].to_numpy(float)
    r = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {"LSTM": "#F8766D", "Transformer": "#00BFC4"}
    for name, sub in d.groupby("model"):
        ax.scatter(
            sub["val_loss_step_AOC"],
            sub["degree"],
            s=8,
            alpha=0.35,
            color=colors[str(name)],
            edgecolors="none",
            label=str(name),
        )
    for name, line_col in [("LSTM", "#b03a2e"), ("Transformer", "#0e6655")]:
        ax.plot(
            pred[name], grid, color=line_col, lw=3, label=f"{name}  mixed-model fit"
        )
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        xs,
        slope * xs + intercept,
        "k--",
        lw=2,
        label=f"figure OLS (degree~AUC), r={r:.3f}",
    )
    ax.set_xlabel("Validation Loss AUC", fontsize=13)
    ax.set_ylabel("Monotonicity (degree)", fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"paper_figure1.png scatter + mixed-effects model fits (same {len(d)} rows)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "figure_vs_mixedmodel.png", dpi=150)
    plt.close(fig)
    print(f"  saved {rel(outdir / 'figure_vs_mixedmodel.png')}")


# --------------------------------------------------------------------------- #
# Why are low-but-nonzero monotonicity degrees rare? (the "gap" in the scatter)
# --------------------------------------------------------------------------- #
def monotonicity_gap_analysis(df: pd.DataFrame, outdir: Path) -> None:
    """Explain the sparse region just above degree 0 in paper_figure1.

    OBSERVATION. Per unique expression, `degree` has a point mass at exactly 0
    (~9%%) and plenty of mass above ~0.2, but almost nothing in (0, 0.15].

    APPROACH -- three tests:

    (1) REAL-DATA DECOMPOSITION. Tabulate the distribution of `degree` and of
        its four directional components. If the components have more low-range
        mass than `degree` does, part of the gap comes from taking the max of
        four values (all four must be simultaneously low for the max to be low).

    (2) SIMULATION on a small lattice (all 2^8 = 256 subsets of an 8-element
        base, uniform distribution; upward sense only). We compute the same
        measure -- 1 - H(Q|pred)/H(Q), pred(x) = "some subset of x satisfies Q"
        -- for two kinds of artificial quantifiers:
          (a) NOISY: an upward-closed truth set with each model's value flipped
              independently with probability eta. If these fill (0, 0.15], the
              measure itself CAN produce low values, and the gap must come from
              the population of quantifiers, not the measure.
          (b) CRISP families mimicking grammar output: thresholds |x & S| >= k
              (upward monotone), negated thresholds (downward monotone),
              equalities |x & S| == k (true only at an exact value -- the truth
              set is a thin shell), and two-clause boolean combinations.
        Prediction if the population explanation is right: crisp families land
        at exactly 0, exactly 1, or >= ~0.2 -- except equalities, the only
        crisp family whose truth sets align only weakly with the subset order.

    (3) ENRICHMENT CHECK on the real gap expressions: does the proportion of
        gap expressions containing an operator exceed the proportion among
        non-gap expressions (their ratio = "enrichment")? Prediction:
        `equals` and `not` over-represented.

    CAVEATS. The simulation lattice is a simplification of the real model
    space (referents assigned to A/B/M zones); the families are proxies for
    grammar output; the enrichment test has only n~34 gap expressions.
    """
    # ---- (1) real-data decomposition ------------------------------------- #
    u = df.drop_duplicates("expression").dropna(subset=DIRS + ["degree"])
    bins = [0.0001, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.90, 0.9899, 1.01]
    labels = [
        "(0,.05]",
        "(.05,.10]",
        "(.10,.15]",
        "(.15,.20]",
        "(.20,.30]",
        "(.30,.50]",
        "(.50,.70]",
        "(.70,.90]",
        "(.90,.99)",
        "[.99,1]",
    ]
    cols = ["degree"] + DIRS
    print(f"(1) distribution over {len(u)} unique expressions:")
    print(f"{'range':<12}" + "".join(f"{c[:9]:>11}" for c in cols))
    print(f"{'== 0':<12}" + "".join(f"{(u[c] <= 0.0001).sum():>11d}" for c in cols))
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        print(
            f"{lab:<12}"
            + "".join(f"{((u[c] > lo) & (u[c] <= hi)).sum():>11d}" for c in cols)
        )
    in_gap = (u["degree"] > 0.0001) & (u["degree"] <= 0.15)
    per_dir_gap = np.mean([(((u[d] > 0.0001) & (u[d] <= 0.15)).mean()) for d in DIRS])
    print(
        f"\n  per-direction share in (0,.15]: ~{per_dir_gap * 100:.1f}%  "
        f"but degree (max of 4): {in_gap.mean() * 100:.1f}%  <- max-of-four squeeze"
    )

    # ---- (2) simulation --------------------------------------------------- #
    n_bits, n_models = 8, 256
    xs = np.arange(n_models)

    def up_closure(q):
        f = q.copy()
        for i in range(n_bits):
            hi_idx = np.flatnonzero((xs >> i) & 1)
            f[hi_idx] |= f[hi_idx ^ (1 << i)]
        return f

    def bin_ent(p):
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    def mon_up(q):
        if q.all() or not q.any():
            return 1.0
        pred = up_closure(q)
        cond = 0.0
        for v in (0, 1):
            m = pred == v
            if m.sum():
                pm = q[m].mean()
                if 0 < pm < 1:
                    cond += m.mean() * bin_ent(pm)
        return 1 - cond / bin_ent(q.mean())

    rng = np.random.default_rng(0)
    card = np.zeros((n_bits, n_models), dtype=int)
    for i in range(n_bits):
        card[i] = (xs >> i) & 1

    fams: dict[str, list[float]] = {
        "noisy": [],
        "threshold": [],
        "negated_thr": [],
        "equality": [],
        "bool2": [],
    }
    for _ in range(1500):
        # noisy: upward-closed seed + independent flips
        gens = rng.integers(0, n_models, rng.integers(1, 5))
        seed = np.zeros(n_models, bool)
        seed[gens] = True
        q = up_closure(seed) ^ (
            rng.random(n_models) < rng.choice([0.05, 0.1, 0.2, 0.35, 0.5])
        )
        if 0.02 < q.mean() < 0.98:
            fams["noisy"].append(mon_up(q))
        # crisp families over |x & S| for random S
        s_mask = rng.random(n_bits) < 0.5
        if s_mask.sum() == 0:
            continue
        c1 = card[s_mask].sum(0)
        k = rng.integers(1, s_mask.sum() + 1)
        for name, q in [
            ("threshold", c1 >= k),
            ("negated_thr", ~(c1 >= k)),
            ("equality", c1 == k),
        ]:
            if 0.02 < q.mean() < 0.98:
                fams[name].append(mon_up(q))
        s2 = rng.random(n_bits) < 0.5
        if s2.sum():
            c2 = card[s2].sum(0)
            k2 = rng.integers(1, s2.sum() + 1)
            q = rng.choice([np.logical_and, np.logical_or, np.logical_xor])(
                c1 >= k, ~(c2 >= k2)
            )
            if 0.02 < q.mean() < 0.98:
                fams["bool2"].append(mon_up(q))

    cats = [
        ("== 0", lambda v: v <= 1e-4),
        ("(0, .15]", lambda v: (v > 1e-4) & (v <= 0.15)),
        ("(.15, .99)", lambda v: (v > 0.15) & (v < 0.99)),
        ("[.99, 1]", lambda v: v >= 0.99),
    ]
    print(
        f"\n(2) simulated upward-sense degrees (256-model lattice), share per family:"
    )
    print(f"{'':12s}" + "".join(f"{f:>13}" for f in fams))
    fam_shares = {}
    for lab, cond in cats:
        row = []
        for f, vals in fams.items():
            v = np.array(vals)
            row.append(cond(v).mean() if len(v) else np.nan)
        fam_shares[lab] = row
        print(f"{lab:<12}" + "".join(f"{r * 100:>12.0f}%" for r in row))
    print(
        "  -> noisy quantifiers DO fill (0,.15]; among crisp families only "
        "equalities do.\n     The measure can produce low values; the grammar's "
        "crisp quantifiers mostly cannot."
    )

    # ---- (3) enrichment among real gap expressions ------------------------ #
    gap_expr = u[in_gap]
    rest = u[u["degree"] > 0.15]
    print(
        f"\n(3) operator enrichment: gap (n={len(gap_expr)}) vs degree>0.15 (n={len(rest)}):"
    )
    print(f"{'op':<14}{'% of gap exprs':>15}{'% of others':>13}{'ratio':>8}")
    for op in OPERATORS:
        pat = re.compile(rf"\b{op}\(")
        g = gap_expr["expression"].apply(lambda e: bool(pat.search(e))).mean()
        r = rest["expression"].apply(lambda e: bool(pat.search(e))).mean()
        print(f"{op:<14}{g * 100:>14.0f}%{r * 100:>12.0f}%{g / max(r, 1e-9):>8.1f}x")
    print("\nexample gap expressions (smallest degrees):")
    for _, row in gap_expr.nsmallest(5, "degree").iterrows():
        print(f"  {row['degree']:.4f}  {row['expression'][:80]}")

    # ---- figure ------------------------------------------------------------ #
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    ax = axes[0]
    deg = u["degree"].to_numpy(float)
    edges = np.arange(0.025, 1.026, 0.025)
    counts, _ = np.histogram(deg[deg > 1e-4], edges)
    ax.bar(
        edges[:-1] + 0.0125, counts, width=0.023, color="#4C72B0", label="degree > 0"
    )
    ax.bar(
        [0.0], [(deg <= 1e-4).sum()], width=0.023, color="#1a1a2e", label="exactly 0"
    )
    ax.axvspan(0.001, 0.15, color="#c0392b", alpha=0.12)
    ax.text(
        0.075,
        ax.get_ylim()[1] * 0.05 + max(counts) * 0.9,
        "the gap:\n34 exprs",
        ha="center",
        fontsize=10,
        color="#c0392b",
        fontweight="bold",
    )
    ax.set_xlabel("monotonicity degree (per unique expression)", fontweight="bold")
    ax.set_ylabel("number of expressions", fontweight="bold")
    ax.set_title(
        "(a) Real data: mass at 0, a sparse zone to ~0.15, then a continuum",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend()
    ax.grid(True, axis="y", ls="--", alpha=0.4)

    ax = axes[1]
    fam_names = list(fams)
    x = np.arange(len(fam_names))
    w = 0.2
    colors = {
        "== 0": "#1a1a2e",
        "(0, .15]": "#c0392b",
        "(.15, .99)": "#4C72B0",
        "[.99, 1]": "#55A868",
    }
    for j, (lab, _) in enumerate(cats):
        ax.bar(
            x + (j - 1.5) * w,
            np.array(fam_shares[lab]) * 100,
            w,
            color=colors[lab],
            label=lab,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(fam_names, fontsize=9)
    ax.set_ylabel("% of simulated quantifiers", fontweight="bold")
    ax.set_title(
        "(b) Simulation: only noise and equality-type\nquantifiers land in (0, .15]",
        fontweight="bold",
        fontsize=11,
    )
    ax.legend(title="degree lands in", fontsize=8, title_fontsize=8)
    ax.grid(True, axis="y", ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(outdir / "monotonicity_gap_analysis.png", dpi=200)
    plt.close(fig)
    print(f"\n  saved {rel(outdir / 'monotonicity_gap_analysis.png')}")


def main() -> None:
    outdir = repo_root() / "figures"
    outdir.mkdir(exist_ok=True)
    tables_dir = repo_root() / "analysis" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = load()
    print(
        f"Loaded {len(df)} rows; {int((df['training'] == True).sum())} trained."
    )  # noqa: E712

    sections = [
        ("9a_nested_model_redraw", nested_model_redraw, True),
        ("9b_directional_panelB_redraw", directional_panelB_redraw, True),
        ("9c_directional_diagnostics", directional_diagnostics, False),
        ("9d_operator_prevalence", operator_prevalence, False),
        ("9e_polarity_counterbalance", polarity_counterbalance, False),
        ("9f_operator_by_architecture", operator_by_architecture, True),
        ("9g_paper_figure1_redl", paper_figure1_redl, True),
        ("9h_figure_vs_mixedmodel", figure_vs_mixedmodel, True),
        ("9i_monotonicity_gap", monotonicity_gap_analysis, True),
    ]
    from contextlib import redirect_stdout

    for name, fn, wants_outdir in sections:
        path = tables_dir / f"{name}.txt"
        print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
        with open(path, "w") as f:
            with redirect_stdout(_Tee(sys.stdout, f)):
                fn(df, outdir) if wants_outdir else fn(df)
        print(f"  table saved to {rel(path)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
