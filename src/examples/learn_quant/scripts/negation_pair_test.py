"""Q2-T1: the negation-pair (complement-pair) test.

What is run here
----------------
This script does NOT train a neural network. The words "model" and "run" occur
in three distinct senses in this project:

1. Neural learners (reused, not run here): an MV_LSTM and a Transformer were
   trained previously for each expression. Their recorded validation-loss AUCs
   are loaded from ``outputs/combined_runs_AOC_monotonicity_updated.csv``.
2. QuantifierModel (evaluated here): despite its name, this is one set-theoretic
   scene <M, A, B>, not a learned model. Each grammar expression is executed on
   sampled scenes to confirm that a candidate pair always returns opposite
   truth values.
3. Statistical model (fit here): an OLS regression of mean AUC on upward and
   downward degree supplies the sample-wide gap that the paired result is
   compared against.

See ``notebooks/negation_pair_test_walkthrough.ipynb`` for an executable,
cell-by-cell version with intermediate tables.

Motivation (HANDOFF.md section 4): training is sigmoid + BCE, which is symmetric
under label complement, and the complement of an up-set is a down-set. So pure
decision-boundary geometry cannot produce the observed upward/downward learning
asymmetry. This script finds pairs of *trained* expressions whose truth vectors
are exact complements of each other and compares their learning difficulty
(validation-loss AUC). Within such a pair the decision boundary is identical
and the monotonicity direction flips, so:

  * AUC ~ equal within pairs  -> label symmetry holds empirically; the
    directional asymmetry must be sample-composition or measure-side (H1/H4/H5).
  * AUC systematically different -> the input encoding or example-sampling
    procedure breaks the symmetry (H2/H3) -- a new finding about the learner.

Procedure:
  1. Load the archived expression pool (9,550 unique meanings) and the M4/X4/d5
     universe (256 models) from the original `altk` run archive; compute each
     pool expression's 256-bit truth vector.
  2. Find complement pairs among the 2,000 trained expressions
     (`expressions_sample_2k.csv`).
  3. Verify each candidate pair is a *functional* complement on the actual
     training-model distribution (generate_batch with M_size=12, X_size=16,
     inclusive=False -- the settings in conf/learn.yaml's generation_args),
     not just on the 256-model universe.
  4. Join per-expression AUC (trained runs of the combined CSV) and the four
     directional degrees; run within-pair comparisons.

Also reported: the measure-mirror check (H4). For a perfect complement pair
(e, ~e), any well-behaved directional measure should satisfy
up(e) = down(~e) and down(e) = up(~e); deviations diagnose the measure.

Run (needs the conda env `altk` and the original run archive):
    python scripts/negation_pair_test.py
    python scripts/negation_pair_test.py --altk-archive /path/to/altk/src/examples

Outputs:
    analysis/tables/11_negation_pair_test.txt
    figures/negation_pair_test.png
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

DEFAULT_ALTK_ARCHIVE = Path.home() / "Documents/UWLing/altk/src/examples"
POOL_REL = "learn_quant/outputs/M4/X4/d5"
# training-data generation settings from conf/learn.yaml (generation_args)
GEN_M_SIZE, GEN_X_SIZE, GEN_INCLUSIVE = 12, 16, False

DIRS = ["right_upward", "left_upward", "right_downward", "left_downward"]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


# --------------------------------------------------------------------------- #
# Stage 1-2: archive loading and complement pairing
# --------------------------------------------------------------------------- #
def load_archive(archive: Path):
    """Load the original run archive's expression pool and universe.

    Returns (terms, V, universe, expressions_by_index) where V is the
    (n_pool, 256) boolean truth matrix and expressions_by_index the parallel
    list of GrammaticalExpression objects (needed for functional evaluation).
    """
    sys.path.insert(0, str(archive))
    from ultk.util.frozendict import FrozenDict

    # the archived FrozenDict blocks __setitem__, which pickle's BUILD needs
    FrozenDict.__setitem__ = dict.__setitem__
    import dill as pkl

    base = archive / POOL_REL
    uni = pkl.load(open(base / "master_universe.pkl", "rb"))
    pool = pkl.load(open(base / "generated_expressions_xidx.pkl", "rb"))
    refs = uni.referents

    terms, vecs, exprs = [], [], []
    for _meaning, expr in pool.items():
        v = np.fromiter(
            (expr.meaning.mapping[r] for r in refs), dtype=bool, count=len(refs)
        )
        terms.append(expr.term_expression)
        vecs.append(v)
        exprs.append(expr)
    V = np.array(vecs)
    print(f"archive pool: {len(terms)} unique meanings on {len(refs)} models")
    return terms, V, uni, exprs


def find_pairs(terms, V, sample_terms):
    """Complement pairs among the trained sample, plus coverage stats."""
    key = {v.tobytes(): i for i, v in enumerate(V)}
    t2i = {t: i for i, t in enumerate(terms)}
    sidx = [t2i[t] for t in sample_terms]
    sset = set(sidx)

    pool_comp = sum(1 for v in V if (~v).tobytes() in key)
    samp_comp_pool = sum(1 for i in sidx if (~V[i]).tobytes() in key)

    pairs, seen = [], set()
    for i in sidx:
        j = key.get((~V[i]).tobytes())
        if j is not None and j in sset:
            tag = (min(i, j), max(i, j))
            if tag not in seen:
                seen.add(tag)
                pairs.append(tag)

    print(f"pool meanings with complement in pool:        {pool_comp}")
    print(f"trained expressions with complement in pool:  {samp_comp_pool} / 2000")
    print(f"complement pairs within the trained 2000:     {len(pairs)}")
    return pairs, sidx


# --------------------------------------------------------------------------- #
# Stage 3: functional verification on the training-model distribution
# --------------------------------------------------------------------------- #
def verify_pairs(pairs, exprs, archive: Path, n_scenes: int, seed: int = 7):
    """Check q1 == ~q2 on scenes drawn like the training data (M=12, X=16).

    ``QuantifierModel`` is the project's name for a set-theoretic scene. This
    function executes symbolic expressions; it does not invoke a neural model.
    """
    from learn_quant.sampling import generate_batch  # from the archive path
    from learn_quant.quantifier import QuantifierModel

    rng_state = np.random.get_state()
    np.random.seed(seed)
    arrays = generate_batch(GEN_M_SIZE, GEN_X_SIZE, n_scenes, inclusive=GEN_INCLUSIVE)
    np.random.set_state(rng_state)
    scenes = [QuantifierModel(a) for a in arrays]

    cache: dict[int, np.ndarray] = {}

    def evaluate(i):
        if i not in cache:
            e = exprs[i]
            cache[i] = np.fromiter(
                (bool(e(scene)) for scene in scenes), dtype=bool, count=len(scenes)
            )
        return cache[i]

    verified, failed = [], []
    for i, j in pairs:
        qi, qj = evaluate(i), evaluate(j)
        n_agree = int((qi == ~qj).sum())
        if n_agree == len(scenes):
            verified.append((i, j))
        else:
            failed.append((i, j, len(scenes) - n_agree))
    print(
        f"functional verification on {n_scenes} training-style scenes "
        f"(M={GEN_M_SIZE}, X={GEN_X_SIZE}): {len(verified)} verified, "
        f"{len(failed)} failed"
    )
    for i, j, bad in failed:
        print(
            f"  FAILED ({bad} disagreements): {exprs[i].term_expression[:70]}"
            f"  vs  {exprs[j].term_expression[:70]}"
        )
    return verified, failed


# --------------------------------------------------------------------------- #
# Stage 4: AUC join and within-pair analysis
# --------------------------------------------------------------------------- #
def load_runs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["expression"].notna() & (df["training"] == True)].copy()  # noqa: E712
    return df


def print_model_provenance(runs: pd.DataFrame, csv_path: Path, n_scenes: int) -> None:
    """State exactly which computations are reused and which run in this script."""
    architectures = ", ".join(sorted(runs["model"].dropna().unique()))
    print("--- what is (and is not) run in this test ---")
    print("NEW NEURAL TRAINING: none")
    print(
        f"REUSED NEURAL RESULTS: {len(runs)} completed rows for {architectures}, "
        f"loaded from {rel(csv_path)}"
    )
    print(
        "  LSTM: MV_LSTM, 3 stacked layers, 20 hidden units; "
        "Transformer: 2 encoder layers, d_model=12, 4 heads"
    )
    print(
        "  original training: one-hot scenes of length 16, Adam (lr=0.001), "
        "BCEWithLogitsLoss, up to 50 epochs, 5 splits"
    )
    print(
        f"RECOMPUTED HERE: symbolic expressions evaluated on {n_scenes} newly "
        "sampled set-theoretic scenes to verify exact label complements"
    )
    print(
        "STATISTICAL MODEL FIT HERE: OLS(mean validation-loss AUC ~ "
        "z(downward degree) + z(upward degree))"
    )
    print(
        "IMPORTANT: QuantifierModel below means one <M,A,B> scene; it is not "
        "an LSTM, Transformer, or fitted predictor.\n"
    )


def per_expression_auc(runs: pd.DataFrame) -> pd.DataFrame:
    """Mean AUC per expression x architecture (over trained runs) plus degrees."""
    auc = (
        runs.groupby(["expression", "model"])["val_loss_step_AOC"]
        .mean()
        .unstack("model")
        .rename(columns={"LSTM": "auc_lstm", "Transformer": "auc_tf"})
    )
    deg = runs.groupby("expression")[DIRS + ["degree"]].first()
    out = auc.join(deg)
    out["upward"] = out[["right_upward", "left_upward"]].clip(0, 1).max(axis=1)
    out["downward"] = out[["right_downward", "left_downward"]].clip(0, 1).max(axis=1)
    out["auc_mean"] = out[["auc_lstm", "auc_tf"]].mean(axis=1)
    return out


def build_pair_table(pairs, terms, stats_df: pd.DataFrame) -> pd.DataFrame:
    """Build one transparent row per complement pair with labels and outcomes."""
    rows = []
    for i, j in pairs:
        ti, tj = terms[i], terms[j]
        if ti not in stats_df.index or tj not in stats_df.index:
            continue
        a, b = stats_df.loc[ti], stats_df.loc[tj]
        # D-member: the more downward-dominant one (larger downward - upward)
        if (a["downward"] - a["upward"]) >= (b["downward"] - b["upward"]):
            d, u, td, tu = a, b, ti, tj
        else:
            d, u, td, tu = b, a, tj, ti
        rows.append(
            {
                "term_D": td,
                "term_U": tu,
                "deg_D": d["degree"],
                "deg_U": u["degree"],
                "down_D": d["downward"],
                "up_D": d["upward"],
                "down_U": u["downward"],
                "up_U": u["upward"],
                "polarity_contrast": (d["downward"] - d["upward"])
                - (u["downward"] - u["upward"]),
                "auc_lstm_D": d["auc_lstm"],
                "auc_lstm_U": u["auc_lstm"],
                "auc_tf_D": d["auc_tf"],
                "auc_tf_U": u["auc_tf"],
                "auc_mean_D": d["auc_mean"],
                "auc_mean_U": u["auc_mean"],
                # measure-mirror checks: up(e) should equal down(~e)
                "mirror_up_D_vs_down_U": d["upward"] - u["downward"],
                "mirror_down_D_vs_up_U": d["downward"] - u["upward"],
                "len_D": td.count("("),
                "len_U": tu.count("("),
            }
        )
    return pd.DataFrame(rows)


def analyze_pairs(
    pairs,
    terms,
    stats_df: pd.DataFrame,
    outdir: Path,
    figdir: Path,
    output_tag: str = "",
):
    pairs_df = build_pair_table(pairs, terms, stats_df)

    def paired_report(col_d, col_u, label):
        sub = pairs_df[[col_d, col_u]].dropna()
        delta = sub[col_u] - sub[col_d]  # >0 means U-member is HARDER
        if len(sub) < 3:
            print(f"  {label}: n={len(sub)} (too few)")
            return
        w = stats.wilcoxon(delta)
        t = stats.ttest_rel(sub[col_u], sub[col_d])
        print(
            f"  {label}: n={len(sub)}  mean D={sub[col_d].mean():8.1f}  "
            f"mean U={sub[col_u].mean():8.1f}  mean Δ(U-D)={delta.mean():+8.1f}  "
            f"median Δ={delta.median():+8.1f}  frac(U harder)={np.mean(delta > 0):.2f}"
        )
        print(f"      Wilcoxon p={w.pvalue:.4f}   paired-t p={t.pvalue:.4f}")

    print("\n--- within-pair AUC comparison (D = downward-dominant member) ---")
    print("ALL verified pairs:")
    paired_report("auc_lstm_D", "auc_lstm_U", "LSTM       ")
    paired_report("auc_tf_D", "auc_tf_U", "Transformer")
    paired_report("auc_mean_D", "auc_mean_U", "mean arch  ")

    contrast = pairs_df["polarity_contrast"] > 0.2
    print(
        f"\nPairs with real polarity contrast (Δ(down-up) gap > 0.2): "
        f"n={int(contrast.sum())}"
    )
    sub = pairs_df[contrast]
    if len(sub) >= 3:
        for cd, cu, lab in [
            ("auc_lstm_D", "auc_lstm_U", "LSTM       "),
            ("auc_tf_D", "auc_tf_U", "Transformer"),
            ("auc_mean_D", "auc_mean_U", "mean arch  "),
        ]:
            d = sub[[cd, cu]].dropna()
            delta = d[cu] - d[cd]
            if len(d) >= 3:
                w = stats.wilcoxon(delta)
                print(
                    f"  {lab}: n={len(d)}  mean Δ(U-D)={delta.mean():+8.1f}  "
                    f"median={delta.median():+8.1f}  frac(U harder)={np.mean(delta>0):.2f}  "
                    f"Wilcoxon p={w.pvalue:.4f}"
                )

    # does the within-pair AUC difference track the polarity contrast?
    d = pairs_df[["auc_mean_D", "auc_mean_U", "polarity_contrast"]].dropna()
    if len(d) > 5:
        r, p = stats.pearsonr(d["auc_mean_U"] - d["auc_mean_D"], d["polarity_contrast"])
        print(
            f"\ncorr( ΔAUC(U-D), polarity contrast ):  r={r:+.3f}  p={p:.4f}  n={len(d)}"
        )

    # ------------------------------------------------------------------ #
    # population-model prediction vs within-pair observation
    # ------------------------------------------------------------------ #
    # If the population-level directional effect (AUC ~ downward + upward over
    # all expressions) were a boundary-level property of the learner, it would
    # have to show up within complement pairs too. Predict each pair's
    # ΔAUC(U-D) from the population OLS and compare with what we observe.
    import statsmodels.api as sm

    full = stats_df.dropna(subset=["auc_mean", "downward", "upward"])
    sd_down, sd_up = full["downward"].std(ddof=0), full["upward"].std(ddof=0)
    X = sm.add_constant(
        pd.DataFrame(
            {
                "downward": (full["downward"] - full["downward"].mean()) / sd_down,
                "upward": (full["upward"] - full["upward"].mean()) / sd_up,
            }
        )
    )
    fit = sm.OLS(full["auc_mean"], X).fit()
    b_down, b_up = fit.params["downward"], fit.params["upward"]
    print("\n--- population prediction vs within-pair observation ---")
    print(
        f"population OLS (n={len(full)} expressions): "
        f"AUC ~ downward {b_down:+.1f}/SD  upward {b_up:+.1f}/SD"
    )
    pred = (
        b_down * (pairs_df["down_U"] - pairs_df["down_D"]) / sd_down
        + b_up * (pairs_df["up_U"] - pairs_df["up_D"]) / sd_up
    )
    obs = pairs_df["auc_mean_U"] - pairs_df["auc_mean_D"]
    ok = pred.notna() & obs.notna()
    pred, obs = pred[ok], obs[ok]
    se = obs.std(ddof=1) / np.sqrt(len(obs))
    print(
        f"within pairs (n={len(obs)}): predicted mean ΔAUC(U-D) = {pred.mean():+8.1f}"
        f"   observed = {obs.mean():+8.1f}  (95% CI ±{1.96*se:.1f})"
    )
    sub_c = pairs_df["polarity_contrast"] > 0.2
    if (sub_c & ok).sum() >= 3:
        pc, oc = pred[sub_c & ok], obs[sub_c & ok]
        sec = oc.std(ddof=1) / np.sqrt(len(oc))
        print(
            f"contrast>0.2 pairs (n={len(oc)}): predicted = {pc.mean():+8.1f}"
            f"   observed = {oc.mean():+8.1f}  (95% CI ±{1.96*sec:.1f})"
        )
        if abs(pc.mean()) > 1e-9:
            print(
                f"  observed / predicted ratio: {oc.mean()/pc.mean():+.2f} "
                f"(1 = fully boundary-level, 0 = fully composition/measure-side)"
            )

    # length confound within pairs
    dl = pairs_df["len_U"] - pairs_df["len_D"]
    print(
        f"length (func count) Δ(U-D): mean={dl.mean():+.2f}  "
        f"(pairs need not be equal-length; check confound)"
    )

    # measure-mirror check (H4)
    print("\n--- measure-mirror check: up(e) vs down(~e) on identical boundaries ---")
    m1 = pairs_df["mirror_up_D_vs_down_U"].abs()
    m2 = pairs_df["mirror_down_D_vs_up_U"].abs()
    both = pd.concat([m1, m2])
    print(
        f"  |up(e) - down(~e)|: mean={both.mean():.4f}  median={both.median():.4f}  "
        f"max={both.max():.4f}  frac>0.05={np.mean(both > 0.05):.2f}"
    )
    ra, pa = stats.pearsonr(
        pd.concat([pairs_df["up_D"], pairs_df["down_D"]]),
        pd.concat([pairs_df["down_U"], pairs_df["up_U"]]),
    )
    print(f"  corr(up(e), down(~e)) pooled both directions: r={ra:.3f} p={pa:.2g}")

    # per-pair listing
    print("\n--- pair listing (sorted by polarity contrast) ---")
    listing = pairs_df.sort_values("polarity_contrast", ascending=False)
    for _, row in listing.iterrows():
        print(
            f"  contrast={row['polarity_contrast']:+.2f}  "
            f"AUCmean D={row['auc_mean_D']:7.1f} U={row['auc_mean_U']:7.1f}  "
            f"Δ={row['auc_mean_U']-row['auc_mean_D']:+8.1f}"
        )
        print(
            f"    D (down={row['down_D']:.2f}, up={row['up_D']:.2f}): {row['term_D'][:95]}"
        )
        print(
            f"    U (down={row['down_U']:.2f}, up={row['up_U']:.2f}): {row['term_U'][:95]}"
        )

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax = axes[0]
    for col_d, col_u, c, lab in [
        ("auc_lstm_D", "auc_lstm_U", "#e8a33d", "LSTM"),
        ("auc_tf_D", "auc_tf_U", "#4477aa", "Transformer"),
    ]:
        ax.scatter(
            pairs_df[col_d], pairs_df[col_u], s=28, alpha=0.75, color=c, label=lab
        )
    lim = [
        0,
        np.nanmax(pairs_df[["auc_lstm_U", "auc_lstm_D", "auc_tf_U", "auc_tf_D"]].values)
        * 1.05,
    ]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("AUC, downward-dominant member")
    ax.set_ylabel("AUC, upward/other member")
    ax.set_title("Within complement pairs\n(identical decision boundary)")
    ax.legend()

    ax = axes[1]
    delta = pairs_df["auc_mean_U"] - pairs_df["auc_mean_D"]
    ax.scatter(pairs_df["polarity_contrast"], delta, s=30, color="#555555")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("polarity contrast  Δ(down−up)$_D$ − Δ(down−up)$_U$")
    ax.set_ylabel("ΔAUC (U − D), mean of architectures")
    ax.set_title("Does difficulty gap track direction gap?")

    ax = axes[2]
    ax.scatter(
        pd.concat([pairs_df["up_D"], pairs_df["down_D"]]),
        pd.concat([pairs_df["down_U"], pairs_df["up_U"]]),
        s=30,
        color="#663399",
        alpha=0.8,
    )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("up(e)  [resp. down(e)]")
    ax.set_ylabel("down(~e)  [resp. up(~e)]")
    ax.set_title("Measure-mirror check (H4)")

    fig.tight_layout()
    suffix = f"_{output_tag}" if output_tag else ""
    figpath = figdir / f"negation_pair_test{suffix}.png"
    fig.savefig(figpath, dpi=150)
    print(f"\nsaved {rel(figpath)}")

    return pairs_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--altk-archive",
        type=Path,
        default=DEFAULT_ALTK_ARCHIVE,
        help="path to the original altk repo's src/examples directory",
    )
    ap.add_argument(
        "--n-verify",
        type=int,
        default=20000,
        help="number of training-style scenes for functional verification",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=repo_root() / "outputs" / "combined_runs_AOC_monotonicity_updated.csv",
    )
    ap.add_argument(
        "--tag",
        default="",
        help="Optional output suffix, e.g. 'corrected' preserves manuscript-era artifacts.",
    )
    args = ap.parse_args()

    outdir = repo_root() / "analysis" / "tables"
    figdir = repo_root() / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""
    table_path = outdir / f"11_negation_pair_test{suffix}.txt"
    tee = _Tee(sys.stdout, open(table_path, "w"))
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        if not args.altk_archive.exists():
            raise SystemExit(
                f"altk archive not found at {args.altk_archive}; pass --altk-archive"
            )
        runs = load_runs(args.csv)
        print_model_provenance(runs, args.csv, args.n_verify)
        terms, V, _uni, exprs = load_archive(args.altk_archive)
        sample = pd.read_csv(repo_root() / "expressions_sample_2k.csv")
        pairs, _sidx = find_pairs(terms, V, sample["term_expression"])
        verified, _failed = verify_pairs(pairs, exprs, args.altk_archive, args.n_verify)
        stats_df = per_expression_auc(runs)
        analyze_pairs(verified, terms, stats_df, outdir, figdir, args.tag)
    finally:
        sys.stdout = old_stdout
    print(f"wrote {rel(table_path)}")


if __name__ == "__main__":
    main()
