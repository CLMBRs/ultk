"""Q2-T2: truth-set-statistics mediation test.

Hypothesis H2 (HANDOFF.md section 4.3): the downward-monotonicity learning
advantage is carried by the *input statistics of the positive class* -- for a
downward-monotone quantifier the verifying models concentrate on small A/B
(few active referent zones), which may speed optimization; upward quantifiers'
positives concentrate on large models.

Test: for every trained expression, compute class-conditional input statistics
on a fixed sample of training-style models (generate_batch, M=12, X=16,
inclusive=False -- exactly how the training data was drawn), then ask whether
the directional-monotonicity effect on AUC collapses once those statistics are
controlled. If the downward coefficient survives, H2 is not the driver; if it
collapses, it is.

Features per expression (positives "+" vs negatives "-"):
    p_true              share of models verified
    mean zone counts    |A-only|, |B-only|, |A&B|, |A|, |B|, |A|+|B| among +
    within-class var    total variance of the zone-count vector among +
    class separation    L2 distance between mean zone vectors of + and -
    boundary density    fraction of single-digit-flip edges of the 256-model
                        universe that cross the truth boundary (exact)

Run (needs the conda env `altk` and the original run archive):
    python scripts/truth_set_stats.py

Outputs:
    analysis/tables/12_truth_set_stats.txt
    figures/truth_set_stats.png
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.api as sm

DEFAULT_ALTK_ARCHIVE = Path.home() / "Documents/UWLing/altk/src/examples"
POOL_REL = "learn_quant/outputs/M4/X4/d5"
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


def load_archive(archive: Path):
    sys.path.insert(0, str(archive))
    from ultk.util.frozendict import FrozenDict

    FrozenDict.__setitem__ = dict.__setitem__
    import dill as pkl

    base = archive / POOL_REL
    uni = pkl.load(open(base / "master_universe.pkl", "rb"))
    pool = pkl.load(open(base / "generated_expressions_xidx.pkl", "rb"))
    by_term = {e.term_expression: e for e in pool.values()}
    return uni, by_term


def training_style_models(n: int, seed: int):
    from learn_quant.sampling import generate_batch
    from learn_quant.quantifier import QuantifierModel

    state = np.random.get_state()
    np.random.seed(seed)
    arrays = generate_batch(GEN_M_SIZE, GEN_X_SIZE, n, inclusive=GEN_INCLUSIVE)
    np.random.set_state(state)
    models = [QuantifierModel(a) for a in arrays]
    # zone counts per model: digits 0 (A-only), 1 (B-only), 2 (both), 3 (M-only)
    Z = np.stack([(arrays == d).sum(axis=1) for d in range(4)], axis=1)
    return models, Z


def universe_boundary_density(uni, expr) -> float:
    """Fraction of single-digit-change edges of the universe crossing the
    truth boundary (exact; 256 models, digits 0-3)."""
    refs = uni.referents
    name_to_val = {r.name: bool(expr.meaning.mapping[r]) for r in refs}
    edges = crossing = 0
    for r in refs:
        name = r.name
        for pos in range(len(name)):
            for d in "0123":
                if d > name[pos]:
                    other = name[:pos] + d + name[pos + 1 :]
                    if other in name_to_val:
                        edges += 1
                        if name_to_val[name] != name_to_val[other]:
                            crossing += 1
    return crossing / edges if edges else np.nan


def compute_features(sample_terms, by_term, uni, n_models: int, seed: int):
    models, Z = training_style_models(n_models, seed)
    zsum = Z.sum(axis=1)  # |A|+|B|+|A&B|+|M-only| (total M occupancy = 12)
    sizeA = Z[:, 0] + Z[:, 2]
    sizeB = Z[:, 1] + Z[:, 2]
    feats = []
    t0 = time.time()
    for k, term in enumerate(sample_terms):
        e = by_term[term]
        q = np.fromiter((bool(e(m)) for m in models), dtype=bool, count=len(models))
        p_true = q.mean()
        row = {"expression": term, "p_true": p_true}
        for lab, mask in (("pos", q), ("neg", ~q)):
            if mask.sum() == 0:
                for c in ("sizeA", "sizeB", "both", "union", "activity", "var"):
                    row[f"{lab}_{c}"] = np.nan
                continue
            zc = Z[mask]
            row[f"{lab}_sizeA"] = sizeA[mask].mean()
            row[f"{lab}_sizeB"] = sizeB[mask].mean()
            row[f"{lab}_both"] = Z[mask, 2].mean()
            row[f"{lab}_union"] = (Z[mask, 0] + Z[mask, 1] + Z[mask, 2]).mean()
            row[f"{lab}_activity"] = (sizeA[mask] + sizeB[mask]).mean()
            row[f"{lab}_var"] = zc.var(axis=0).sum()
        if not np.isnan(row.get("pos_sizeA", np.nan)) and not np.isnan(
            row.get("neg_sizeA", np.nan)
        ):
            mp = np.array([row["pos_sizeA"], row["pos_sizeB"], row["pos_both"]])
            mn = np.array([row["neg_sizeA"], row["neg_sizeB"], row["neg_both"]])
            row["class_sep"] = float(np.linalg.norm(mp - mn))
        else:
            row["class_sep"] = np.nan
        row["boundary_density"] = universe_boundary_density(uni, e)
        feats.append(row)
        if (k + 1) % 400 == 0:
            print(f"  {k+1}/{len(sample_terms)} expressions "
                  f"({time.time()-t0:.0f}s elapsed)")
    return pd.DataFrame(feats).set_index("expression")


def load_auc(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["expression"].notna() & (df["training"] == True)].copy()  # noqa: E712
    auc = (
        df.groupby(["expression", "model"])["val_loss_step_AOC"]
        .mean()
        .unstack("model")
        .rename(columns={"LSTM": "auc_lstm", "Transformer": "auc_tf"})
    )
    deg = df.groupby("expression")[DIRS + ["degree"]].first()
    out = auc.join(deg)
    out["upward"] = out[["right_upward", "left_upward"]].clip(0, 1).max(axis=1)
    out["downward"] = out[["right_downward", "left_downward"]].clip(0, 1).max(axis=1)
    out["auc_mean"] = out[["auc_lstm", "auc_tf"]].mean(axis=1)
    out["func_count"] = out.index.str.count(r"\(")
    return out


STAT_COLS = [
    "p_true",
    "pos_sizeA",
    "pos_sizeB",
    "pos_both",
    "pos_activity",
    "pos_var",
    "neg_activity",
    "class_sep",
    "boundary_density",
]


def _z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def fit_and_report(df: pd.DataFrame, outcome: str):
    """Nested OLS: directional effects with/without truth-set statistics."""
    use = df.dropna(subset=[outcome, "downward", "upward", *STAT_COLS]).copy()
    zX = pd.DataFrame(
        {c: _z(use[c]) for c in ["downward", "upward", "func_count", *STAT_COLS]}
    )
    y = use[outcome]

    def ols(cols):
        X = sm.add_constant(zX[cols])
        return sm.OLS(y, X).fit()

    m_dir = ols(["downward", "upward"])
    m_stats = ols(STAT_COLS)
    m_both = ols(["downward", "upward", *STAT_COLS])
    m_all = ols(["downward", "upward", "func_count", *STAT_COLS])

    print(f"\n===== outcome: {outcome}  (n={len(use)}) =====")
    print(f"  directions only:      R2={m_dir.rsquared:.3f}   "
          f"downward β={m_dir.params['downward']:+7.1f} (p={m_dir.pvalues['downward']:.2g})   "
          f"upward β={m_dir.params['upward']:+7.1f} (p={m_dir.pvalues['upward']:.2g})")
    print(f"  truth-set stats only: R2={m_stats.rsquared:.3f}")
    print(f"  directions + stats:   R2={m_both.rsquared:.3f}   "
          f"downward β={m_both.params['downward']:+7.1f} (p={m_both.pvalues['downward']:.2g})   "
          f"upward β={m_both.params['upward']:+7.1f} (p={m_both.pvalues['upward']:.2g})")
    print(f"  + func_count:         R2={m_all.rsquared:.3f}   "
          f"downward β={m_all.params['downward']:+7.1f} (p={m_all.pvalues['downward']:.2g})")
    shrink = 1 - m_both.params["downward"] / m_dir.params["downward"]
    print(f"  downward-β shrinkage from stats control: {shrink:+.1%}")
    print("\n  stats coefficients in the full model (per SD):")
    for c in STAT_COLS:
        print(f"    {c:18s} β={m_both.params[c]:+8.1f}  p={m_both.pvalues[c]:.2g}")
    return m_dir, m_both


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--altk-archive", type=Path, default=DEFAULT_ALTK_ARCHIVE)
    ap.add_argument("--n-models", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--csv", type=Path,
                    default=repo_root() / "outputs" / "combined_runs_AOC_monotonicity_updated.csv")
    args = ap.parse_args()

    outdir = repo_root() / "analysis" / "tables"
    figdir = repo_root() / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    table_path = outdir / "12_truth_set_stats.txt"
    tee = _Tee(sys.stdout, open(table_path, "w"))
    old = sys.stdout
    sys.stdout = tee
    try:
        uni, by_term = load_archive(args.altk_archive)
        sample = pd.read_csv(repo_root() / "expressions_sample_2k.csv")
        print(f"computing truth-set statistics for {len(sample)} expressions on "
              f"{args.n_models} training-style models (M={GEN_M_SIZE}, X={GEN_X_SIZE})")
        feats = compute_features(
            sample["term_expression"], by_term, uni, args.n_models, args.seed
        )
        auc = load_auc(args.csv)
        df = auc.join(feats, how="inner")
        print(f"joined: {len(df)} expressions with trained runs + features")

        # how do the stats relate to direction? (the confound structure)
        print("\n--- correlation of truth-set stats with directional degrees ---")
        for c in STAT_COLS:
            rd = df["downward"].corr(df[c])
            ru = df["upward"].corr(df[c])
            ra = df["auc_mean"].corr(df[c])
            print(f"  {c:18s} corr(down)={rd:+.3f}  corr(up)={ru:+.3f}  corr(AUC)={ra:+.3f}")

        m_dir, m_both = fit_and_report(df, "auc_mean")
        fit_and_report(df, "auc_lstm")
        fit_and_report(df, "auc_tf")

        # ---------------- figure ----------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        ax = axes[0]
        labels = ["directions\nonly", "+ truth-set\nstats"]
        betas = [m_dir.params["downward"], m_both.params["downward"]]
        errs = [1.96 * m_dir.bse["downward"], 1.96 * m_both.bse["downward"]]
        ax.bar(labels, betas, yerr=errs, color=["#4477aa", "#aa7744"], width=0.55)
        ax.axhline(0, color="k", lw=1)
        ax.set_ylabel("downward β on mean AUC (per SD)")
        ax.set_title("Does the downward effect survive\ntruth-set-statistics control?")

        ax = axes[1]
        ax.scatter(df["pos_activity"], df["auc_mean"], s=8, alpha=0.4, color="#555555")
        ax.set_xlabel("mean |A|+|B| among positive examples")
        ax.set_ylabel("mean AUC")
        ax.set_title("H2's proposed mediator vs difficulty")

        ax = axes[2]
        ax.scatter(df["boundary_density"], df["auc_mean"], s=8, alpha=0.4, color="#663399")
        ax.set_xlabel("truth-boundary edge density (universe)")
        ax.set_ylabel("mean AUC")
        ax.set_title("Boundary complexity vs difficulty")

        fig.tight_layout()
        figpath = figdir / "truth_set_stats.png"
        fig.savefig(figpath, dpi=150)
        print(f"\nsaved {rel(figpath)}")

        feats_path = repo_root() / "analysis" / "truth_set_stats_features.csv"
        df.reset_index().to_csv(feats_path, index=False)
        print(f"features saved to {rel(feats_path)}")
    finally:
        sys.stdout = old
    print(f"wrote {rel(table_path)}")


if __name__ == "__main__":
    main()
