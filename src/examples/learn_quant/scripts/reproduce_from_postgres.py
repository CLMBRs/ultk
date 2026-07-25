"""Reproduce the manuscript figures from the LIVE MLflow Postgres DB.

This pulls the *original* experiment data (not the CSV snapshot) through an SSH
tunnel to the klone cluster, using the exact queries from the paper's notebooks
(``get_AOC.ipynb``, ``get_experiment_data.ipynb``). It reconstructs, per run:

- ``expression``, ``expression_depth``, ``model``      (params)
- ``monotonicity_entropic``                             (metric, last step)
- ``val_loss_step_AOC`` = SUM(val_loss_step)            (get_AOC.ipynb)
- ``first_step`` = MIN(step) where val_loss_running_avg50 < 0.05
                                                        (get_experiment_data.ipynb)

Prerequisite: open the tunnel first (separate terminal on your Mac)::

    ssh -N klone-postgres          # forwards localhost:5432 -> g3115:5432

Then run::

    python scripts/reproduce_from_postgres.py --explore     # inspect the DB
    python scripts/reproduce_from_postgres.py               # build figures

Figures are written to ``learn_quant/figures/`` with a ``_postgres`` suffix so
they sit alongside the CSV-based ones for comparison.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the leaf-count parser from the sibling script.
from reproduce_figures import count_leaves, count_nodes, plot_complexity_vs_learning, rel

DEFAULT_DSN = os.environ.get(
    "MLFLOW_PG_DSN", "postgresql://USER:PASSWORD@localhost:5432/mlflow_db"
)

# The four experiments the manuscript combined for Figure 1:
#   LSTM:        40 (expressions_shuffled_2k) + 42 (repeated_runs)      = 4005 runs
#   Transformer: 46 (transformers_improved_1) + 47 (transformers_improved_2) = 4000 runs
# => 8005 runs total, matching outputs/combined_runs_AOC_monotonicity_updated.csv
PAPER_EXPERIMENTS = ["40", "42", "46", "47"]


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def connect(dsn: str):
    try:
        import psycopg  # psycopg 3
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "psycopg not installed. Run: pip install 'psycopg[binary]'"
        ) from exc
    try:
        return psycopg.connect(dsn, connect_timeout=10)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Could not connect to {dsn}\n"
            f"  {exc}\n"
            "Is the tunnel open?  ssh -N klone-postgres"
        ) from exc


# --------------------------------------------------------------------------- #
# Exploration
# --------------------------------------------------------------------------- #
def explore(conn) -> None:
    cur = conn.cursor()

    cur.execute(
        """
        SELECT e.experiment_id, e.name, count(*) AS n_runs
        FROM runs r JOIN experiments e ON r.experiment_id = e.experiment_id
        GROUP BY e.experiment_id, e.name
        ORDER BY n_runs DESC
        """
    )
    print("== experiments ==")
    for eid, name, n in cur.fetchall():
        print(f"  id={eid:<6} runs={n:<7} {name}")

    cur.execute("SELECT key, count(*) FROM metrics GROUP BY key ORDER BY 2 DESC")
    print("\n== metric keys ==")
    for k, n in cur.fetchall():
        print(f"  {k:<28} {n}")

    cur.execute("SELECT key, count(*) FROM params GROUP BY key ORDER BY 2 DESC LIMIT 30")
    print("\n== param keys (top 30) ==")
    for k, n in cur.fetchall():
        print(f"  {k:<28} {n}")


# --------------------------------------------------------------------------- #
# Data assembly (one row per run)
# --------------------------------------------------------------------------- #
def _model_short(target: str | None) -> str | None:
    if not target:
        return None
    tail = target.rsplit(".", 1)[-1].lower()
    if "lstm" in tail:
        return "LSTM"
    if "transformer" in tail:
        return "Transformer"
    return target.rsplit(".", 1)[-1]


def fetch_run_table(conn, experiment_ids: list[str] | None) -> pd.DataFrame:
    cur = conn.cursor()

    if not experiment_ids:
        experiment_ids = PAPER_EXPERIMENTS
    placeholders = ",".join(["%s"] * len(experiment_ids))
    ids = list(experiment_ids)

    def _dec(v):
        return v.decode() if isinstance(v, (bytes, bytearray)) else v

    # run -> experiment (restricted set)
    cur.execute(
        f"SELECT r.run_uuid, r.experiment_id FROM runs r "
        f"WHERE r.experiment_id IN ({placeholders})",
        ids,
    )
    runs = pd.DataFrame(cur.fetchall(), columns=["run_uuid", "experiment_id"])
    if runs.empty:
        return runs
    runs["run_uuid"] = runs["run_uuid"].apply(_dec)
    uuids = runs["run_uuid"].tolist()

    # The metrics table is ~675M rows; there is a btree index on run_uuid.
    # Passing the explicit uuid list makes every metric query an index scan
    # instead of a 195 GB sequential scan.

    # params of interest, pivoted
    cur.execute(
        """
        SELECT run_uuid, key, value FROM params
        WHERE run_uuid = ANY(%s)
          AND key IN ('expression','expression_depth','model__target_')
        """,
        (uuids,),
    )
    params = pd.DataFrame(cur.fetchall(), columns=["run_uuid", "key", "value"])
    params["run_uuid"] = params["run_uuid"].apply(_dec)
    params["key"] = params["key"].apply(_dec)
    params["value"] = params["value"].apply(_dec)
    params = params.pivot_table(
        index="run_uuid", columns="key", values="value", aggfunc="first"
    ).reset_index()

    # monotonicity_entropic: value at the last step per run
    cur.execute(
        """
        SELECT DISTINCT ON (run_uuid) run_uuid, value
        FROM metrics
        WHERE run_uuid = ANY(%s) AND key = 'monotonicity_entropic'
        ORDER BY run_uuid, step DESC, timestamp DESC
        """,
        (uuids,),
    )
    mono = pd.DataFrame(cur.fetchall(), columns=["run_uuid", "monotonicity_entropic"])

    # val_loss_step_AOC = SUM(val_loss_step)   (get_AOC.ipynb)
    cur.execute(
        """
        SELECT run_uuid, SUM(value) AS aoc
        FROM metrics
        WHERE run_uuid = ANY(%s) AND key = 'val_loss_step'
        GROUP BY run_uuid
        """,
        (uuids,),
    )
    aoc = pd.DataFrame(cur.fetchall(), columns=["run_uuid", "val_loss_step_AOC"])

    # first_step = MIN(step) where val_loss_running_avg50 < 0.05
    cur.execute(
        """
        SELECT run_uuid, MIN(step) AS first_step
        FROM metrics
        WHERE run_uuid = ANY(%s)
          AND key = 'val_loss_running_avg50' AND value < 0.05
        GROUP BY run_uuid
        """,
        (uuids,),
    )
    first = pd.DataFrame(cur.fetchall(), columns=["run_uuid", "first_step"])

    # Normalize run_uuid to str across all frames so merges line up.
    def _dec(s: pd.Series) -> pd.Series:
        return s.apply(lambda v: v.decode() if isinstance(v, (bytes, bytearray)) else v)

    for frame in (runs, params, mono, aoc, first):
        if "run_uuid" in frame.columns:
            frame["run_uuid"] = _dec(frame["run_uuid"])

    df = (
        runs.merge(params, on="run_uuid", how="left")
        .merge(mono, on="run_uuid", how="left")
        .merge(aoc, on="run_uuid", how="left")
        .merge(first, on="run_uuid", how="left")
    )

    df = df.rename(columns={"model__target_": "model_target"})
    df["model"] = df["model_target"].apply(_model_short)
    df["expression_depth"] = pd.to_numeric(df["expression_depth"], errors="coerce")
    df["monotonicity_entropic"] = pd.to_numeric(df["monotonicity_entropic"], errors="coerce")
    df["val_loss_step_AOC"] = pd.to_numeric(df["val_loss_step_AOC"], errors="coerce")
    df["first_step"] = pd.to_numeric(df["first_step"], errors="coerce")

    df = df[df["expression"].notna()].copy()
    df["leaf_count"] = df["expression"].apply(count_leaves)
    df["node_count"] = df["expression"].apply(count_nodes)
    return df


# --------------------------------------------------------------------------- #
# Paper Figure 1 (from live data)
# --------------------------------------------------------------------------- #
def plot_paper_figure1_pg(df: pd.DataFrame, outpath: Path) -> None:
    data = df.dropna(subset=["val_loss_step_AOC", "monotonicity_entropic"]).copy()
    if data.empty:
        print("  [skip] no rows with AOC + monotonicity")
        return

    colors = {"LSTM": "#F8766D", "Transformer": "#00BFC4"}
    rng = np.random.default_rng(0)
    x = data["val_loss_step_AOC"].to_numpy(float)
    y = data["monotonicity_entropic"].to_numpy(float)
    yj = y + rng.uniform(-0.01, 0.01, size=len(y))

    fig, ax = plt.subplots(figsize=(12, 8))
    if data["model"].notna().any():
        for name, sub in data.groupby("model"):
            idx = (data["model"] == name).to_numpy()
            ax.scatter(x[idx], yj[idx], s=14, alpha=0.5, edgecolors="none",
                       color=colors.get(str(name)), label=str(name))
    else:
        ax.scatter(x, yj, s=14, alpha=0.5, edgecolors="none", color="#0073C2")

    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, "r--", lw=2)
    r = np.corrcoef(x, y)[0, 1]

    ax.set_xlabel("Validation Loss AUC", fontsize=20, fontweight="bold")
    ax.set_ylabel("Monotonicity", fontsize=20, fontweight="bold")
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", color="gray", alpha=0.5)
    ax.legend(title=f"Model   (r = {r:.2f})", fontsize=14, title_fontsize=15, loc="upper right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=600)
    plt.close(fig)
    print(f"  saved {rel(outpath)}  ({len(data)} runs, r={r:.3f})")


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dsn", default=DEFAULT_DSN)
    p.add_argument("--explore", action="store_true", help="Print DB inventory and exit.")
    p.add_argument("--experiment", action="append", default=None,
                   help="Restrict to experiment id(s). Repeatable. Default: all.")
    p.add_argument("--outdir", type=Path, default=root / "figures")
    p.add_argument("--dump-csv", type=Path, default=None,
                   help="Also write the assembled per-run table to this CSV.")
    args = p.parse_args()

    conn = connect(args.dsn)

    if args.explore:
        explore(conn)
        return

    args.outdir.mkdir(parents=True, exist_ok=True)
    print("Fetching run table from Postgres...")
    df = fetch_run_table(conn, args.experiment)
    print(f"  assembled {len(df)} runs")
    print("  models:", df["model"].value_counts(dropna=False).to_dict())
    print("  with AOC:", int(df["val_loss_step_AOC"].notna().sum()),
          " with monotonicity:", int(df["monotonicity_entropic"].notna().sum()),
          " with first_step:", int(df["first_step"].notna().sum()))

    if args.dump_csv:
        df.to_csv(args.dump_csv, index=False)
        print(f"  wrote {args.dump_csv}")

    print("Figure: paper Figure 1 (Monotonicity vs Validation Loss AUC)")
    plot_paper_figure1_pg(df, args.outdir / "paper_figure1_postgres.png")

    print("Figure: depth vs learning")
    plot_complexity_vs_learning(
        df, xcol="expression_depth",
        xlabel="Expression depth\n(parenthesis nesting)",
        outpath=args.outdir / "depth_vs_learning_postgres.png",
    )

    print("Figure: length (leaf count) vs learning")
    plot_complexity_vs_learning(
        df, xcol="leaf_count",
        xlabel="Expression length\n(number of leaf nodes / atoms)",
        outpath=args.outdir / "length_vs_learning_postgres.png",
    )

    print(f"\nDone. Figures written to {args.outdir}")


if __name__ == "__main__":
    main()
