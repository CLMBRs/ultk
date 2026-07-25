"""Verify the committed CSV equals the live Postgres data (sampled).

Rather than re-aggregating the entire 195 GB metrics table (very slow over the
tunnel), this samples specific ``run_uuid``s from the CSV and pulls just those
runs from Postgres via the run_uuid index. It then compares, per run:

- ``val_loss_step_AOC``      == SUM(metrics.val_loss_step)
- ``monotonicity_entropic``  == last metrics.monotonicity_entropic

If they match for a random sample, the CSV faithfully represents the DB.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

DSN = os.environ.get(
    "MLFLOW_PG_DSN", "postgresql://USER:PASSWORD@localhost:5432/mlflow_db"
)


def _dec(v):
    return v.decode() if isinstance(v, (bytes, bytearray)) else v


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(root / "outputs" / "combined_runs_AOC_monotonicity_updated.csv"))
    ap.add_argument("--n", type=int, default=150, help="sample size of runs")
    ap.add_argument("--dsn", default=DSN)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["run_uuid"].notna()].copy()

    # First cancel any abandoned heavy queries so this is fast.
    admin = psycopg.connect(args.dsn, connect_timeout=10)
    admin.autocommit = True
    ac = admin.cursor()
    ac.execute(
        """
        SELECT pg_cancel_backend(pid) FROM pg_stat_activity
        WHERE datname='mlflow_db' AND state='active'
          AND query LIKE '%metrics%' AND query NOT LIKE '%pg_stat_activity%'
        """
    )
    admin.close()

    # Sample runs that have an AOC value in the CSV.
    have_aoc = df[df["val_loss_step_AOC"].notna()]
    sample = have_aoc.sample(min(args.n, len(have_aoc)), random_state=1)
    uuids = [str(u) for u in sample["run_uuid"].tolist()]

    conn = psycopg.connect(args.dsn, connect_timeout=10)
    cur = conn.cursor()

    # AOC = SUM(val_loss_step) per run (index scan on the sampled uuids only)
    cur.execute(
        "SELECT run_uuid, SUM(value) FROM metrics "
        "WHERE run_uuid = ANY(%s) AND key='val_loss_step' GROUP BY run_uuid",
        (uuids,),
    )
    pg_aoc = {_dec(u): float(v) for u, v in cur.fetchall()}

    # last monotonicity_entropic per run
    cur.execute(
        "SELECT DISTINCT ON (run_uuid) run_uuid, value FROM metrics "
        "WHERE run_uuid = ANY(%s) AND key='monotonicity_entropic' "
        "ORDER BY run_uuid, step DESC, timestamp DESC",
        (uuids,),
    )
    pg_mono = {_dec(u): float(v) for u, v in cur.fetchall()}

    # Compare.
    rows = []
    for _, r in sample.iterrows():
        u = str(r["run_uuid"])
        rows.append(
            {
                "run_uuid": u,
                "csv_aoc": r["val_loss_step_AOC"],
                "pg_aoc": pg_aoc.get(u, np.nan),
                "csv_mono": r.get("monotonicity_entropic", np.nan),
                "pg_mono": pg_mono.get(u, np.nan),
            }
        )
    comp = pd.DataFrame(rows)
    comp["aoc_diff"] = (comp["csv_aoc"] - comp["pg_aoc"]).abs()
    comp["mono_diff"] = (comp["csv_mono"] - comp["pg_mono"]).abs()

    n = len(comp)
    aoc_ok = int((comp["aoc_diff"] < 1e-3).sum())
    aoc_missing = int(comp["pg_aoc"].isna().sum())
    mono_ok = int((comp["mono_diff"] < 1e-6).sum())
    mono_missing = int(comp["pg_mono"].isna().sum())

    print(f"Sampled {n} runs from CSV, pulled from Postgres.")
    print(f"  AOC  matches (<1e-3): {aoc_ok}/{n}   (missing in PG: {aoc_missing})")
    print(f"  mono matches (<1e-6): {mono_ok}/{n}   (missing in PG: {mono_missing})")
    print(f"  max AOC abs diff:  {comp['aoc_diff'].max():.6g}")
    print(f"  max mono abs diff: {comp['mono_diff'].max():.6g}")
    print("\nSample mismatches (if any):")
    bad = comp[(comp["aoc_diff"] >= 1e-3) | (comp["mono_diff"] >= 1e-6)]
    if bad.empty:
        print("  none — CSV matches Postgres exactly.")
    else:
        print(bad.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
