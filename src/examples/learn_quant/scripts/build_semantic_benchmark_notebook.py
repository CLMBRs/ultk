"""Build the pedagogical semantic benchmark notebook."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "notebooks/two_sided_min_semantic_benchmark.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Does two-sided minimum match familiar monotonicity intuitions?

This notebook stress-tests the proposed symmetric measure on **34 familiar
set-theoretic meanings**, not just the three examples in manuscript Table 4.

The test is deliberately theory-first:

1. state each meaning in ordinary language and set notation;
2. state its expected exact monotonicity directions;
3. verify those expectations by exhaustively checking every comparable pair in
   the manuscript's M6/X6 universe;
4. only then inspect the entropy scores.

Thus the metric does not choose its own test cases or gold labels.
"""
    ),
    code(
        r"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.cwd()
if not (ROOT / "analysis").is_dir():
    ROOT = ROOT.parent

CSV = ROOT / "analysis/semantic_monotonicity_benchmark.csv"
df = pd.read_csv(CSV)
DIRECTIONS = ["RU", "LU", "RD", "LD"]
print(f"{len(df)} benchmark meanings across {df.category.nunique()} semantic families")
df.groupby("category").size().rename("meanings").to_frame()
"""
    ),
    md(
        r"""
## 1. What the four columns mean

- **RU:** preserve truth when B grows.
- **LU:** preserve truth when A grows.
- **RD:** preserve truth when B shrinks.
- **LD:** preserve truth when A shrinks.

An expected value of 1 means exact monotonicity in that direction. A blank
expectation means the meaning has at least one counterexample in the finite
universe. That does **not** require a graded score of exactly zero: a graded
measure may reasonably say that a nonmonotone meaning is close to monotone.
The question is whether the amount and pattern of partial credit are persuasive.
"""
    ),
    code(
        r"""
display_cols = [
    "category", "meaning", "formula", "expected_exact_directions",
    "min_RU", "min_LU", "min_RD", "min_LD",
]
table = df[display_cols].copy()
table[["min_RU", "min_LU", "min_RD", "min_LD"]] = table[
    ["min_RU", "min_LU", "min_RD", "min_LD"]
].round(3)
table
"""
    ),
    md(
        r"""
## 2. First result: exact monotonicity is recovered perfectly

For every direction independently verified as exactly monotone, two-sided
minimum returns 1. No non-exact direction returns 1. This is the strongest
positive result: the measure gets the categorical endpoints right throughout
this benchmark, including all/no/some, proper subset, cardinal comparisons,
overlap thresholds, proportional meanings, and constants.
"""
    ),
    code(
        r"""
direction_rows = []
for _, row in df.iterrows():
    for direction in DIRECTIONS:
        direction_rows.append(
            {
                "meaning": row["meaning"],
                "direction": direction,
                "expected_exact": bool(row[f"expected_{direction}"]),
                "score": row[f"min_{direction}"],
                "preservation": row[f"preservation_{direction}"],
            }
        )
directions = pd.DataFrame(direction_rows)
summary = directions.groupby("expected_exact").score.agg(
    ["count", "min", "median", "mean", "max"]
)
summary.index = ["non-exact", "exact"]
summary.round(3)
"""
    ),
    md(
        r"""
## 3. Clear successes on familiar meanings

The following cases fit ordinary semantic judgments especially well:

- `all A are B`: RU and LD are 1; the other directions are 0.
- `no A are B`: RD and LD are 1; the upward directions are 0.
- `some A are B`: RU and LU are 1; downward directions are 0.
- proper subset/superset and greater/fewer cardinality reverse exactly as they
  should.
- set equality, cardinal equality, comparability, and their complements are
  near 0 or exactly 0 in every direction.

These are not isolated expressions harvested from the grammar. They are direct
semantic predicates evaluated over all 4,096 M6/X6 situations.
"""
    ),
    code(
        r"""
success_names = [
    "all A are B", "no A are B", "some A are B",
    "A proper-subset B", "more A than B",
    "A equals B", "same number of A and B",
    "A and B are comparable", "A and B are incomparable",
]
table[table.meaning.isin(success_names)].reset_index(drop=True)
"""
    ),
    md(
        r"""
## 4. The important counterexamples

Two-sided minimum is **not** uniformly intuitive as a graded scale.

The table below ranks non-exact directions by the partial score they receive.
Three patterns matter:

1. **Finite-boundary exact counts.** `exactly five overlap` is classically
   nonmonotone, but on a six-element universe it is only one upward step away
   from the maximum. It receives a substantial upward score.
2. **Boundary biconditionals/XOR.** Whether both sets are empty or nonempty
   receives partial upward evidence; its complement receives the mirrored
   downward evidence. Symmetry is satisfied, but the magnitude is not an
   obvious semantic intuition.
3. **Proportional quantifiers.** `most A are B` is exactly RU, but receives some
   LU credit even though growing A can either preserve or destroy truth. The
   amount depends on the finite universe and threshold convention.

These are genuine limitations, not implementation errors.
"""
    ),
    code(
        r"""
nonexact = directions[~directions.expected_exact].sort_values(
    "score", ascending=False
)
nonexact.head(20).assign(
    score=lambda x: x.score.round(3),
    preservation=lambda x: x.preservation.round(3),
)
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(nonexact.preservation, nonexact.score, alpha=0.65)
ax.set(
    xlabel="conditional truth-preservation over comparable pairs",
    ylabel="two-sided-min entropy score",
    title="Partial entropy score is not just pairwise truth preservation",
)
ax.axline((0, 0), (1, 1), color="grey", linestyle="--", linewidth=1)
plt.show()
print(
    "Spearman correlation:",
    nonexact[["preservation", "score"]].corr(method="spearman").iloc[0, 1].round(3),
)
"""
    ),
    md(
        r"""
The transparent preservation rate asks: among comparable changes beginning at
a true situation, how often is truth preserved? It is not proposed here as the
new primary measure; it is a diagnostic. The imperfect relationship shows that
entropy scores also depend on truth prevalence and feature balance. Therefore a
number such as 0.61 cannot be read directly as “61% monotone.”
"""
    ),
    md(
        r"""
## 5. Comparison with the majorant and two-sided mean

Two-sided minimum fixes two concrete problems:

- unlike the majorant, it mirrors complements by construction;
- unlike the mean, one boundary-inflated feature cannot raise a direction when
  the other feature has score 0.

But taking the minimum does not remove every finite-boundary effect when **both**
closure and interior are informative. Exact high overlap counts demonstrate
this remaining issue.
"""
    ),
    code(
        r"""
comparison_names = [
    "A and B are comparable",
    "A equals B",
    "exactly one overlaps",
    "exactly three overlap",
    "exactly five overlap",
    "most A are B",
    "exactly one of A and B is nonempty",
]
rows = []
for _, row in df[df.meaning.isin(comparison_names)].iterrows():
    for direction in DIRECTIONS:
        if not row[f"expected_{direction}"]:
            rows.append(
                {
                    "meaning": row.meaning,
                    "direction": direction,
                    "majorant": row[f"majorant_{direction}"],
                    "two-sided mean": row[f"mean_{direction}"],
                    "two-sided minimum": row[f"min_{direction}"],
                }
            )
pd.DataFrame(rows).round(3)
"""
    ),
    md(
        r"""
## 6. Judgment

**Does two-sided minimum match intuition better than the two-sided mean? Yes.**
It preserves both desired symmetries, returns the correct 1/0 patterns for the
core textbook quantifiers, and removes the mean's comparability artifact.

**Is it established as an intuitively calibrated general degree? No.** It
classifies exact monotonicity correctly, but its intermediate values can be
large for meanings that are nonmonotone in the ordinary unbounded-domain sense.
Those values reflect closeness on this particular finite lattice, truth
prevalence, and entropy—not only the number or severity of monotonicity
violations.

### Recommendation

- Keep the **manuscript majorant** for faithful reproduction.
- Treat **two-sided minimum** as the best symmetric *candidate tested so far*,
  not a validated replacement.
- Before using it as a scientific predictor, test stability across M4, M6, M8,
  and alternative universe weightings.
- In parallel, evaluate a direct distance-to-nearest-monotone-function measure.
  Its interpretation (“minimum weighted truth-value changes”) may align more
  directly with the intended graded construct.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "altk",
            "language": "python",
            "name": "altk",
        },
        "language_info": {"name": "python", "version": "3.12"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
