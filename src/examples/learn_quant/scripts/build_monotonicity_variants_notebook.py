"""Build the pedagogical monotonicity-measure variants notebook."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "notebooks/monotonicity_measure_variants_walkthrough.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Which graded monotonicity measure do we want?

## A walkthrough of closures, interiors, complement symmetry, and order duality

This notebook separates four questions that had become conflated:

1. **What did the manuscript calculate?**
2. **Why does that calculation fail the complement-pair mirror test?**
3. **Was replacing `down(Q)` with `up(not Q)` a bug fix?**
4. **Can one entropy measure preserve both order reversal and truth-label complementation?**

### Short answer

- The manuscript used a coherent **closure / least-majorant** measure. It is the
  direct order-reversed implementation of the formula stated in the paper.
- It is **not complement invariant**. This is a structural property of using a
  one-sided closure, not numerical noise.
- The later replacement `down(Q) = up(not Q)` enforces complement invariance,
  but changes downward scoring from a majorant to a minorant. It is an
  **alternative measure**, not a neutral correction.
- The two desired symmetries are **not impossible together**. A symmetric
  function of closure and interior scores can preserve both.
- The arithmetic mean is not automatically the right symmetric function. It
  can reward boundary agreement from an interior even when the closure score is
  zero. The conservative **two-sided minimum** avoids that failure for the
  comparability predicate while preserving both symmetries.
- For the existing manuscript, retain the majorant as the primary published
  measure. For a future symmetric metric, the two-sided minimum is the best
  current candidate among the variants tested here, but it still needs broader
  construct validation.

The recommendation is justified step by step below rather than assumed.
"""),
    md(r"""
## 1. The two symmetries are different claims

Let `Q(A,B)` be a Boolean quantifier. For one argument, write `x <= y` when the
varying set in situation `x` is a subset of the corresponding set in `y`.

| Property | What changes? | Example identity | Why we might want it |
|---|---|---|---|
| **Order duality** | Reverse `<=` to `>=` | upward construction becomes the downward construction | Upward and downward should be defined by the same recipe, with order reversed |
| **Complement invariance** | Replace true by false: `Q -> not Q` | `up(Q) = down(not Q)` | `Q` and `not Q` have the same boundary with labels exchanged; BCE learning is label-complement symmetric |
| **Argument symmetry** | Exchange arguments `A` and `B` | right score equals left score for an argument-symmetric quantifier | Appropriate only when the quantifier itself is symmetric in its arguments |
| **Within-row equality** | Compare two directions of one expression | e.g. `RU(Q) = LD(Q)` | True for some textbook meanings, but not a general logical law |

The complement mirror test compares **different expressions in the same
argument**:

\[
RU(Q)=RD(\neg Q),\qquad LU(Q)=LD(\neg Q).
\]

It does **not** entail the within-expression statement `RU(Q) = LD(Q)`.
Table 4 happens to contain meanings with additional semantic symmetries, which
is why some within-row values match there.
"""),
    md(r"""
## 2. Four monotone approximations

For a Boolean function `Q` on a partial order, there are four natural
approximations:

\[
\begin{aligned}
C_\uparrow Q(x) &= \bigvee_{y\le x} Q(y)
&&\text{least upward-monotone majorant}\\
I_\uparrow Q(x) &= \bigwedge_{y\ge x} Q(y)
&&\text{greatest upward-monotone minorant}\\
C_\downarrow Q(x) &= \bigvee_{y\ge x} Q(y)
&&\text{least downward-monotone majorant}\\
I_\downarrow Q(x) &= \bigwedge_{y\le x} Q(y)
&&\text{greatest downward-monotone minorant.}
\end{aligned}
\]

- A **majorant** can change false points to true: it adds truths until the
  result is monotone.
- A **minorant** can change true points to false: it removes offending truths
  until the result is monotone.

Both equal `Q` when `Q` is exactly monotone in the relevant direction. Away
from exact monotonicity they are different approximations.
"""),
    code(r"""
import numpy as np
import pandas as pd

# A three-point chain x0 <= x1 <= x2.
q = np.array([1, 0, 1], dtype=int)

def cumulative_or(values):
    return np.maximum.accumulate(values)

def cumulative_and(values):
    return np.minimum.accumulate(values)

toy = pd.DataFrame(
    {
        "point": ["x0", "x1", "x2"],
        "Q": q,
        "C_up(Q)": cumulative_or(q),
        "I_up(Q)": cumulative_and(q[::-1])[::-1],
        "C_down(Q)": cumulative_or(q[::-1])[::-1],
        "I_down(Q)": cumulative_and(q),
    }
)
toy
"""),
    md(r"""
At `x1`, for example, `C_up(Q)=1` because a true predecessor (`x0`) exists,
while `I_down(Q)=0` because not **all** predecessors are true. Existence and
universality are not interchangeable.

This is the exact point behind the earlier phrase “existential feature
construction.” The implementation does not compare `Q` directly with every
ordering constraint. It constructs a binary feature:

> Does this situation have **at least one** true predecessor?

That feature is `C_up(Q)`. With `flip=True`, it asks for at least one true
successor, which is `C_down(Q)`.
"""),
    md(r"""
## 3. Why closure does not commute with negation

Now do the algebra explicitly:

\[
\begin{aligned}
C_\uparrow(\neg Q)(x)
  &= \bigvee_{y\le x}\neg Q(y)\\
  &= \neg\bigwedge_{y\le x}Q(y)\\
  &= \neg I_\downarrow Q(x).
\end{aligned}
\]

Therefore `C_up(not Q)` is the complement of the **downward interior**, not the
complement of the downward closure:

\[
C_\uparrow(\neg Q)=\neg I_\downarrow Q
\quad\text{but generally}\quad
C_\uparrow(\neg Q)\ne\neg C_\downarrow Q.
\]

Negation changes OR to AND (De Morgan's law), so it swaps **closure** and
**interior**. This is what breaks complement invariance in a closure-only
measure.
"""),
    code(r"""
not_q = 1 - q
demorgan = pd.DataFrame(
    {
        "point": ["x0", "x1", "x2"],
        "C_up(not Q)": cumulative_or(not_q),
        "not I_down(Q)": 1 - cumulative_and(q),
        "not C_down(Q)": 1 - cumulative_or(q[::-1])[::-1],
    }
)
demorgan["first_identity_holds"] = (
    demorgan["C_up(not Q)"] == demorgan["not I_down(Q)"]
)
demorgan["incorrect_closure_identity_holds"] = (
    demorgan["C_up(not Q)"] == demorgan["not C_down(Q)"]
)
demorgan
"""),
    md(r"""
The first identity holds at every point. The proposed closure-to-closure
identity fails. This is a logical difference, not a floating-point problem.
"""),
    md(r"""
## 4. Where entropy enters

The manuscript score is normalized mutual information between `Q` and one of
these approximations:

\[
\operatorname{score}(Q,M(Q))
=1-\frac{H(Q\mid M(Q))}{H(Q)}
=\frac{I(Q;M(Q))}{H(Q)}.
\]

It equals 1 when the approximation predicts `Q` perfectly. It is near 0 when
knowing the approximation gives little information about `Q`.

Entropy is **not itself** the source of the asymmetry. Mutual information is
invariant under complementing either binary variable. The asymmetry arises
before entropy is calculated: complementation sends a closure to an interior,
but the manuscript compares closures in both order directions.
"""),
    md(r"""
## 5. The measure variants

### A. Manuscript-era majorant measure

\[
U_{\rm major}(Q)=s(Q,C_\uparrow Q),\qquad
D_{\rm major}(Q)=s(Q,C_\downarrow Q).
\]

It uses the same majorant recipe with the order reversed. This is the most
faithful implementation of the paper's stated “minimal monotone extension.”

### B. Complement-dual alternative

\[
U_{\rm comp}(Q)=s(Q,C_\uparrow Q),\qquad
D_{\rm comp}(Q)=U_{\rm comp}(\neg Q)
               =s(Q,I_\downarrow Q).
\]

This guarantees `U(Q)=D(not Q)`, but it treats upward and downward differently:
upward uses a majorant and downward uses a minorant.

### C. Two-sided entropy

\[
\begin{aligned}
U_{\rm 2s}(Q)&=\tfrac12[s(Q,C_\uparrow Q)+s(Q,I_\uparrow Q)],\\
D_{\rm 2s}(Q)&=\tfrac12[s(Q,C_\downarrow Q)+s(Q,I_\downarrow Q)].
\end{aligned}
\]

Under complement, closure and interior exchange. Because the arithmetic mean
is symmetric in its two inputs, this measure preserves both order duality and
complement invariance.

`min` and `max` are also symmetric combinations:

- **min** is conservative: both approximations must score highly.
- **max** is permissive: whichever approximation scores better wins.
- **mean** gives equal weight to adding missing truths and removing offending
  truths. It is the neutral default used in the recommendation.

### D. Direct violation rate

For upward monotonicity, count comparable pairs `x < y` for which
`Q(x)=1` and `Q(y)=0`, then report

\[
1-\frac{\#\text{violations}}{\#\text{proper comparable pairs}}.
\]

This is transparent and has both symmetries. Its drawback is scaling: a large
lattice can contain many irrelevant nonviolating pairs, making scores cluster
near 1.
"""),
    code(r"""
properties = pd.DataFrame(
    [
        {
            "variant": "majorant (manuscript)",
            "upward object": "C_up",
            "downward object": "C_down",
            "uniform under order reversal": "yes",
            "complement invariant": "no",
            "exact monotone -> 1": "yes",
            "entropy based": "yes",
        },
        {
            "variant": "complement dual",
            "upward object": "C_up",
            "downward object": "I_down",
            "uniform under order reversal": "no",
            "complement invariant": "yes",
            "exact monotone -> 1": "yes",
            "entropy based": "yes",
        },
        {
            "variant": "two-sided mean",
            "upward object": "mean(C_up, I_up)",
            "downward object": "mean(C_down, I_down)",
            "uniform under order reversal": "yes",
            "complement invariant": "yes",
            "exact monotone -> 1": "yes",
            "entropy based": "yes",
        },
        {
            "variant": "direct violation rate",
            "upward object": "upward violations",
            "downward object": "downward violations",
            "uniform under order reversal": "yes",
            "complement invariant": "yes",
            "exact monotone -> 1": "yes",
            "entropy based": "no",
        },
    ]
)
properties
"""),
    md(r"""
### Important pushback on “both properties are impossible”

They are impossible for a **one-sided closure-only** score. They are not
impossible for monotonicity metrics in general. Two-sided entropy works because
it includes the interior that complementation necessarily introduces.
"""),
    md(r"""
## 6. Load the actual manuscript universes and implementations

The next cells are executable rather than copied result tables. They load the
archived expression pools needed by the original pickles, then load the current
metric source under private module names.

Use the `altk` conda environment. No neural network is trained here.
"""),
    code(r"""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import os
import sys
import warnings

def find_learn_root():
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        direct = base / "src/examples/learn_quant"
        if (direct / "measures.py").exists():
            return direct
        if base.name == "learn_quant" and (base / "measures.py").exists():
            return base
    raise FileNotFoundError("Could not locate src/examples/learn_quant")

LEARN_ROOT = find_learn_root()
ALTK_ARCHIVE = Path.home() / "Documents/UWLing/altk/src/examples"
assert ALTK_ARCHIVE.exists(), f"Archived expression pools not found: {ALTK_ARCHIVE}"

sys.path.insert(0, str(ALTK_ARCHIVE))
os.chdir(ALTK_ARCHIVE)

import dill as pkl
from ultk.util.frozendict import FrozenDict

FrozenDict.__setitem__ = dict.__setitem__

def load_private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

measures = load_private_module(
    "learn_quant._walkthrough_measures", LEARN_ROOT / "measures.py"
)
variants = load_private_module(
    "learn_quant._walkthrough_variants", LEARN_ROOT / "monotonicity_variants.py"
)
cfg = SimpleNamespace(
    measures=SimpleNamespace(monotonicity=SimpleNamespace(debug=False))
)
print("Current analysis root:", LEARN_ROOT)
print("Archived pools:", ALTK_ARCHIVE)
"""),
    md(r"""
## 7. The three identifiable Table 4 expressions

This table explains the apparent new RU/LD asymmetry.

The manuscript-era values are generated by the majorant measure. The
complement-dual alternative was designed to satisfy **cross-expression**
identities such as `RU(Q)=RD(not Q)`. It was not designed to preserve
within-expression coincidences such as `RU(Q)=LD(Q)`.

The two-sided variants restore both structural symmetries. A within-row match
can still fail for an arbitrary expression; it appears here because these
particular meanings have extra left/right and order symmetries.
"""),
    code(r"""
M6_BASE = ALTK_ARCHIVE / "learn_quant/outputs/M6/X6/d3"
with open(M6_BASE / "master_universe.pkl", "rb") as handle:
    universe_m6 = pkl.load(handle)
with open(M6_BASE / "generated_expressions_xidx.pkl", "rb") as handle:
    pool_m6 = pkl.load(handle)

by_term_m6 = {
    expression.term_expression: expression for expression in pool_m6.values()
}
table4_terms = [
    "subset_eq(A, B)",
    "not(subset_eq(A, B))",
    "or(subset_eq(A, B), subset_eq(B, A))",
]
published = {
    "subset_eq(A, B)": [1.0, 0.0, 0.0, 1.0],
    "not(subset_eq(A, B))": [0.059, 1.0, 1.0, 0.059],
    "or(subset_eq(A, B), subset_eq(B, A))": [0.0, 0.0, 0.0, 0.0],
}

all_m6 = universe_m6.binarize_referents(mode="set_vectors_w_padding")
ref_a_m6 = universe_m6.binarize_referents(mode="A")
ref_b_m6 = universe_m6.binarize_referents(mode="B")

variant_names = [
    "majorant",
    "complement_dual",
    "two_sided_mean",
    "two_sided_min",
    "two_sided_max",
]
rows = []
for term in table4_terms:
    expression = by_term_m6[term]
    q_m6 = np.fromiter(
        (expression.meaning.mapping[ref] for ref in universe_m6.referents),
        dtype=int,
        count=len(universe_m6.referents),
    )
    rows.append({"expression": term, "variant": "published", **dict(zip(["RU", "LU", "RD", "LD"], published[term]))})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for variant_name in variant_names:
            values = variants.entropy_variant_scores(
                all_m6,
                ref_a_m6,
                ref_b_m6,
                q_m6,
                measures.upward_monotonicity_entropy,
                cfg,
                variant=variant_name,
            )
            rows.append(
                {
                    "expression": term,
                    "variant": variant_name,
                    **dict(zip(["RU", "LU", "RD", "LD"], values)),
                }
            )
    direct = variants.violation_rate_scores(
        all_m6, ref_a_m6, ref_b_m6, q_m6
    )
    rows.append(
        {
            "expression": term,
            "variant": "violation_rate",
            **dict(zip(["RU", "LU", "RD", "LD"], direct)),
        }
    )

table4 = pd.DataFrame(rows)
table4[["RU", "LU", "RD", "LD"]] = table4[
    ["RU", "LU", "RD", "LD"]
].round(3)
table4
"""),
    md(r"""
### Reading the table

For `not(subset_eq(A,B))`, the published majorant score has `RU=LD=.059`.
The complement-dual alternative keeps `RU=.059` but changes `LD` to `0`.
That is not a contradiction:

- published `LD` asks whether the **downward closure** predicts the expression;
- alternative `LD` asks whether the **downward interior** predicts it.

The `.059` and `0` answer different approximation questions. The alternative
gains the cross-expression mirror identity but loses the original uniform
majorant construction.

The direct violation scores are high even for nonmonotone directions because
most comparable pairs are nonviolations. This illustrates their transparent
numerator but awkward scale.
"""),
    md(r"""
## 8. Test all 44 complement pairs in the trained sample

The next cell identifies exact truth-vector complements among the 2,000
trained expressions. For each pair it checks all four mirror equations. “Exact
mirrored pairs” means all four errors are below `1e-12`.
"""),
    code(r"""
M4_BASE = ALTK_ARCHIVE / "learn_quant/outputs/M4/X4/d5"
with open(M4_BASE / "master_universe.pkl", "rb") as handle:
    universe_m4 = pkl.load(handle)
with open(M4_BASE / "generated_expressions_xidx.pkl", "rb") as handle:
    pool_m4 = pkl.load(handle)

by_term_m4 = {
    expression.term_expression: expression for expression in pool_m4.values()
}
variant_2k = pd.read_csv(
    LEARN_ROOT / "analysis/monotonicity_measure_variants_2k.csv"
)
sample_terms = set(variant_2k["expression"])
vectors = {
    term: np.fromiter(
        (
            by_term_m4[term].meaning.mapping[ref]
            for ref in universe_m4.referents
        ),
        dtype=bool,
        count=len(universe_m4.referents),
    )
    for term in sample_terms
}
truth_to_term = {vector.tobytes(): term for term, vector in vectors.items()}

pairs = []
seen = set()
for term, vector in vectors.items():
    complement = truth_to_term.get((~vector).tobytes())
    if complement is not None:
        pair = tuple(sorted((term, complement)))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)

indexed = variant_2k.set_index("expression")
metric_names = [
    "majorant",
    "complement_dual",
    "two_sided_mean",
    "two_sided_min",
    "two_sided_max",
    "violation_rate",
]
pair_rows = []
detail_rows = []
for metric in metric_names:
    pair_maxima = []
    for q_term, not_q_term in pairs:
        q_row = indexed.loc[q_term]
        nq_row = indexed.loc[not_q_term]
        errors = [
            abs(q_row[f"{metric}_right_upward"] - nq_row[f"{metric}_right_downward"]),
            abs(q_row[f"{metric}_left_upward"] - nq_row[f"{metric}_left_downward"]),
            abs(q_row[f"{metric}_right_downward"] - nq_row[f"{metric}_right_upward"]),
            abs(q_row[f"{metric}_left_downward"] - nq_row[f"{metric}_left_upward"]),
        ]
        pair_maxima.append(max(errors))
        detail_rows.append(
            {
                "metric": metric,
                "Q": q_term,
                "not_Q": not_q_term,
                "max_mirror_error": max(errors),
            }
        )
    pair_rows.append(
        {
            "metric": metric,
            "pairs tested": len(pairs),
            "exact mirrored pairs": sum(error < 1e-12 for error in pair_maxima),
            "mean pair error": np.mean(pair_maxima),
            "maximum error": np.max(pair_maxima),
        }
    )

mirror_summary = pd.DataFrame(pair_rows)
mirror_summary[["mean pair error", "maximum error"]] = mirror_summary[
    ["mean pair error", "maximum error"]
].round(6)
mirror_summary
"""),
    code(r"""
# The ten largest failures under the manuscript-era measure.
(
    pd.DataFrame(detail_rows)
    .query("metric == 'majorant'")
    .sort_values("max_mirror_error", ascending=False)
    .head(10)
    .assign(max_mirror_error=lambda frame: frame["max_mirror_error"].round(3))
)
"""),
    md(r"""
The result is decisive:

- The manuscript majorant measure mirrors exactly for only **2 of 44** pairs.
- The complement-dual, all two-sided variants, and direct violation rate mirror
  for **44 of 44** pairs.

This establishes a property of the metrics. It does not by itself decide which
property should define the scientific construct.
"""),
    md(r"""
## 9. What changes over all 2,000 expressions?

`degree` below is the maximum of RU, LU, RD, and LD, matching the manuscript's
aggregation. The AUC correlation uses the 1,795 trained/converged expressions
with available outcomes.

The correlation is descriptive, **not a criterion for choosing a metric**. A
metric should be chosen from its semantics and invariances, not because it
maximizes association with the dependent variable.
"""),
    code(r"""
from scipy.stats import pearsonr

runs = pd.read_csv(
    LEARN_ROOT / "outputs/combined_runs_AOC_monotonicity_updated.csv"
)
auc_by_expression = (
    runs.loc[runs["training"] == True]
    .groupby("expression")["val_loss_step_AOC"]
    .mean()
    .rename("mean_AUC")
)

distribution_rows = []
for metric in metric_names:
    degree = indexed[f"{metric}_degree"].rename("degree")
    joined = pd.concat([degree, auc_by_expression], axis=1).dropna()
    distribution_rows.append(
        {
            "metric": metric,
            "all-2k mean degree": degree.mean(),
            "all-2k SD": degree.std(),
            "AUC n": len(joined),
            "Pearson r(degree, AUC)": pearsonr(
                joined["degree"], joined["mean_AUC"]
            ).statistic,
        }
    )

distribution_summary = pd.DataFrame(distribution_rows)
distribution_summary[
    ["all-2k mean degree", "all-2k SD", "Pearson r(degree, AUC)"]
] = distribution_summary[
    ["all-2k mean degree", "all-2k SD", "Pearson r(degree, AUC)"]
].round(3)
distribution_summary
"""),
    code(r"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
for ax, metric in zip(axes.flat, metric_names):
    ax.hist(indexed[f"{metric}_degree"], bins=np.linspace(0, 1, 26), color="#4472C4", alpha=0.85)
    ax.set_title(metric.replace("_", " "))
    ax.set_xlabel("degree")
    ax.set_ylabel("expressions")
fig.suptitle("Degree distributions for the 2,000 sampled expressions", fontsize=14)
fig.tight_layout()
plt.show()
"""),
    md(r"""
The direct violation score clusters near 1 because its denominator includes
all comparable pairs. The entropy variants use more of the 0--1 range.
`two_sided_max` has the strongest raw AUC correlation here, but selecting it
for that reason would be outcome-driven. Its substantive meaning is permissive:
either the majorant or minorant can make a direction look monotone.
"""),
    md(r"""
## 9b. A boundary artifact in the mean — and why it does not rule out every
two-sided score

`or(subset_eq(A,B), subset_eq(B,A))` is the **comparability predicate**: it is
true exactly when A and B are comparable under inclusion. It is not monotone
in any of the four directions.

The table below distinguishes the manuscript's M6/X6 universe (the source of
the quoted Table 4 value `0.201`) from the M4/X4 universe used for the 2,000
learned expressions. The closure feature includes the point itself, so it is
exactly constant 1 for this predicate in both universes. Its entropy score is
therefore 0. The interior score is nonzero partly because an interior agrees
with Q automatically at order boundaries.
"""),
    code(r"""
def comparability_diagnostic(universe, by_term, universe_label):
    term = "or(subset_eq(A, B), subset_eq(B, A))"
    q = np.fromiter(
        (by_term[term].meaning.mapping[ref] for ref in universe.referents),
        dtype=int,
        count=len(universe.referents),
    )
    all_models = universe.binarize_referents(mode="set_vectors_w_padding")
    ref_a = universe.binarize_referents(mode="A")
    not_q = 1 - q
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        closure = measures.upward_monotonicity_entropy(
            all_models, ref_a, q, cfg, False
        )
        interior = measures.upward_monotonicity_entropy(
            all_models, ref_a, not_q, cfg, True
        )
    closure_feature = measures.get_true_predecessors(
        all_models, ref_a, q, False
    )
    return {
        "universe": universe_label,
        "closure feature = 1": closure_feature.mean(),
        "C_up score": closure,
        "I_up score": interior,
        "two-sided mean": (closure + interior) / 2,
        "two-sided min": min(closure, interior),
    }


comparability = pd.DataFrame(
    [
        comparability_diagnostic(universe_m6, by_term_m6, "M6/X6 (Table 4)"),
        comparability_diagnostic(universe_m4, by_term_m4, "M4/X4 (2k sample)"),
    ]
)
comparability.round(4)
"""),
    md(r"""
The mean inherits half of the interior's boundary-driven information: `0.201`
on M6/X6 and `0.216` on M4/X4. Calling that value “monotonicity” is hard to
defend for this predicate.

But this is a problem with the **mean**, not a proof that symmetric feature
representations are impossible. The two-sided minimum gives 0 whenever either
the closure or interior provides no directional evidence. Here it gives 0 in
all four directions, while also passing all 44 complement-mirror tests.
"""),
    md(r"""
## 10. Which measure is most in line with the intuition?

There are two defensible intuitions:

### Intuition 1: “minimal monotone extension”

If the construct is exactly the one written in the manuscript -- make the
smallest truth-set expansion needed for monotonicity, then ask how informative
that expansion is -- the **manuscript majorant measure is correct**. Downward
must use the downward closure. The cost is lack of truth-label symmetry.

### Intuition 2: “degree should not depend on which side is called true”

If `Q` and `not Q` should have equal mirrored degree because they share the
same boundary with labels exchanged, a closure-only score is unsuitable. This
intuition is especially relevant here because sigmoid + binary cross-entropy
learning is symmetric under label complementation.

The complement-dual alternative satisfies this second intuition, but mixes
majorant and minorant constructions across directions. A symmetric combination
of both features avoids that directional asymmetry:

- **mean:** both features contribute additively, including boundary artifacts;
- **max:** either feature can make the score high;
- **min:** both must support the score, so one uninformative feature vetoes it.

For the intuitions at issue here, the minimum is the most conservative choice.

There is also a feature-free alternative: define degree from the minimum
weighted Hamming distance between Q and any exactly monotone truth set
(isotonic Boolean regression). That distance is 0 exactly for monotone
functions, is preserved by order reversal, and maps upward(Q) to
downward(not Q) under complementation. It avoids the existential-feature
collapse entirely. Its unresolved design choice is normalization: the raw
number of required label edits must be scaled before it becomes a 0--1 degree.
"""),
    md(r"""
## 11. Recommendation for the manuscript

### What I would report

1. **Published-result reproduction:** retain the manuscript-era majorant scores
   and describe them accurately as closure / least-majorant entropy. Do not call
   their lack of complement invariance an implementation bug.
2. **Symmetric revised candidate:** investigate **two-sided minimum entropy**.
   It gives exactly monotone functions 1, preserves order duality and complement
   invariance, mirrors all 44 tested complement pairs, and gives the
   comparability predicate 0 in all four directions.
3. **Sensitivity analysis:** report the majorant, two-sided mean/max, and direct
   violation rate. This exposes whether conclusions depend on requiring both
   approximations to agree, averaging them, or accepting the better one.
4. **Do not use the complement-dual replacement as the sole “corrected”
   measure.** It enforces the desired mirror equation by an
   upward-majorant/downward-minorant asymmetry.

### Why minimum rather than mean or max?

`max` rewards a direction when either repair looks informative. `mean` still
allows one feature to raise the score when the other contains no directional
information. `min` asks the conservative question: **how strong is the weaker
of the expansion-based and contraction-based signals?** Thus boundary agreement
from an interior cannot rescue a zero closure score.

This is a principled candidate, not a theorem that every intuitive calibration
problem is solved. A scalar degree still requires choices about weighting,
finite-universe boundaries, and what numerical value “maximally nonmonotone”
should receive.

If those calibration choices remain troubling, minimum edit distance to the
set of monotone functions is the cleaner next family to investigate; it gives
up the manuscript's feature-predictability interpretation in exchange for a
direct “how many truth values must change?” interpretation.
"""),
    md(r"""
## 12. Final conceptual map

| Question | Answer |
|---|---|
| Did the manuscript use a nonsensical calculation? | No. It used the natural order-dual closure/majorant construction. |
| Does that calculation have complement mirror errors? | Yes, structurally, because negation swaps closure and interior. |
| Did `down(Q)=up(not Q)` simply fix the implementation? | No. It selected a different, complement-dual metric. |
| Should Table 4 have RU=LD within every row? | Not as a general law. Those coincidences depend on the semantics of the listed expressions and the chosen approximation. |
| Are order duality and complement invariance incompatible? | Only for a one-sided closure-only score. Two-sided entropy and direct violations can have both. |
| Does the mean's comparability score show that every two-sided feature measure fails? | No. The two-sided minimum gives this predicate 0 while preserving both symmetries. |
| Best revised entropy candidate among those tested? | Two-sided minimum, with broader construct validation and mean/max/violation-rate sensitivity checks. |

The practical lesson is to state the desired invariances **before** choosing
both the approximation operators and their aggregation. Entropy cannot repair
a symmetry absent from the features, and a symmetric aggregation can still be
poorly calibrated if it rewards only one side.
"""),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "altk",
            "language": "python",
            "name": "altk",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
