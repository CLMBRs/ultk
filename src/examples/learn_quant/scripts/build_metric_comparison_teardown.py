"""Build the criterion-first comparison of graded monotonicity measures."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "notebooks/metric_comparison_teardown.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(r"""
# Graded monotonicity measures: what each calculates, and what can go wrong

This notebook compares the measures **by the question each one answers**. That
qualification matters: a measure is not "better" merely because it gives a
larger or smaller number.

The central conclusions are:

1. Exact monotonicity is a yes/no property. Every graded measure below adds a
   separate notion of *closeness*.
2. The manuscript majorant, two-sided minimum, and pair-counting scores do not
   estimate the same quantity.
3. A nonzero score for a non-exact predicate is not automatically a false
   positive. It is partial credit, unless the score was explicitly advertised
   as a binary test.
4. There is no criterion-free example where one measure is "definitively
   worse." There are examples where a measure is worse **for a stated goal**.

The notebook first calculates every ingredient on a three-point chain, then
uses four M6/X6 benchmark meanings to expose the tradeoffs.
"""),
    md(r"""
## 1. Start with the property, not the score

Fix one argument of a quantifier and let `x <= y` mean that the other argument
has grown by set inclusion.

- **Exact upward monotonicity:** if `Q(x)=1` and `x <= y`, then `Q(y)=1`.
- **Exact downward monotonicity:** if `Q(y)=1` and `x <= y`, then `Q(x)=1`.

These definitions quantify over **every** comparable pair. A single
counterexample defeats exact monotonicity.

A graded score must answer an additional question, for example:

| Goal | Natural question |
|---|---|
| Exactness detection | Is there any violating pair? |
| One-sided approximation | How informative is the least monotone majorant? |
| Symmetric approximation | Do both a majorant and a minorant represent `Q` well? |
| Typical-case preservation | How often does truth survive under sampled expansions? |
| Minimal repair | How much probability mass must be relabeled to make `Q` monotone? |

Different goals can rank the same predicate differently without any
implementation being faulty.
"""),
    md(r"""
## 2. The four monotone approximations

For Boolean `Q` on a partial order:

\[
\begin{aligned}
C_\uparrow Q(x) &= \max_{y\le x} Q(y)
&&\text{least upward-monotone majorant}\\
I_\uparrow Q(x) &= \min_{y\ge x} Q(y)
&&\text{greatest upward-monotone minorant}\\
C_\downarrow Q(x) &= \max_{y\ge x} Q(y)
&&\text{least downward-monotone majorant}\\
I_\downarrow Q(x) &= \min_{y\le x} Q(y)
&&\text{greatest downward-monotone minorant.}
\end{aligned}
\]

A majorant repairs monotonicity by changing some `0`s to `1`s. A minorant
repairs it by changing some `1`s to `0`s.

The implementation phrase "has a true predecessor" is exactly
`C_up(Q)(x)`. There is not a second independent predictor built after the
closure. Closure is idempotent: `C_up(C_up(Q)) = C_up(Q)`.
"""),
    code(r"""
import numpy as np
import pandas as pd
from pathlib import Path

# Three points ordered x0 <= x1 <= x2.
q = np.array([0, 1, 0], dtype=int)

def closure_up(values):
    return np.maximum.accumulate(values)

def interior_up(values):
    return np.minimum.accumulate(values[::-1])[::-1]

def closure_down(values):
    return np.maximum.accumulate(values[::-1])[::-1]

def interior_down(values):
    return np.minimum.accumulate(values)

toy = pd.DataFrame(
    {
        "point": ["x0", "x1", "x2"],
        "Q": q,
        "C_up(Q)": closure_up(q),
        "I_up(Q)": interior_up(q),
        "C_down(Q)": closure_down(q),
        "I_down(Q)": interior_down(q),
    }
)
toy
"""),
    md(r"""
The middle-only predicate is neither upward nor downward monotone.

- `C_up(Q) = (0,1,1)`: change the final `0` to `1`.
- `I_up(Q) = (0,0,0)`: remove the isolated true point.
- `C_down(Q) = (1,1,0)`: change the initial `0` to `1`.
- `I_down(Q) = (0,0,0)`: again remove the isolated true point.

The closure retains some structure; the interior retains none. This simple
case will distinguish the variants.
"""),
    md(r"""
## 3. The entropy score

For approximation `F`, the code calculates normalized mutual information:

\[
s(Q,F)
=1-\frac{H(Q\mid F)}{H(Q)}
=\frac{I(Q;F)}{H(Q)}.
\]

Interpretation:

- `1`: `F` determines `Q` perfectly.
- `0`: observing `F` gives no information about `Q`.
- intermediate values: reduction in uncertainty about `Q`.

This is **not** percent truth preservation, percent agreement, or percent of
violations avoided. It depends on the complete `2 x 2` table of `(Q,F)` and on
the distribution over situations.
"""),
    code(r"""
def entropy(bits):
    p = np.asarray(bits, dtype=float).mean()
    if p in (0.0, 1.0):
        return 0.0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def entropy_score(target, feature):
    target = np.asarray(target, dtype=int)
    feature = np.asarray(feature, dtype=int)
    h_target = entropy(target)
    if h_target == 0:
        return 1.0
    h_cond = 0.0
    for value in (0, 1):
        selected = target[feature == value]
        if len(selected):
            h_cond += len(selected) / len(target) * entropy(selected)
    return 1 - h_cond / h_target

toy_scores = {
    name: entropy_score(q, toy[name].to_numpy())
    for name in ["C_up(Q)", "I_up(Q)", "C_down(Q)", "I_down(Q)"]
}
pd.Series(toy_scores, name="entropy score").round(3).to_frame()
"""),
    md(r"""
For the upward direction, the majorant score is about `0.274`, while the
interior score is `0`. Neither number is an error:

- `0.274` says the upward closure retains some information about `Q`.
- `0` says the upward interior is constant and therefore uninformative.

The normalized score is well suited to an **informativeness-of-approximation**
question. Its main limitations are that it is sensitive to truth prevalence,
the chosen universe distribution, and how balanced the approximation feature
is.
"""),
    md(r"""
## 4. How the entropy variants combine the ingredients

Write `s(Q,F)` as `s(F)` below:

| Variant | Upward score | Downward score | Structural properties |
|---|---|---|---|
| Majorant | `s(C_up Q)` | `s(C_down Q)` | Order-dual; not generally complement-invariant |
| Complement-dual | `s(C_up Q)` | `s(I_down Q)` | Complement-invariant; asymmetric closure/interior recipe |
| Two-sided mean | mean of `s(C)` and `s(I)` | mean of `s(C)` and `s(I)` | Order-dual and complement-invariant |
| Two-sided minimum | minimum of `s(C)` and `s(I)` | minimum of `s(C)` and `s(I)` | Order-dual and complement-invariant |

Negation swaps closure and interior:

\[
C_\uparrow(\neg Q)=\neg I_\downarrow(Q).
\]

That identity explains the tradeoff. A closure-only score can use the same
construction after reversing the order, or it can mirror truth-value
complements, but it cannot do both without also incorporating interiors.
"""),
    code(r"""
up_closure = toy_scores["C_up(Q)"]
up_interior = toy_scores["I_up(Q)"]
pd.Series(
    {
        "majorant": up_closure,
        "two-sided mean": (up_closure + up_interior) / 2,
        "two-sided minimum": min(up_closure, up_interior),
    },
    name="toy upward score",
).round(3).to_frame()
"""),
    md(r"""
This toy example shows the aggregation choice clearly:

- **Majorant** preserves the one-sided approximation signal.
- **Mean** discounts it because only one repair direction is informative.
- **Minimum** returns zero because it requires *both* repair directions to be
  informative.

Therefore the minimum is conservative, not automatically more correct. It is
better if the construct means "supported from both sides"; it is worse if the
construct means "quality of the least monotone majorant."
"""),
    md(r"""
## 5. Two pair-counting measures that must not be conflated

The repository contains two related but distinct diagnostics.

### A. Unconditional violation-rate score

\[
V_\uparrow
=1-\frac{\#\{x<y:Q(x)=1,Q(y)=0\}}
{\#\{x<y\}}.
\]

Its denominator is **all** proper comparable pairs. It has clean order and
complement symmetries, but violations can be diluted by a very large number of
irrelevant nonviolating pairs.

### B. Conditional truth-preservation score

\[
P_\uparrow
=1-\frac{\#\{x<y:Q(x)=1,Q(y)=0\}}
{\#\{x<y:Q(x)=1\}}.
\]

This asks: among expansions whose source is true, how often does truth survive?
It is easy to interpret as a conditional frequency, but it does **not**
generally satisfy complement mirror symmetry because complementing `Q` changes
the conditioning population.

The benchmark CSV columns named `preservation_*` use the second denominator.
They are not the unconditional violation-rate score.
"""),
    code(r"""
# The three proper ordered pairs are x0<x1, x0<x2, and x1<x2.
pairs = [(0, 1), (0, 2), (1, 2)]
violations = [(i, j) for i, j in pairs if q[i] == 1 and q[j] == 0]
eligible = [(i, j) for i, j in pairs if q[i] == 1]

pair_scores = pd.Series(
    {
        "unconditional violation-rate score": 1 - len(violations) / len(pairs),
        "conditional truth-preservation": 1 - len(violations) / len(eligible),
    }
)
pair_scores.round(3).to_frame("toy upward score")
"""),
    md(r"""
For the same predicate, the two pair scores are `2/3` and `0`. The difference
comes entirely from the denominator. This is why a label such as "violation
rate" is incomplete unless the eligible population is stated.
"""),
    md(r"""
## 6. Four M6/X6 cases: what the disagreements really mean

The following table comes from the exhaustive semantic benchmark. `Preservation`
is the **conditional** pair score just defined.
"""),
    code(r"""
ROOT = Path.cwd()
if not (ROOT / "analysis").is_dir():
    ROOT = ROOT.parent

benchmark = pd.read_csv(ROOT / "analysis/semantic_monotonicity_benchmark.csv")
names = [
    "A differs from B",
    "A and B are incomparable",
    "exactly five overlap",
    "A and B are both empty or both nonempty",
]
rows = []
for _, row in benchmark[benchmark.meaning.isin(names)].iterrows():
    direction = "RU"
    closure = row[f"majorant_{direction}"]
    interior = np.clip(2 * row[f"mean_{direction}"] - closure, 0.0, 1.0)
    rows.append(
        {
            "meaning": row.meaning,
            "prevalence": row.truth_prevalence,
            "majorant": closure,
            "interior": interior,
            "two-sided mean": row[f"mean_{direction}"],
            "two-sided minimum": row[f"min_{direction}"],
            "conditional preservation": row[f"preservation_{direction}"],
        }
    )
cases = pd.DataFrame(rows).set_index("meaning")
cases.round(3)
"""),
    md(r"""
### Case A: `A != B`

Conditional preservation is `0.984`, but the majorant and two-sided minimum are
about `0.013`.

This is not evidence that the minimum is damaged by the interior: the interior
score is about `0.341`, so the **majorant is the bottleneck**. The upward closure
is almost constant true because nearly every false equality point has a true
predecessor. A nearly constant feature carries little information, hence the
small entropy score.

- If the goal is typical-case truth preservation, `0.984` is informative.
- If the goal is informativeness of the least upward majorant, `0.013` is
  coherent.
- If the goal is exact monotonicity, both summaries are secondary: a violating
  pair exists, so the exact answer is simply "no."

The earlier claim that the interior "collapses" the minimum in this example was
backwards.

### Case B: `A and B are incomparable`

The majorant is informative (`0.403`), while the interior is constant and has
score `0`; the two-sided minimum is therefore `0`.

This is the cleanest tradeoff:

- The majorant is better for one-sided monotone approximability.
- The minimum is doing exactly what its two-sided definition says: refusing
  partial credit when only one repair direction preserves information.

Calling the lost `0.403` "destroyed signal" assumes in advance that one-sided
signal is the target construct.

### Case C: `exactly five overlap`

The predicate is not exactly upward monotone, yet the two-sided minimum is
`0.610`. That is **not a false positive** unless `score > 0` was defined as a
test for exactness. On the finite six-element universe, the predicate is
concentrated near a boundary, and both approximations remain informative.

The legitimate concern is calibration: `0.610` cannot be read as "61% monotone,"
and the value may change with domain size and situation weighting.

### Case D: both sets empty or both nonempty

Conditional preservation is `0.998` because almost all eligible expansions in
the uniformly enumerated finite universe begin in the large "both nonempty"
region. The rare empty-set violation receives little weight.

The score is not mathematically wrong. It is unsuitable if the intended
criterion is worst-case sensitivity, where one counterexample should dominate.
"""),
    md(r"""
## 7. Defects and limitations, stated precisely

| Measure | What it genuinely measures | Main defect if used as a general "degree" |
|---|---|---|
| Exact pair check | Whether any violation exists | Binary; gives no graded similarity |
| Majorant entropy | Information about `Q` retained by a least monotone majorant | One-sided repair; lacks complement invariance; prevalence/distribution sensitive |
| Complement-dual entropy | Closure upward, complement-mirrored interior downward | Uses different repair semantics across order directions |
| Two-sided mean | Average information retained by adding and removing truths | A strong side can compensate for an uninformative side |
| Two-sided minimum | Information guaranteed by both repair directions | A zero on either side erases useful one-sided structure |
| Unconditional violation rate | Fraction of all comparable pairs that are not violations | Severe denominator dilution; dependent pairs are counted repeatedly |
| Conditional preservation | Truth survival among true-source expansions | Not complement-invariant; sensitive to the sampled eligible population |

All graded variants are sensitive to the finite universe and its probability
measure. None should be interpreted as an intrinsic, scale-free percentage of
monotonicity.
"""),
    md(r"""
## 8. Recommendation

For the manuscript, retain the **majorant entropy measure** because it matches
the stated minimal-monotone-extension construction and reproduces the published
numbers. Describe it as one-sided approximability, not as the uniquely correct
degree of monotonicity.

For future work:

1. Use the exhaustive pair check for categorical claims of exact monotonicity.
2. Report majorant and interior scores separately before combining them. This
   makes cases like incomparability immediately interpretable.
3. Use conditional preservation only when the scientific question is explicitly
   about typical truth-preserving expansions.
4. If one scalar must satisfy both order duality and complement invariance,
   two-sided minimum is a defensible conservative summary, but it is not yet a
   validated perceptual or learning-theoretic scale.
5. Evaluate minimum weighted distance to the nearest monotone Boolean function
   if the desired interpretation is "how much truth-value mass must be repaired?"

The pedagogical lesson is not that one measure wins. It is that each number must
be named after its estimand: **exactness, approximation information, pairwise
preservation, or repair distance**.
"""),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
    },
)
nbf.write(notebook, OUTPUT)
print(f"Wrote {OUTPUT}")
