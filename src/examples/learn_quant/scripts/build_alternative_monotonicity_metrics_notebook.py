"""Build the intuition benchmark for alternative monotonicity metrics."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "notebooks/alternative_monotonicity_metrics_benchmark.ipynb"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        r"""
# Which alternative monotonicity metrics match which intuitions?

This notebook implements and stress-tests the twelve proposals in the prompt on
a transparent finite universe. Its conclusion is deliberately not "metric X
wins." The proposals encode at least four different intuitions:

1. **logical exactness:** one counterexample defeats monotonicity;
2. **repair distance:** few truth-value changes should mean nearly monotone;
3. **local regularity:** one-element changes should usually have the right sign;
4. **simple organization:** a small number of boundaries may be cognitively easy.

A metric can fit one intuition and fail another. To avoid letting a metric
define its own success, the notebook preregisters several qualitative tests
before examining the scores.

## Before comparing metrics: what "one-sided" and "two-sided" mean

The manuscript's entropic measure is **one-sided in its choice of monotone
approximation**. This does **not** mean that the manuscript measures only upward
monotonicity: it computes right-upward, left-upward, right-downward, and
left-downward scores. For each tested direction, however, it compares the
quantifier with only the least monotone function lying **above** it.

For upward monotonicity, let $Q(x)\in\{0,1\}$ and let $x\leq y$ mean that
$y$ enlarges the argument under test. Two canonical upward-monotone
approximations bracket $Q$:

$$
C^\uparrow Q(x)=1
\quad\text{iff}\quad
\text{some }y\leq x\text{ has }Q(y)=1,
$$

$$
I^\uparrow Q(x)=1
\quad\text{iff}\quad
\text{every }y\geq x\text{ has }Q(y)=1.
$$

- **Majorant / upward closure $C^\uparrow Q$:** the smallest
  upward-monotone function greater than or equal to $Q$. It repairs violations
  only by changing false points to true.
- **Minorant / upward interior $I^\uparrow Q$:** the largest
  upward-monotone function less than or equal to $Q$. It repairs violations
  only by changing true points to false.

The manuscript score used in this notebook as the baseline is

$$
\operatorname{mon}_{\mathrm{majorant}}(Q)
=
1-\frac{H(Q\mid C^\uparrow Q)}{H(Q)}
=
\frac{I(Q;C^\uparrow Q)}{H(Q)}.
$$

It is therefore called **one-sided majorant entropy**: only the majorant side of
the bracket contributes. The notebook's **two-sided entropy** variants also
score the interior and combine the two values:

$$
s_C=\frac{I(Q;C^\uparrow Q)}{H(Q)},\qquad
s_I=\frac{I(Q;I^\uparrow Q)}{H(Q)}.
$$

The mean, minimum, and maximum variants use
$\operatorname{mean}(s_C,s_I)$, $\min(s_C,s_I)$, or
$\max(s_C,s_I)$, respectively. The benchmark emphasizes the **two-sided
minimum**, the conservative choice: a high score requires both the majorant and
the interior to preserve information about $Q$. Symmetrically combining both
sides also removes the majorant-only measure's asymmetry under Boolean
complementation.

### A four-point example

Read each row as the truth values along a chain $0<1<2<3$:

| Quantifier pattern | $Q$ | $C^\uparrow Q$ | $I^\uparrow Q$ | majorant score | interior score | two-sided minimum |
|---|---|---|---|---:|---:|---:|
| at least 2 | `0 0 1 1` | `0 0 1 1` | `0 0 1 1` | 1.000 | 1.000 | 1.000 |
| between 1 and 2 | `0 1 1 0` | `0 1 1 1` | `0 0 0 0` | 0.311 | 0.000 | 0.000 |
| even | `1 0 1 0` | `1 1 1 1` | `0 0 0 0` | 0.000 | 0.000 | 0.000 |

The bounded interval illustrates the substantive choice. Its majorant adds only
one true point, so the manuscript measure retains some information and scores
`0.311`. Its interior is constant false: no true point is safe under every
enlargement. The two-sided minimum therefore scores it `0`. Thus two-sidedness
is not merely a technical correction; it imposes a stricter intuition about
what should count as approximately monotone.

## Main findings

- Pairwise preservation and counterfactual robustness are the same estimand
  under uniform sampling of eligible comparable pairs.
- Edge preservation is the one-step version. It gives the clearest local
  interpretation and is a plausible **learnability hypothesis**, but this
  notebook does not establish a correlation with neural AUC.
- The nearest-monotone edit score is the cleanest global "minimal repair"
  measure and performs well on the preregistered near-monotone ladder.
- Closure inflation is a transparent one-sided repair cost; closure precision
  can be misleading for highly prevalent truth sets.
- Switch count measures boundary simplicity, not directional monotonicity.
- Chain inversions generalize pairwise violations with a chain-sampling
  weighting; they are not independent of pairwise comparison.
- The derivative-sign score conditions only on edges where truth changes. It
  captures directional transition purity, not violation frequency.
- Context profiles and directional profiles are valuable **representations**,
  not competing scalar definitions.
- Threshold and LoT scores test restricted representational hypotheses. They
  should not be sold as general monotonicity measures.
"""
    ),
    md(
        r"""
## 1. What is actually distinct among the twelve proposals?

| Prompt proposal | Implementation here | Status |
|---|---|---|
| 1. Pairwise violation | conditional truth preservation over all comparable pairs | distinct weighting |
| 2. Hasse-edge violation | conditional truth preservation over cover edges | local version of 1 |
| 3. Nearest monotone | exact minimum Hamming edits via an s-t minimum cut | distinct repair metric |
| 4. Closure cost | normalized closure inflation and closure precision | one-sided repair metrics |
| 5. Switch count | excess switches averaged over maximal chains | nondirectional simplicity metric |
| 6. Chain score | inversions averaged over maximal chains | chain-weighted pair violations |
| 7. Robustness | survival by expansion distance `k`, then equal-`k` average | proposal 1 plus a sampling policy |
| 8. Derivative sign | favorable / all nonzero one-edge derivatives | local transition-sign metric |
| 9. Context sensitive | mean, SD, and coverage across fixed-argument contexts | profile, not one scalar |
| 10. Direction profile | four-vector plus max, mean, and purity | aggregation choice |
| 11. Threshold fit | best accuracy among simple cardinality thresholds | restricted model-family fit |
| 12. LoT fit | exception-code proxy based on minimum edits | **proxy only**, not a full LoT/MDL model |

The final row requires special caution. A real LoT score needs a specified
grammar, rule probabilities or code lengths, and inference over expressions.
Calling an exception count "LoT" would overstate what was measured, so the
notebook reports it as a transparent proxy.

## Data provenance: what generates what

All source and generated artifacts live in this repository:

1. `alternative_monotonicity_metrics.py` implements the metric formulas.
2. `scripts/alternative_monotonicity_metric_benchmark.py` constructs the
   exhaustive M4/X4 meaning suite, calls those formulas, and writes
   `analysis/alternative_monotonicity_metric_benchmark.csv`.
3. `scripts/build_alternative_monotonicity_metrics_notebook.py` generates this
   notebook.
4. This executed notebook reads the committed CSV for the M4/X4 comparison and
   directly calls `cardinality_lattice_scores` for the exact `n=2...100`
   parity-size sweep.

To rebuild the data and notebook from `src/examples/learn_quant/`:

```bash
python scripts/alternative_monotonicity_metric_benchmark.py
python scripts/build_alternative_monotonicity_metrics_notebook.py
jupyter nbconvert --to notebook --execute \
  notebooks/alternative_monotonicity_metrics_benchmark.ipynb \
  --output alternative_monotonicity_metrics_benchmark.ipynb
```
"""
    ),
    code(
        r"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "analysis").is_dir():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
import alternative_monotonicity_metrics as alt_metrics

CSV = ROOT / "analysis/alternative_monotonicity_metric_benchmark.csv"
df = pd.read_csv(CSV)

METRICS = {
    "majorant_entropy": "manuscript majorant entropy",
    "two_sided_min": "two-sided entropy minimum",
    "pairwise_preservation": "all-pair preservation",
    "edge_preservation": "edge preservation",
    "unconditional_pair": "unconditional pair score",
    "nearest_monotone_edit": "nearest-monotone edit score",
    "best_cardinality_threshold_repair": "best cardinality-threshold repair score",
    "closure_inflation": "closure inflation",
    "closure_precision": "closure precision",
    "chain_switch": "chain switch simplicity",
    "chain_inversion": "chain inversion",
    "equal_distance_robustness": "equal-distance robustness",
    "derivative_sign": "derivative sign",
    "context_mean": "mean context preservation",
    "best_simple_threshold": "best simple-threshold fit",
    "exception_code_proxy": "exception-code proxy",
}

print(
    f"{df.meaning.nunique()} meanings x {df.direction.nunique()} directions "
    f"= {len(df)} directional cases"
)
"""
    ),
    md(
        r"""
## 2. Benchmark universe and meanings

The universe is M4/X4: every ordered pair `(A,B)` of subsets of a four-element
domain, giving `16 x 16 = 256` situations. All displayed scores are exhaustive,
not Monte Carlo estimates.

The suite includes:

- textbook quantifiers (`all`, `some`, `no`, `not all`, `most`);
- monotone overlap and cardinality thresholds;
- organized nonmonotone bands (`exactly two`, `one through three`);
- alternating parity predicates;
- one deliberately near-monotone predicate with a single exceptional
  `(A,B)` situation;
- equality, difference, comparability, incomparability, and boundary meanings.

M4/X4 is used because the exact nearest-monotone edit score is then easy to audit.
Its scores are finite-universe properties, not claims about all domain sizes.
"""
    ),
    code(
        r"""
df[["family", "meaning", "formula"]].drop_duplicates().reset_index(drop=True)
"""
    ),
    md(
        r"""
## 3. Formulas and denominator choices

For an oriented strict order `x < y`, an upward violation is `Q(x)=1`,
`Q(y)=0`.

### All-pair preservation

$$
1-\frac{\#\{x<y:Q(x)=1,Q(y)=0\}}
        {\#\{x<y:Q(x)=1\}}.
$$

### Edge preservation

The same ratio, restricted to cover edges `x \prec y`. This is exactly "I
added one element and truth broke."

### Unconditional pair score

$$
1-\frac{\#\text{violations}}{\#\text{all comparable pairs}}.
$$

This has complement mirror symmetry but can dilute violations with irrelevant
pairs.

### Nearest-monotone edit score (general)

The numerator is the exact minimum number of truth values that must be flipped.
The notebook normalizes by `min(#true,#false)`, the cost of replacing `Q` by
the closer constant function:

$$
1-\frac{\min_{g\in Mon}d_H(Q,g)}
        {\min(|Q|,|\neg Q|)}.
$$

Thus `1` is exact and `0` means no better than a constant. The minimization is
solved as a minimum cut; order edges receive effectively infinite capacity.

The word **nearest** means nearest among **all upward-monotone Boolean
functions on the finite lattice**. It does not restrict the repaired function
to a cardinality threshold.

### Closure scores

$$
1-\frac{|C_\uparrow Q|-|Q|}{|\neg Q|}
\quad\text{and}\quad
\frac{|Q|}{|C_\uparrow Q|}.
$$

The first asks what fraction of currently false situations need **not** be
added. The second asks what fraction of closure truths were already true.

### Chains and derivatives

- Switch simplicity penalizes switches beyond one along maximal chains, but
  ignores whether the one switch has the correct direction.
- Chain inversion counts all `1...0` pairs within each maximal chain and then
  weights chains equally. A binary chain of length `L` has at most
  `floor(L^2 / 4)` such pairs, attained by a balanced block of ones followed by
  zeros; this is the denominator used here.
- Derivative sign is the fraction of nonzero cover-edge changes that are
  favorable (`0 -> 1` rather than `1 -> 0`).
"""
    ),
    md(
        r"""
## 4. Intuition tests: what each test actually asks

Before comparing graded scores, we need independent judgments about what should
happen in easy cases. These judgments are the **tests**. The metric values are
then compared with them; the metrics do not supply their own answer key.

### Test 1: exact monotonicity must receive the top score

Here **endpoint** means the top endpoint, `1`, of a score ranging from `0` to
`1`. It does not mean an endpoint of the set lattice.

An **independently exact direction** is a direction proven monotone by directly
checking the logical condition over every comparable pair, without consulting
any graded metric:

$$
Q(x)=1\ \&\ x\le y \quad\Longrightarrow\quad Q(y)=1.
$$

Example: `all A are B` is exactly right-upward. If `A` is already a subset of
`B`, enlarging `B` cannot make that false. Therefore every candidate advertised
as a directional monotonicity score should return exactly `1` for the RU
direction of `all`.

The scorecard column **all exact directions = 1** asks whether this holds for
every meaning/direction independently verified as exact in the benchmark.

### Test 2: a non-exact direction must not receive the top score

Now reverse the check. If direct pair enumeration finds even one true-to-false
counterexample, that direction is not exactly monotone. A graded metric may
still give partial credit such as `0.8`; that is its purpose. But a score of
exactly `1` would make the non-exact case indistinguishable from an exact one.

Example: `B has even cardinality` is not right-upward. Starting with an even
`B` and adding one element produces an odd `B`, changing truth from `1` to `0`.
Thus a directional monotonicity metric should give it a value below `1`.

The scorecard column **non-exact directions < 1** reports the *proportion* of
non-exact benchmark directions that stay below `1`. A value of `1.000` means
all passed this test. This test does **not** require every non-exact case to
score `0`.

### Test 3: the right-upward near-monotone ladder

This test asks whether a metric distinguishes four increasingly irregular
truth patterns:

1. `|B| >= 2`: exactly right-upward; once true, it stays true.
2. `|B| >= 2` with one exceptional false model: almost the same monotone
   threshold, but with one deliberately introduced violation.
3. `1 <= |B| <= 3`: an organized interval. It first turns on, then turns off at
   the upper boundary, so it is clearly nonmonotone but not alternating.
4. `|B| is even`: it flips truth at every one-element step.

The hypothesized ranking is therefore:

$$
\text{exact threshold} >
\text{one exception} >
\text{bounded interval} >
\text{parity}.
$$

This ordering is a stated geometric/learnability intuition, **not a theorem of
monotonicity**. The scorecard column **ladder comparisons passed** counts the
three adjacent inequalities that hold. `3` means the full ranking is recovered;
`2` means one pair tied or reversed.

### Test 4: complement-mirror consistency

Negating a predicate exchanges truth and falsity. It also exchanges upward and
downward violations. A complement-invariant score should therefore satisfy,
for the same argument:

$$
RU(Q)=RD(\neg Q),\qquad RD(Q)=RU(\neg Q),
$$

and likewise for `LU`/`LD`.

Example: `some A are B` and `no A are B` are exact complements. The RU score of
`some` should equal the RD score of `no` if the metric is intended to ignore
which truth label is called positive.

The scorecard reports the largest absolute mismatch across the listed
complement pairs. `0` means perfect mirror consistency. This property is
desirable for some scientific questions, but a conditional score such as
"preservation among true sources" can coherently fail it because complementation
changes the conditioning population.

### Test 5: can we explain why the score is high or low?

This is a transparency test rather than a single number. For example:

- Was the score high because violations were rare among all comparable pairs?
- Because only one-element transitions were regular?
- Because few truth values needed repair?
- Because most fixed-`A` contexts were easy?
- Because the meaning fit a restricted threshold family?

The later pair, distance, context, and threshold decompositions answer these
questions. We do not collapse transparency into a subjective pass/fail total.
"""
    ),
    code(
        r"""
ladder_names = [
    "B has at least two elements",
    "B threshold with one exception",
    "B has one to three elements",
    "B has even cardinality",
]
ladder = (
    df[(df.direction == "RU") & df.meaning.isin(ladder_names)]
    .set_index("meaning")
    .loc[ladder_names, list(METRICS)]
    .T
)
ladder.index = [METRICS[name] for name in ladder.index]
ladder.round(3)
"""
    ),
    md(
        r"""
The ladder is highly diagnostic:

- Nearest-monotone edit score, closure inflation, chain switches, and the exception-code proxy
  express the intended spacing well.
- Edge preservation also ranks the examples correctly and gives the
  near-monotone exception `0.984`.
- Unconditional pairs and closure precision rank correctly but compress severe
  nonmonotonicity upward.
- Chain inversion and derivative sign tie the interval and parity predicates.
  They see different organizations but the same aggregate balance under this
  weighting.
- Best simple-threshold fit tests only a small hand-built feature family. A high
  score means simple threshold representability, not monotonicity in general.
"""
    ),
    md(
        r"""
## 4b. Larger model universes: does `|B| is even` approach zero?

Yes for some measures, but **not for all of them**.

The M4/X4 table has 256 `(A,B)` situations. Here we sweep domain sizes from
`n=2` through `n=100`. The number of full situations would be
`4^n`, so explicit enumeration quickly becomes impossible.

For a predicate depending only on `|B|`, every fixed `A` context is identical.
Repeating the same B-lattice once for every A multiplies every relevant count by
the same factor. We can therefore calculate exact right-upward scores from the
`n+1` cardinality layers, weighting layer `k` by `C(n,k)`. This is an exact
compression of the full uniform M_n/X_n calculation for the displayed
right-upward metrics, not a sample.

One terminology correction: increasing `n` enlarges the **model universe**, not
the meaning space. The meaning remains `Q(B) = 1` iff `|B|` is even.
"""
    ),
    code(
        r"""
sizes = list(range(2, 101))
parity_rows = []
for n in sizes:
    scores = alt_metrics.cardinality_lattice_scores(
        [cardinality % 2 == 0 for cardinality in range(n + 1)]
    )
    parity_rows.append({"n": n, **scores})
parity_scaling = pd.DataFrame(parity_rows).set_index("n")

# Compute the general min-cut score where explicit subset-lattice optimization
# remains practical. Keep the same column at larger n, but mark it unavailable.
parity_scaling["nearest_monotone_edit"] = np.nan
for n in range(2, 11):
    subset_bits = np.arange(1 << n, dtype=np.int64)
    cardinalities = np.asarray(
        [int(value).bit_count() for value in subset_bits]
    )
    parity = cardinalities % 2 == 0
    upward_edges = (
        (subset_bits[:, None] & subset_bits[None, :]) == subset_bits[:, None]
    ) & (cardinalities[None, :] == cardinalities[:, None] + 1)
    _, score = alt_metrics.nearest_monotone_edit(parity, upward_edges)
    parity_scaling.loc[n, "nearest_monotone_edit"] = score

# Use the same public names as the ladder table. Snake-case names remain only
# in implementation files and the generated CSV schema.
parity_scaling = parity_scaling.rename(columns=METRICS)

display_sizes = [2, 4, 6, 8, 10, 20, 50, 100]
parity_columns = [
    "manuscript majorant entropy",
    "two-sided entropy minimum",
    "all-pair preservation",
    "edge preservation",
    "unconditional pair score",
    "nearest-monotone edit score",
    "best cardinality-threshold repair score",
    "closure inflation",
    "closure precision",
    "chain switch simplicity",
    "chain inversion",
    "equal-distance robustness",
    "derivative sign",
]
parity_scaling.loc[display_sizes, parity_columns].T.round(3)
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

zero_or_decaying = {
    "manuscript majorant entropy": "majorant entropy",
    "two-sided entropy minimum": "two-sided minimum",
    "edge preservation": "edge preservation",
    "closure inflation": "closure inflation",
    "chain switch simplicity": "switch simplicity",
    "nearest-monotone edit score": "nearest-monotone edit score (n <= 10)",
    "best cardinality-threshold repair score": "best cardinality-threshold repair score",
}
nonzero_limits = {
    "all-pair preservation": "all-pair preservation",
    "unconditional pair score": "unconditional pair",
    "closure precision": "closure precision",
    "chain inversion": "chain inversion",
    "equal-distance robustness": "equal-distance robustness",
    "derivative sign": "derivative sign",
}

for column, label in zero_or_decaying.items():
    axes[0].plot(parity_scaling.index, parity_scaling[column], label=label)
for column, label in nonzero_limits.items():
    axes[1].plot(parity_scaling.index, parity_scaling[column], label=label)

axes[0].set_title("Scores that are zero or decay toward zero")
axes[1].set_title("Scores with nonzero large-n limits")
for ax in axes:
    ax.set(xlabel="domain size n", ylabel="right-upward score", ylim=(-0.03, 1.03))
    ax.legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        r"""
### How to read the result

Your zero intuition is exactly right for metrics that ask whether **one-element
growth consistently preserves truth** or whether upward closure is useful:

- majorant entropy = `0` for every `n`, because the empty set is true and its
  upward closure is therefore the constant-true function;
- two-sided minimum = `0` for every `n`;
- edge preservation = `0` for every `n`, because adding one element always
  flips even cardinality to odd;
- closure inflation = `0` for every `n`;
- switch simplicity = `0` for every `n` in this normalization, because parity
  switches at every cardinality step.

### Is cardinality-threshold repair the same as nearest-monotone edit?

No. Both rows now appear in both `ladder` and `parity_scaling`, under the same
names:

- `nearest-monotone edit score`;
- `best cardinality-threshold repair score`.

The **nearest-monotone edit score** in the M4/X4 benchmark minimizes repair
cost over *every* monotone Boolean function on the lattice. The large-`n` sweep
computes that same general min-cut score through `n=10`. Cells after `n=10` are
shown as `NA` because the notebook does not run the full `2^n`-node optimization
there.

The **best cardinality-threshold repair score** is computed at every displayed
size, but minimizes only over:

$$
g_t(B)=1\quad\text{iff}\quad |B|\ge t.
$$

Because threshold functions are only a subset of all monotone functions, this
restricted score can never exceed the general nearest-monotone edit score for
the same normalization. The two scores happen to agree for the checked parity
cases (`n <= 12`, with regression tests through `n=6`), but that empirical
agreement does not make their definitions identical.

The best cardinality-threshold repair score approaches zero slowly:
`0.375` at `n=4`, `0.246` at `n=10`, and `0.080` at `n=100`. This is the
large-domain result closest to "parity becomes arbitrarily far from a monotone
threshold."

Other scores do **not** approach zero:

- all-pair preservation approaches `0.5`;
- unconditional pair score approaches `0.75`;
- closure precision, chain inversion, equal-distance robustness, and derivative
  sign are `0.5` (apart from small-size parity effects where applicable).

Those nonzero limits are not contradictions. They reveal different
denominators. For example, half of parity's nonzero one-step derivatives point
`0 -> 1` and half point `1 -> 0`, so derivative sign is exactly `0.5`; edge
preservation conditions only on true sources, all of which make the bad
`1 -> 0` transition, so it is exactly `0`.

**Bottom line:** if the desired intuition is "alternation should become
maximally nonmonotone," edge preservation, closure-based scores, switch
simplicity, and threshold repair express it. A generic pair average need not.
"""
    ),
    code(
        r"""
complement_pairs = [
    ("all A are B", "not all A are B"),
    ("some A are B", "no A are B"),
    ("at least two overlap", "at most one overlaps"),
    ("A equals B", "A differs from B"),
    ("A and B are comparable", "A and B are incomparable"),
    ("tautology", "contradiction"),
]
opposite = {"RU": "RD", "RD": "RU", "LU": "LD", "LD": "LU"}

scorecard_rows = []
for metric, label in METRICS.items():
    exact_values = df.loc[df.exact, metric]
    nonexact_values = df.loc[~df.exact, metric]
    mirror_errors = []
    for first, second in complement_pairs:
        for direction, mirror_direction in opposite.items():
            first_score = df.loc[
                (df.meaning == first) & (df.direction == direction), metric
            ].iloc[0]
            second_score = df.loc[
                (df.meaning == second)
                & (df.direction == mirror_direction),
                metric,
            ].iloc[0]
            mirror_errors.append(abs(first_score - second_score))
    ladder_values = ladder.loc[label].to_numpy()
    scorecard_rows.append(
        {
            "metric": label,
            "all exact directions = 1": np.allclose(exact_values, 1),
            "non-exact directions < 1": np.mean(nonexact_values < 1 - 1e-12),
            "ladder comparisons passed": sum(
                left > right
                for left, right in zip(ladder_values[:-1], ladder_values[1:])
            ),
            "max complement-mirror error": max(mirror_errors),
        }
    )

scorecard = pd.DataFrame(scorecard_rows).set_index("metric")
scorecard.round(3)
"""
    ),
    md(
        r"""
## 5. What the scorecard says

### Strong general-purpose candidates

**Nearest-monotone edit score** is the cleanest global repair measure in this
test: it gives `1` to all independently exact directions, recovers the full
intuition ladder, and has complement symmetry.
Its interpretation is direct. Its costs are computational optimization and
dependence on the situation weights and chosen normalization.

**Edge preservation** is the cleanest local behavioral measure. It gets exact
endpoints and the ladder, and its events correspond to one-element
counterfactuals. It is not complement-invariant because it conditions on a
true source; that is a property of the estimand, not an implementation defect.

**Two-sided entropy minimum** passes the exact-endpoint and complement tests,
but only two of the three ladder comparisons because both the interval and
parity examples receive zero. Its values remain normalized information, not
percentages of preserved entailments or edits.

### Useful but narrower

**Closure inflation** is transparent and preserves the manuscript's one-sided
repair idea. Like the majorant entropy and closure precision, it is not
complement-invariant because adding false points and deleting true points are
different normalizations.

**Chain inversion** and **unconditional pair score** satisfy complement
symmetry. They differ mainly in weighting: the chain measure counts a pair once
for every maximal chain containing it, whereas the pair score counts each pair
once.

### Measures that fail as general directional scores

**Switch simplicity** assigns `1` to some non-exact directions. A clean
`1 -> 0` boundary has one switch and looks maximally simple even when testing
upward monotonicity. This is useful cognitive-organization information, but not
a directional monotonicity score.

**Best simple-threshold fit** answers whether a meaning resembles one
restricted cardinality-feature family. Exact monotone meanings outside that
family can still be penalized.
"""
    ),
    md(
        r"""
## 6. Local versus global behavior

The next table compares cases that expose denominator and locality effects.
"""
    ),
    code(
        r"""
representatives = [
    "all A are B",
    "most A are B",
    "exactly two overlap",
    "between one and three overlap",
    "even overlap",
    "A equals B",
    "A differs from B",
    "A and B are incomparable",
    "both empty or both nonempty",
]
columns = [
    "meaning",
    "exact",
    "majorant_entropy",
    "pairwise_preservation",
    "edge_preservation",
    "nearest_monotone_edit",
    "closure_inflation",
    "chain_switch",
    "chain_inversion",
    "derivative_sign",
]
df[(df.direction == "RU") & df.meaning.isin(representatives)][columns].round(3)
"""
    ),
    md(
        r"""
Three warnings emerge:

1. `between one and three overlap` is non-exact but receives very high local,
   pairwise, closure, edit, and threshold scores. On M4, its only upper boundary
   failure occurs at maximal overlap. The high values are honest finite-domain
   geometry, not evidence of exact monotonicity.
2. `A differs from B` receives high pair/edge preservation but very low edit
   and closure-inflation scores. Violations are rare **conditional events**, yet
   repairing all of them globally forces a large structural change.
3. `A equals B` gets high closure precision and switch simplicity despite zero
   upward preservation. Those measures reward a sparse organized boundary, not
   robust upward entailment.
"""
    ),
    md(
        r"""
## 7. Robustness curves: proposal 7 becomes distinct only after weighting

If `(x,y)` is sampled uniformly from all comparable pairs with `Q(x)=1`, then
counterfactual robustness is exactly all-pair preservation. A new metric appears
only after choosing a transformation distribution.

Here every expansion distance `k` receives equal weight. Undefined distances,
where no true source has such an expansion, are shown as missing rather than
vacuously scored `1`.
"""
    ),
    code(
        r"""
curve_names = [
    "B threshold with one exception",
    "B has one to three elements",
    "B has even cardinality",
]
curves = (
    df[(df.direction == "RU") & df.meaning.isin(curve_names)]
    .set_index("meaning")[
        ["robustness_k1", "robustness_k2", "robustness_k3", "robustness_k4"]
    ]
)
display(curves.round(3))

ax = curves.T.plot(marker="o", figsize=(8, 4))
ax.set(
    xlabel="number of elements added (k)",
    ylabel="truth survival",
    title="Right-upward monotonicity survival curves",
    xticks=range(4),
    xticklabels=[1, 2, 3, 4],
    ylim=(-0.05, 1.05),
)
plt.show()
"""
    ),
    md(
        r"""
The parity predicate alternates: survival is `0` at odd distances and `1` at
even distances. A single scalar conceals exactly the pattern that makes parity
intuitively irregular. For a learnability study, the curve or its first few
values may be more informative than an average.
"""
    ),
    md(
        r"""
## 8. Context-sensitive profiles

Global pairwise preservation weights contexts according to how many eligible
pairs they contribute. The context mean instead gives every fixed `A` equal
weight, but excludes contexts with no true-source prediction; `context_coverage`
reports how much of the context space was defined.
"""
    ),
    code(
        r"""
context_names = [
    "most A are B",
    "exactly two overlap",
    "A differs from B",
    "A and B are incomparable",
    "both empty or both nonempty",
]
df[(df.direction == "RU") & df.meaning.isin(context_names)][
    [
        "meaning",
        "pairwise_preservation",
        "context_mean",
        "context_sd",
        "context_coverage",
    ]
].round(3)
"""
    ),
    md(
        r"""
The standard deviation is not "more monotonicity"; it is a second axis:
**stability across restrictor contexts**. Reporting `(mean, SD, coverage)` is
more honest than folding all three into an undocumented scalar.
"""
    ),
    md(
        r"""
## 9. Directional profiles instead of a maximum

The maximum asks whether *some* direction is strong. It cannot distinguish one
clean direction from diffuse partial scores. The following uses edge
preservation because its units are easy to interpret.
"""
    ),
    code(
        r"""
profile_names = [
    "all A are B",
    "most A are B",
    "exactly two overlap",
    "A differs from B",
    "A and B are incomparable",
]
profiles = (
    df[df.meaning.isin(profile_names)]
    .pivot(index="meaning", columns="direction", values="edge_preservation")
    [["RU", "LU", "RD", "LD"]]
)
profiles["max"] = profiles.max(axis=1)
profiles["mean"] = profiles[["RU", "LU", "RD", "LD"]].mean(axis=1)
profiles["purity = max - mean"] = profiles["max"] - profiles["mean"]
profiles.round(3)
"""
    ),
    md(
        r"""
`purity` is not automatically desirable. A quantifier can be exactly monotone
in multiple directions. The scientifically useful object is the four-vector;
max, mean, and purity are optional hypotheses about what learners exploit.
"""
    ),
    md(
        r"""
## 10. Threshold and LoT proposals

### Threshold fit

The implemented candidate family is intentionally narrow. It searches
thresholds over simple features that are monotone in the tested direction:
overlap size, the varied-set size, directional set difference, and the
`overlap - nonoverlap` contrast used by `most`. Accuracy is maximized over the
feature and threshold.

This family can represent `all`, `some`, `most`, and simple cardinality
thresholds, but many monotone functions lie outside it. Its failures are useful
evidence against treating threshold fit as a general monotonicity score.

### Exception-code proxy

For minimum edit count `e` among `N` situations, the proxy code length is:

$$
\log_2\sum_{i=0}^{e}\binom{N}{i}.
$$

The reported score is one minus this code length divided by `N`. It rewards a
monotone base plus a short list of exceptions. It is **not** a full LoT result:
the monotone base itself is treated as free, all exception locations have the
same cost, and grammar/inference costs are omitted.
"""
    ),
    code(
        r"""
df[(df.direction == "RU") & df.meaning.isin(representatives)][
    [
        "meaning",
        "best_simple_threshold",
        "best_threshold_description",
        "edit_count",
        "exception_code_proxy",
    ]
].round(3)
"""
    ),
    md(
        r"""
## 11. Recommendations for a learnability follow-up

### Primary candidates

1. **Edge preservation** for the hypothesis that learners exploit local
   one-element regularities.
2. **Nearest-monotone edit score** for global truth-table repair cost.
3. **Majorant entropy** as the manuscript baseline, preserving comparability
   with the published analysis.

Keep all four directions for each metric. Do not replace the profile with a max
until the aggregation hypothesis is tested.

### Secondary diagnostics

- robustness curves by `k`;
- context mean, SD, and coverage;
- chain-switch simplicity as a separate boundary-complexity predictor;
- threshold/LoT fit as representational-simplicity predictors.

### What this notebook establishes

It establishes the measures' finite-universe behavior, endpoint properties,
complement symmetry, and response to hand-chosen semantic contrasts.

### What it does not establish

It does **not** show which metric predicts neural learnability best. That
requires computing these features for the 2,000 grammar-generated quantifiers
and comparing held-out model fit or predictive performance against AUC. The
best next experiment is a preregistered model comparison with manuscript
entropy, edge preservation, nearest-monotone edit score, switch simplicity, and context
stability entered as separate predictors.
"""
    ),
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
