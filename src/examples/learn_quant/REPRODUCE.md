# Reproducing the Monotonicity / Learnability Figures & Analyses

This document explains how to regenerate every figure and statistical result
used in the SALT35 monotonicity manuscript and the follow-up analyses, from the
data in this repository.

All commands are run from the package root:

```
src/examples/learn_quant/
```

Use any Python ≥3.10 environment with `pandas`, `numpy`, `matplotlib`,
`plotnine`, `statsmodels`, `scikit-learn`, and `scipy` (`psycopg` is needed
only for the optional database-verification step):

```bash
pip install pandas numpy matplotlib plotnine statsmodels scikit-learn scipy
```

---

## 1. Data source and provenance

There are now two deliberately separate run tables:

```
outputs/combined_runs_AOC_monotonicity_updated.csv    # manuscript-era metric
outputs/combined_runs_AOC_monotonicity_corrected.csv  # complement-dual alternative
```

The first CSV was originally exported from the project's **MLflow PostgreSQL backend**
(experiments `40 expressions_shuffled_2k` + `42 repeated_runs` = LSTM, and
`46 transformers_improved_1` + `47 transformers_improved_2` = Transformer),
using the queries in `notebooks/get_AOC.ipynb` and
`notebooks/get_experiment_data.ipynb`.

**It has been verified byte-for-byte against the live database** (see step 2).
So none of the figures require a running database — the CSV is sufficient and
authoritative for reproducing the *published* numbers.

The second CSV has the identical 8,005 learning-run rows and AUC outcomes, but
replaces the directional monotonicity values with a complement-dual alternative.
It retains every manuscript-era score in `*_original` columns. Generate it with:

```bash
/path/to/altk/bin/python scripts/recalculate_corrected_monotonicity.py
```

The alternative computes downward monotonicity by identity:
`down(Q) = up(not Q)`. The original code instead used a true-successor feature
(`flip=True`) for downward while using a true-predecessor feature for upward.
The original is a uniform **majorant** construction and preserves order duality,
but it violates complement mirror symmetry. The alternative pairs an upward
majorant with a downward **minorant**. It preserves complement symmetry but is
not a neutral correction of the published definition.

For the operator derivations, a property matrix, Table 4 examples, and a
two-sided entropy alternative that preserves both symmetries, open
`notebooks/monotonicity_measure_variants_walkthrough.ipynb`.

Key columns:

| column | meaning |
|---|---|
| `expression` | the quantifier expression (grammar string) |
| `model` | `LSTM` or `Transformer` |
| `run` | repeated-training index 1–4 |
| `training` | `True` = converged/trained, `False` = untrained baseline |
| `monotonicity_entropic`, `degree` | overall monotonicity (0–1); `degree` uses the complement-dual alternative in the historically named `corrected` CSV |
| `right_upward`, `left_upward`, `right_downward`, `left_downward` | directional monotonicity components (0–1) |
| `*_original` | manuscript-era directional/degree values (alternative CSV only) |
| `first_step` | training step at which `val_loss_running_avg50 < 0.05` (learning speed; converged runs only) |
| `val_loss_step_AOC` | **Validation Loss AUC** = `SUM(val_loss_step)` = area under the validation-loss curve (learning difficulty). Higher = harder. |
| `expression_depth` | max parenthesis nesting depth (1–4, mostly 4) |

> **Naming note:** the column is called `val_loss_step_AOC` for historical
> reasons, but it is the **area *under* the loss curve (AUC)** — the integral of
> validation loss over training. All figure labels say "Validation Loss AUC".
> The positive correlation with length (and negative with monotonicity) simply
> reflects that more accumulated loss = harder to learn; it is not about
> "over vs under" the curve.

Two size measures are computed on the fly from `expression` (see
`scripts/reproduce_figures.py`):

- `leaf_count` — number of leaf operands / atoms (`A`, `B`, indices); range 2–13
- `func_count` — number of function/operator applications; range 1–15

`func_count` and `leaf_count` correlate at r≈0.95 (interchangeable "length"
measures); both correlate only r≈0.58 with `expression_depth`, so **length is not
the same as depth**. Their correlations with learning are also nearly identical
(leaf/func vs AUC: r≈0.23/0.22; vs first_step: r≈0.21/0.24), so it makes no
practical difference which "length" measure you report.

---

## 2. (Optional) Verify the CSV against the live Postgres DB

Only needed if you want to re-confirm provenance. Requires the cluster Postgres
running and an SSH tunnel forwarding local port 5432 (see
`tracking/` and your `~/.ssh/klone-postgres` host):

```bash
ssh -N klone-postgres          # separate terminal; forwards localhost:5432
export MLFLOW_PG_DSN="postgresql://USER:PASSWORD@localhost:5432/mlflow_db"
python scripts/verify_postgres_matches_csv.py --n 150
```

The scripts read the connection string from the `MLFLOW_PG_DSN` environment
variable (or pass `--dsn`); no credentials are stored in the repository.

Expected result: monotonicity matches for 100% of sampled runs (diff < 1e-6) and
`val_loss_step_AOC` matches to ~1e-13. (Confirmed: 300/300 monotonicity, all
sampled AOC exact.)

To re-export the whole table from the DB instead of using the CSV:

```bash
python scripts/reproduce_from_postgres.py --dump-csv outputs/combined_from_postgres.csv
```

> Note: the `metrics` table is ~675M rows / 195 GB, so a full re-export over the
> tunnel is slow (~10 min). The committed CSV is identical, so this is optional.

---

## 3. Figures

```bash
python scripts/reproduce_figures.py

# Complement-dual counterparts, preserving published reproductions:
python scripts/reproduce_figures.py \
  --csv outputs/combined_runs_AOC_monotonicity_corrected.csv \
  --outdir figures/corrected_metric
```

Writes to `figures/`:

| file | description |
|---|---|
| `paper_figure1.png` | **Exact manuscript Figure 1**: Monotonicity vs Validation-Loss-AUC, coloured by model, red dashed linear fit |
| `length_vs_auc.png` | Twin of Fig 1 with **leaf count** (length) on the y-axis |
| `functions_vs_auc.png` | Twin of Fig 1 with **function count** on the y-axis |
| `monotonicity_vs_training_step.png` | Monotonicity vs step-at-convergence |
| `first_step_vs_auc.png` | **Appendix figure**: learning speed (`first_step`) vs difficulty (AUC), per-model linear fits; prints & saves the caption's partial Spearman r (controlling for model) to `analysis/tables/10_first_step_auc_partial_spearman.txt`. Reproduced r = 0.893 vs published 0.892 |
| `depth_vs_learning.png` | 2-panel: expression depth vs `first_step` and AUC |
| `length_vs_learning.png` | 2-panel: leaf count vs `first_step` and AUC |
| `functions_vs_learning.png` | 2-panel: function count vs `first_step` and AUC |

Flags:
- `--from-mlruns` — build the training-step figure by walking the local
  `mlruns/` folder instead of the CSV (slow on iCloud drives).
- `--csv PATH`, `--outdir PATH` — override input/output locations.

---

## 4. Deeper statistical analyses

```bash
python scripts/deeper_analysis.py
```

Prints statistical tables to stdout, writes each section's tables verbatim to
`analysis/tables/<section>.txt` (so the full OLS/ridge/nested-model results are
committed artifacts, not just console output), and writes figures to `figures/`:

| table file | contents |
|---|---|
| `analysis/tables/1_joint_partial_correlations.txt` | raw + partial correlations, joint OLS coefficient table |
| `analysis/tables/4_directional_monotonicity.txt` | directional correlations, joint OLS (n, R², full coefficient table) |
| `analysis/tables/6_operator_effects.txt` | operator prevalence, Ridge coefficients + bootstrap CIs, R² comparison |
| `analysis/tables/7_nested_model_comparison.txt` | nested model R²/AIC/BIC, F-tests, commonality analysis |
| `analysis/tables/8_predict_monotonicity.txt` | flipped-outcome nested models, F-tests, commonality analysis |

Figures:

| file | analysis |
|---|---|
| `partial_corr_heatmap.png` | **#1** Joint model + partial correlations: are monotonicity and length independent predictors of learning difficulty? |
| `directional_monotonicity.png` | **#4** Directional monotonicity: upward vs downward, left vs right |
| `operator_effects.png` | **#6** Per-operator difficulty (Ridge regression + bootstrap CIs) |
| `nested_model_comparison.png` | **#7** Hierarchical regression predicting **AUC**: which predictor is stronger, and does each add unique variance? |
| `predict_monotonicity.png` | **#8** Flip the outcome — predict **monotonicity** from complexity (length) vs learnability (AUC); which better explains it? |

### Headline results

These numbers are from the **manuscript-era majorant metric** (`updated.csv`).
For complement-dual sensitivity results see `FIGURES_TECHNICAL.ipynb`.

- **#1** Monotonicity and length remain *independent* signals.
  Controlling for each other, monotonicity partial r = −0.264 and
  func_count partial r = +0.155 with AUC. More monotone **and** shorter →
  independently easier to learn.
- **#4** The **downward** direction dominates. In a joint pooled model,
  `downward` β = −1196 and `upward` β = −593; in the random-intercept model
  they are −1193 (p = 1.3e−15) and −597 (p = 5.1e−6). Both directions predict
  easier learning; the T2 truth-set analysis (§4d) shows this is substantially
  mediated by corpus statistics.
- **#6** Operator *identity* more than doubles explained variance over raw
  length (R² 0.10 → 0.25). `union` is a large difficulty outlier (+782 per SD),
  far worse than `intersection`/`difference`; `not` and `greater_than` are
  associated with *easier* learning.
- **#7** Length and monotonicity each add unique variance.
  Standardized |β| ratio mono/length = 1.74; unique R² is 0.066 for
  monotonicity and 0.022 for length. Full model R² is 0.167 (manuscript ~0.19).
- **#8** Flipping the outcome to predict **monotonicity**
  (per-expression, n=1795), AUC remains stronger than length:
  AUC-alone R²=0.106 vs length-alone R²=0.062; unique R² is 0.076 vs 0.032.
  Caveat: monotonicity is intrinsic; AUC is a training outcome. This reports
  covariance strength, not causal direction.

> Statistical notes: these are correlational and pooled across the 4 repeated
> runs. For inference-grade p-values, add a random intercept per expression
> (mixed model), as in the mixed-model cells of `FIGURES_TECHNICAL.ipynb` (Models A–C). The operator
> regression is intentionally Ridge-regularized because the 11 operator counts
> are exactly collinear (rank 10) due to grammar constraints — plain OLS is
> rank-deficient and returns unstable coefficients.

---

## 4b. Review extensions (2026-07)

```bash
python scripts/review_extensions.py
```

Additional figures & tables from the reproduction review, all from the same CSV.
Figures to `figures/`, printed tables teed to `analysis/tables/9*.txt`:

| output | contents |
|---|---|
| `nested_model_redraw.png` (+ `9a`) | clarified nested-model figure: one bar per **distinct** model (the original showed the full model twice and omitted the mono-only model) |
| `directional_panelB_redraw.png` (+ `9b`) | readable 2×2 upward/downward split (downward on x, cell means and n on bars) |
| `9c_directional_diagnostics.txt` | directional-degree distributions (57% of expressions have zero upward degree; `degree` attained by a downward sense in 83%), per-architecture asymmetry + interaction OLS |
| `9d_operator_prevalence.txt` | % of expressions attested + cumulative counts per operator |
| `9e_polarity_counterbalance.txt` | the downward surplus is **not** driven by `difference` (r ≈ 0.00 with the gap); `not` is the operative polarity flipper (mean upward degree 0.33 → 0.47 → 0.65 with 0/1/2 negations) |
| `operator_by_architecture.png` (+ `9f`) | per-operator Ridge fit **separately per architecture**, 3 panels: absolute units, Transformer−LSTM difference, and **rescaled by each architecture's own AUC SD** — the rescaling shows the difficulty profiles nearly coincide (r = 0.990); the Transformer's larger absolute coefficients mostly reflect its larger overall loss scale (AUC SD 2368 vs 1680). Significant absolute differences: union +215, equals +194, greater_than −108; only `equals` stays relatively harder after rescaling |
| `paper_figure1_redl.png` (+ `9g`) | scripted version of the full-range-fit-line variant (was ad hoc) |
| `figure_vs_mixedmodel.png` (+ `9h`) | scripted version of the scatter + mixed-model-fits overlay (was ad hoc). Bonus: its model `AUC ~ degree * model + (1|expression)` reproduces the **published Table 6** to ~0.2% (monotonicity −1792.6 vs −1789.7; Transformer +1583.2 vs +1583.4; interaction −1058.6 vs −1058.1) |
| `monotonicity_gap_analysis.png` (+ `9i`) | why low-but-nonzero degrees are rare (the sparse zone in (0, 0.15] of Figure 2's y-axis): (1) real-data decomposition — per-direction ~6.6% of expressions fall there but the max-of-four `degree` only 1.7%; (2) lattice simulation — noise-corrupted quantifiers fill the low range but crisp formula-like families don't, except **equality-type** predicates; (3) the 34 real gap expressions are enriched 3.0× in `not` and 2.2× in `equals` |

---

## 4c. Question-2 follow-ups (2026-07-26): negation pairs & truth-set statistics

These two scripts address HANDOFF.md §4 (what drives the upward/downward
learning asymmetry). Unlike everything above, they need the **original run
archive** (the old `altk` repo checkout) for the expression pool and universe
pickles, plus the `altk` conda env:

```bash
python scripts/negation_pair_test.py    # Q2-T1; --altk-archive PATH to override
python scripts/negation_pair_test.py \
  --csv outputs/combined_runs_AOC_monotonicity_corrected.csv --tag corrected
python scripts/truth_set_stats.py       # Q2-T2
python scripts/truth_set_stats.py \
  --csv outputs/combined_runs_AOC_monotonicity_corrected.csv --tag corrected
```

For a narrated, cell-by-cell version of T1, open
`notebooks/negation_pair_test_walkthrough_expanded.ipynb` with the `altk` kernel.
The notebook makes the provenance explicit: **T1 trains no new neural
network**. It evaluates symbolic grammar expressions on sampled set-theoretic
scenes, then joins validation-loss AUCs from the already completed LSTM and
Transformer runs in `outputs/combined_runs_AOC_monotonicity_updated.csv`.

| output | contents |
|---|---|
| `analysis/tables/11_negation_pair_test.txt` (+ `figures/negation_pair_test.png`) | **T1 negation-pair test.** 44 complement pairs found among the trained 2,000 (884 complement-closed meanings in the 9,550-meaning pool); 41 verified as *functional* complements on 20k training-style scenes (3 pairs are complements only on the 256-scene universe — a caveat for any universe-level analysis). Within pairs the decision boundary is identical and direction flips, yet AUC is statistically indistinguishable (mean-arch Δ = +16, Wilcoxon p = 0.59; polarity-contrast subset Δ = +66, p = 0.21). The population directional model predicts Δ = +182 (all pairs) / +558 (contrast pairs); observed/predicted ≈ **0.12**. So ~90% of the directional asymmetry is *not* boundary-level — it lives in sample composition / measure, not in the learner's treatment of a given boundary. Bonus (H4): the measure-mirror identity up(e) = down(¬e) fails badly for half the pairs (median |dev| 0.05, max 0.79, r = 0.71) — direct evidence of measure-side noise in the directional degrees. |
| `analysis/tables/11_negation_pair_test_corrected.txt` (+ historically named corrected figure) | **Complement-dual T1.** Mirror error is exactly zero. Mean-architecture Δ = +22 (p=.29); among 21 strong-contrast pairs Δ = +32 (p=.23). Population prediction is +192 / +327, giving observed/predicted ≈ **0.10**. The substantive paired conclusion is unchanged. |
| `analysis/tables/12_truth_set_stats.txt` (+ `figures/truth_set_stats.png`, `analysis/truth_set_stats_features.csv`) | **T2 truth-set-statistics mediation.** Per-expression class-conditional input statistics computed on 4,000 training-style models (M=12, X=16). Truth-set stats alone explain **R² = 0.54** of per-expression mean AUC (directions alone: 0.13); adding them shrinks the downward β by **80%** (−708 → −138/SD, still p = 0.002). Dominant mediator: **class separation** (L2 distance between mean zone-count vectors of positive vs negative examples), r = −0.68 with AUC, β = −1500/SD, p ≈ 1e-138 — and it correlates +0.48 with downward degree. Verdict: the downward advantage is mostly carried by input-statistic separability of the classes (H2 in generalized form), consistent with T1's small boundary-level residual. |
| `analysis/tables/12_truth_set_stats_corrected.txt` (+ historically named corrected figure/features) | **Complement-dual T2.** Directions-only R² falls from 0.126 to **0.063**. Downward/upward β become −425/−256. After truth-set controls they are +61 (p=.13) and −32 (p=.38): neither supports a remaining negative directional advantage. Class separation remains dominant; the earlier claim of a specifically downward residual does not survive. |

## 4d. Manuscript metric audit

```bash
/path/to/altk/bin/python scripts/verify_manuscript_metric.py
```

Writes `analysis/tables/14_manuscript_metric_verification.txt` and two CSVs.
The audit shows that the manuscript used the intended order-dual majorant
calculation, which is not complement invariant:

- Three identifiable Table 4 expressions reproduce all 12 published directional
  cells under the old code (to displayed precision), not the complement-dual code.
- The committed old CSV reproduces Table 6 to within 0.2% (e.g. published
  monotonicity β = −1789.666; reproduction = −1792.639).
- The complement-dual Table 6 coefficients are not directly comparable in raw units
  because the degree distribution changes; the standardized mixed-model effect
  weakens from −640 to −561.
- Published Figure 2 reports r = −0.3453; the committed manuscript-era export
  gives −0.3326 (an earlier-data-snapshot discrepancy), while complement-dual r = −0.3045.

## 4e. Monotonicity-measure variants

```bash
/path/to/altk/bin/python scripts/calculate_monotonicity_variants.py
```

This writes `analysis/monotonicity_measure_variants_2k.csv`, containing the
four closure/interior primitives, manuscript majorant, complement-dual,
two-sided mean/min/max entropy, and direct violation-rate scores for all 2,000
expressions. Open
`notebooks/monotonicity_measure_variants_walkthrough.ipynb` for the derivation,
Table 4 comparison, 44-pair mirror audit, distribution comparison, and measure
recommendation.

## 4f. Semantic benchmark for two-sided minimum

```bash
/path/to/altk/bin/python scripts/semantic_monotonicity_benchmark.py
/path/to/altk/bin/python scripts/build_semantic_benchmark_notebook.py
```

The benchmark defines 34 familiar set-theoretic meanings independently of the
grammar sample, verifies their expected exact directions by exhaustive
comparable-pair checks on M6/X6, and writes
`analysis/semantic_monotonicity_benchmark.csv`. Open and run
`notebooks/two_sided_min_semantic_benchmark.ipynb` for the interpretation.

Two-sided minimum gets all categorical endpoints right in this suite: all 45
exact directions score 1 and none of 91 non-exact directions scores 1. Its
graded values are not uniformly intuitive, however. Non-exact scores reach
0.610 for `exactly five overlap` on the six-element universe and 0.423 for
emptiness biconditional/XOR meanings. Treat it as the strongest symmetric
candidate tested here, not as a validated replacement for the manuscript
majorant.

## 4g. Alternative monotonicity metrics

```bash
python scripts/alternative_monotonicity_metric_benchmark.py
python scripts/build_alternative_monotonicity_metrics_notebook.py
```

This exhaustive M4/X4 benchmark compares the manuscript entropy score with
pairwise and Hasse-edge preservation, nearest-monotone edit distance, closure
costs, chain switches and inversions, distance-sensitive robustness,
coordinatewise derivative signs, context and direction profiles, simple
threshold fit, and an exception-code proxy. It writes
`analysis/alternative_monotonicity_metric_benchmark.csv`.

Open and run
`notebooks/alternative_monotonicity_metrics_benchmark.ipynb` for the formulas,
preregistered intuition tests, complement-symmetry audit, survival curves, and
measure-by-measure limitations. The threshold and exception-code columns are
restricted representation diagnostics, not general monotonicity measures.

---

## 5. One-shot reproduction

`FIGURES_TECHNICAL.ipynb` is the primary technical companion notebook. It
loads both run tables, replicates every figure and statistic from the manuscript
using the majorant metric (`updated.csv`), and runs all T1/T2 analyses with
complement-dual as a sensitivity comparison. Open it with the `altk` kernel and
Run All. The historical one-shot notebook remains at
`notebooks/reproduce_figures_and_analysis.ipynb`.

---

## Script reference

| script | purpose |
|---|---|
| `scripts/reproduce_figures.py` | all figures (paper Fig 1, length/function twins, complexity-vs-learning, appendix first-step-vs-AUC) from the CSV |
| `scripts/deeper_analysis.py` | #1 partial correlations, #4 directional monotonicity, #6 per-operator difficulty (tables to `analysis/tables/`) |
| `scripts/review_extensions.py` | 2026-07 review figures/diagnostics: clarified redraws, per-architecture operator analysis (with within-architecture rescaling), directional diagnostics, operator prevalence, polarity-counterbalance test |
| `scripts/reproduce_from_postgres.py` | rebuild the run table directly from the live MLflow Postgres DB (needs tunnel) |
| `scripts/verify_postgres_matches_csv.py` | confirm the CSV equals the live DB (sampled) |
| `scripts/negation_pair_test.py` | Q2-T1: complement-pair test of the up/down asymmetry; no new neural training (needs the altk run archive) |
| `scripts/truth_set_stats.py` | Q2-T2: truth-set-statistics mediation of the downward advantage (needs the altk run archive) |
| `scripts/calculate_monotonicity_variants.py` | calculate six documented metric variants for all 2,000 expressions |
| `scripts/build_monotonicity_variants_notebook.py` | regenerate the pedagogical variants notebook source |
