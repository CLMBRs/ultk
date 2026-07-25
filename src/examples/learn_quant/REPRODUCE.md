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

The single source of truth for the figures is:

```
outputs/combined_runs_AOC_monotonicity_updated.csv   (8005 rows, one per run)
```

This CSV was originally exported from the project's **MLflow PostgreSQL backend**
(experiments `40 expressions_shuffled_2k` + `42 repeated_runs` = LSTM, and
`46 transformers_improved_1` + `47 transformers_improved_2` = Transformer),
using the queries in `notebooks/get_AOC.ipynb` and
`notebooks/get_experiment_data.ipynb`.

**It has been verified byte-for-byte against the live database** (see step 2).
So none of the figures require a running database — the CSV is sufficient and
authoritative.

Key columns:

| column | meaning |
|---|---|
| `expression` | the quantifier expression (grammar string) |
| `model` | `LSTM` or `Transformer` |
| `run` | repeated-training index 1–4 |
| `training` | `True` = converged/trained, `False` = untrained baseline |
| `monotonicity_entropic`, `degree` | overall monotonicity (0–1) |
| `right_upward`, `left_upward`, `right_downward`, `left_downward` | directional monotonicity components (0–1) |
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
```

Writes to `figures/`:

| file | description |
|---|---|
| `paper_figure1.png` | **Exact manuscript Figure 1**: Monotonicity vs Validation-Loss-AUC, coloured by model, red dashed linear fit |
| `length_vs_auc.png` | Twin of Fig 1 with **leaf count** (length) on the y-axis |
| `functions_vs_auc.png` | Twin of Fig 1 with **function count** on the y-axis |
| `monotonicity_vs_training_step.png` | Monotonicity vs step-at-convergence |
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

- **#1** Monotonicity and length are *independent* signals. Controlling for each
  other, both remain strong: monotonicity partial r ≈ −0.24, func_count partial
  r ≈ +0.19 with AUC (both p < 1e-50). More monotone **and** shorter →
  independently easier to learn.
- **#4** The monotonicity effect is **downward-driven**: in a joint model,
  `downward` β ≈ −2184 (p ≈ 1e-160) while `upward` β ≈ +163. Downward-entailing
  quantifiers are markedly easier; upward monotonicity barely matters.
- **#6** Operator *identity* more than doubles explained variance over raw
  length (R² 0.10 → 0.25). `union` is a large difficulty outlier (+782 per SD),
  far worse than `intersection`/`difference`; `not` and `greater_than` are
  associated with *easier* learning.
- **#7** Length and monotonicity each add unique, significant variance on top of
  the other (nested F-tests p < 1e-50 both ways). **Monotonicity is the stronger
  predictor**: standardized |β| ratio mono/length ≈ 1.8, and its unique R²
  contribution (0.089) is ~3.4× length's unique R² (0.026). Adding monotonicity
  on top of length raises R² from 0.10 to 0.19; adding length on top of
  monotonicity raises it from 0.16 to 0.19. So both matter, but monotonicity
  carries most of the explanatory weight.
- **#8** Flipping the outcome to predict **monotonicity** (per-expression,
  n≈1795): **learnability (AUC) explains monotonicity far better than length**.
  AUC alone R²=0.127 vs length alone R²=0.032; in the joint model AUC's unique R²
  (0.105) is ~10× length's unique R² (0.010), and its standardized |β| is ~3.3×
  larger. Adding AUC on top of length lifts R² from 0.03 to 0.14; adding length
  on top of AUC barely helps (+0.010). Caveat: this is descriptive —
  monotonicity is intrinsic to the expression and AUC is a training *outcome*,
  so the causal arrow is monotonicity → learnability; the analysis only reports
  which covaries more strongly.

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

## 5. One-shot reproduction

The notebook `notebooks/reproduce_figures_and_analysis.ipynb` runs steps 3, 4
and 4b end-to-end and displays every figure inline — including all review
figures. Open it with the `altk` kernel and Run All. (Verified clean on
2026-07-23: 0 errors, every figure and `analysis/tables/*.txt` regenerated.)

---

## Script reference

| script | purpose |
|---|---|
| `scripts/reproduce_figures.py` | all figures (paper Fig 1, length/function twins, complexity-vs-learning) from the CSV |
| `scripts/deeper_analysis.py` | #1 partial correlations, #4 directional monotonicity, #6 per-operator difficulty (tables to `analysis/tables/`) |
| `scripts/review_extensions.py` | 2026-07 review figures/diagnostics: clarified redraws, per-architecture operator analysis (with within-architecture rescaling), directional diagnostics, operator prevalence, polarity-counterbalance test |
| `scripts/reproduce_from_postgres.py` | rebuild the run table directly from the live MLflow Postgres DB (needs tunnel) |
| `scripts/verify_postgres_matches_csv.py` | confirm the CSV equals the live DB (sampled) |
