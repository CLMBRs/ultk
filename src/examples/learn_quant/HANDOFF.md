# HANDOFF — monotonicity & learnability follow-up work (start here)

*Written 2026-07-24 at the end of a reproduction-review session conducted in the
older `altk` repo. This document is self-contained: a fresh session in THIS
repo (`ultk`) should be able to proceed from it without the prior conversation.*

## 0. Context in three sentences

The SALT 35 paper (Haberland & Steinert-Threlkeld, "Quantifiers that are more
monotone are easier to learn"; PDF in `manuscript/`) trained LSTM/Transformer
learners on 2,000 LoT-generated quantifiers and found monotonicity predicts
ease of learning. A 2026-07 review verified the results end-to-end from the
committed CSV, added deeper analyses, and surfaced three open questions. This
handoff ports the verified analysis stack into this repo and specifies those
three research programs.

## 1. What was ported here (all verified working from this directory)

| path | what it is |
|---|---|
| `analysis/combined_runs_AOC_monotonicity_updated.csv` | **source of truth**: 8,005 rows (one per training run; 7,172 trained; 2,000 unique expressions), verified byte-for-byte against the MLflow Postgres backend (`scripts/verify_postgres_matches_csv.py`; needs the klone SSH tunnel, see `tracking/`) |
| `analysis/tables/*.txt` | every statistical table from the review, regenerable |
| `scripts/reproduce_figures.py` | paper figure + length/depth figures; also home of the leaf/function-count parsers |
| `scripts/deeper_analysis.py` | partial correlations, directional monotonicity, per-operator Ridge, nested models, predict-monotonicity (sections 1/4/6/7/8) |
| `scripts/review_extensions.py` | review additions (sections 9a–9i): clarified redraws, per-architecture operator analysis, operator prevalence, polarity-counterbalance test, **monotonicity-gap analysis** |
| `REPRODUCE.md` | how to run everything + headline results |
| `EXPERIMENT_PLAN.md` | full design for Question 1 (grammar → monotonicity) |
| `notebooks/reproduce_figures_and_analysis.ipynb` | one-shot Run-All (outputs stripped for the port; re-execute to regenerate) |
| `notebooks/FIGURES_TECHNICAL.ipynb` | number-by-number companion incl. mixed models (outputs stripped) |

Environment: conda env **`altk`** has everything (pandas, statsmodels,
scikit-learn, scipy, plotnine, jupyter): `conda activate altk`, run scripts
from this directory. Figures were NOT ported — `python scripts/reproduce_figures.py`,
`python scripts/deeper_analysis.py`, `python scripts/review_extensions.py`
regenerate all of them into `figures/`.

**Repo caveats.** (1) This repo's `.git` is corrupted (no `HEAD`/`config`) —
`git` commands fail. Re-clone from the fork or restore `.git/HEAD` before
committing anything. (2) `measures.py` here is the black-formatted equivalent
of the fixed altk version — same algorithm (diff checked: formatting/comments
only). (3) The old `altk` repo (`~/Documents/UWLing/altk/.../learn_quant`)
remains the archive of the original run artifacts (mlruns, outputs) — data
provenance lives there if ever needed again.

## 2. Verified facts every follow-up should treat as ground truth

All recomputed from the CSV during the review (details in `analysis/tables/`):

- `degree` = max of the four clipped directional senses; exact for 100% of rows.
- Monotonicity–AUC: r = −0.333 (published −0.3453 is NOT exactly reproducible
  from any committed CSV — the monotonicity computation was fixed after
  publication; same story, different third decimal). The published **Table 6
  reproduces to ≤0.2%** on corrected data (`9h`).
- Length (leaf r = +0.234 / func r = +0.219 vs AUC; interchangeable, r = 0.946).
- Nested models: monotonicity unique R² 0.089 ≈ 3.4× length's 0.026.
- **The effect survives full composition control**: degree β = −540/SD with
  length + all 11 operator counts controlled (mixed p = 2e-38). Not an
  operator artifact.
- **Directional asymmetry** (the subject of Question 2): downward β = −571/SD
  (p = 1e-39) vs upward +80/SD (n.s.) under the same full controls; downward
  raw r = −0.332 ≈ the entire degree effect; upward raw −0.120.
- **Gap** (Question 3): only 1.7% of expressions have degree in (0, 0.15]
  vs 8.8% at exactly 0; explained mechanistically in `9i` (see §5).
- Per-architecture: same operator difficulty profile (r = 0.990), amplified in
  the Transformer; downward advantage significantly larger for the Transformer
  (interaction β = −1048, p = 6e-11).

## 3. QUESTION 1 — How does the grammar shape monotonicity?

**Read `EXPERIMENT_PLAN.md` in full; it is the design.** Summary: one maximal
operator inventory (add the domain leaf `M` — enables complement via existing
`difference(M, X)` — and integer constants 0…m, which the current grammar
lacks entirely: "at least three" is currently inexpressible), then randomize
production weights per batch (Dirichlet draws over the rule simplex; the
`weight:` fields already exist in `grammar.yml` but have never been varied).
Angle 1 = expression-level ZOIB regressions with polarity-calculus features;
Angle 2 = grammar-level response surface + Sobol indices. Monotonicity is
cheap (no training), so ~60k expressions is CPU-scale.

**First concrete steps:**
1. Phase 0: add `M` leaf + integer constants to `grammar.yml`; re-derive
   anchor quantifiers (*all, some, no, most, an-even-number-of, at-least-3*)
   and confirm expected degrees.
2. Phase 1, Option B sampler (log-linear tilt of a fixed enumerated pool —
   minimal code; see plan §3.5) before writing a full PCFG sampler.
3. Pre-registered predictions are in plan §5.4 (7 of them) — check results
   against them explicitly.

## 4. QUESTION 2 — What drives the upward/downward learning asymmetry?

This needs the fullest statement, since the prior session's bullets were
compressed. The puzzle, the constraint that shapes it, five candidate drivers,
and the discriminating experiments:

### 4.1 The puzzle, precisely

Downward monotonicity predicts easier learning with a huge, robust effect;
upward monotonicity predicts almost nothing (facts in §2). Why would a neural
learner care about the *direction* of monotonicity at all?

### 4.2 The symmetry constraint (why this is genuinely puzzling)

The learner never sees the expression — only (model, label) pairs. Training
is sigmoid + binary cross-entropy, which is **symmetric under label
complement**: learning Q and learning ¬Q are the same optimization problem up
to a sign flip of the output layer. And the complement of an upward-monotone
truth set (an up-set) is a downward-monotone one (a down-set). Therefore
**pure decision-boundary geometry cannot explain a direction effect**: for
every up-set there is a down-set (its complement) with the identical boundary.
Any real asymmetry must come from something that breaks this symmetry:

- WHICH up-sets vs down-sets the grammar actually samples (they are not
  complements of each other in the sample);
- the input encoding (referent codes `[1,0,1]`-style are not symmetric under
  set complement);
- the example-sampling procedure (balanced true/false, but the *positive
  examples* of a downward quantifier concentrate on small models while an
  upward quantifier's concentrate on large ones — different input statistics);
- or the measure itself (the directional degrees may not be equally valid).

### 4.3 Candidate drivers (each with mechanism and status)

- **H1 — Sample composition / estimation artifact.** Upward variation is
  scarce (57% of expressions have zero upward degree; only 507 upward-dominant
  runs) and upward correlates +0.40 with downward, so the upward effect is
  weakly estimated and suppression flips its joint-model sign. *Mechanism:
  statistics, not learning.* Status: certainly contributes; cannot explain
  why downward's own effect is so large.
- **H2 — Positive-example geometry (input statistics).** For a downward-
  monotone quantifier, verifying models cluster at low cardinality: few
  active input bits, low input variance, possibly faster convergence of the
  relevant weights. Upward quantifiers' positives are large models. *Mechanism:
  optimization speed depends on the input distribution of the positive class.*
  Status: untested; testable in existing data (see 4.4-T2).
- **H3 — Absence-detection surfaces.** A down-set is characterized by "no
  forbidden referent-type present" — implementable by inhibitory weights on a
  threshold unit; gradient descent finds such solutions early. The Transformer
  interaction (downward advantage LARGER with attention pooling, which makes
  global absence checks cheap) is loosely consistent. *Constraint: by §4.2
  this must be a claim about the sampled truth sets plus the encoding, not
  about down-sets per se.* Status: speculative; needs matched pairs (4.4-T3).
- **H4 — Measure-side artifact.** The upward degrees may be attenuated or
  distorted: 57% zeros, the saturation issue (paper's discussion §7: models
  with verified submodels assigned to no set can saturate the measure), and
  the `pred ≡ 1` boundary condition (TODO at `measures.py` line ~58). If
  upward degrees are noisier measurements of "true" upward monotonicity, the
  upward coefficient is attenuated toward zero. Status: partially supported
  (distributional pathologies confirmed); the generalized-switches variant of
  the measure (discussed with Steinert-Threlkeld) is the fix to compare.
- **H5 — Genuine learner preference confined to this LoT's meanings.** Even
  after H1–H4, downward-monotone meanings *as sampled by this grammar* may
  simply have simpler statistical structure (e.g., closer to cardinality
  thresholds). *Mechanism: confound between direction and meaning simplicity
  within the sample.* Status: partially addressed (effect survives operator
  controls, §2), but truth-set-level covariates were never controlled.

### 4.4 Discriminating experiments, in cost order

- **T1 (hours; existing data + universe re-evaluation). Negation-pair test.**
  Find pairs (e, `not(e)`) — or more generally pairs whose truth vectors are
  complements — by evaluating all 2,000 expressions on the universe
  (`quantifier.py` / `get_verified_models` machinery). Within complement
  pairs, the decision boundary is identical and monotonicity direction flips.
  If AUC is ~equal within pairs → geometry/label symmetry holds empirically,
  and the asymmetry must be composition/measure-side (H1/H4/H5). If AUC
  differs systematically → the encoding or example-sampling breaks the
  symmetry (H2/H3), which is a new finding about the learner. **Do this
  first; every other hypothesis's interpretation depends on it.**
- **T2 (hours). Truth-set-statistics mediation.** For each expression compute
  p(true), mean/variance of |model| among positives, boundary size (number of
  edge pairs in the subset order crossing the truth boundary). Regress AUC on
  direction + these stats: if the downward effect collapses when positive-
  example cardinality stats enter, H2 is the driver. All computable without
  training.
- **T3 (days; small training runs). Matched minimal pairs.** Construct
  explicit up-set/down-set complement pairs and cardinality-threshold
  families (`≥k` vs `≤k` — mirror images under complementation of B), train
  the existing pipeline on just these (~dozens of quantifiers), compare
  learning curves. Direct test of H3 free of grammar confounds.
- **T4 (cluster; the decisive one). Polarity-balanced regeneration** — Phase 4
  of `EXPERIMENT_PLAN.md` with the `M`-leaf grammar. Pre-registered
  predictions 6–7 in the plan state the expected outcome: downward
  coefficient stable, aggregate degree–AUC r attenuates.
- **T5 (measure work). Generalized-switches degrees** for all 2,000
  expressions; re-run the directional analyses; if upward's effect appears
  under the better measure, H4 was a major driver.

## 5. QUESTION 3 — The gap in (0, 0.15]: measure, sampling, or something else?

**Answer so far (from `9i_monotonicity_gap`, verified): it is a joint
product — the measure is bimodal *on crisp inputs*, and the grammar supplies
only crisp inputs.** Neither alone suffices: noise-corrupted quantifiers fill
the gap easily (so it is not the measure alone), and each directional sense
has ~6.6% of expressions in the gap while `degree` has 1.7% (so the max-of-
four aggregation amplifies but does not create it). The only crisp route into
the gap is exact-value ("equality-type") predicates; the 34 real gap
expressions are enriched 3.0× in `not` and 2.2× in `equals`, exemplar:
`not(equals(cardinality(A), cardinality(B)))` at degree 0.0087.

**What remains open for this session:**
1. **Re-run the family simulation on the real model space** with the actual
   `measures.py` (the review used a simplified 256-subset lattice; the real
   universe has A/B/M zone assignments and a different predecessor structure).
   Confirm the bimodality-on-crisp-inputs claim transfers.
2. **Audit the boundary conditions**: label every expression for (a) the
   `pred ≡ 1` condition (the measures.py TODO), (b) saturation cases from the
   paper's discussion. How much of the atom at 0 is each?
3. **Grammar predictions** (ties to Q1): equality-free pseudo-grammar → gap
   empties; equality-rich → gap fills roughly linearly in `equals` weight
   (plan §5.4 prediction 2). This is the clean test of "sampling-driven".
4. **Generalized-switches measure**: does the gap persist under the
   alternative measure? If yes, the crisp-bimodality account generalizes; if
   the low range fills in, part of the gap was measure-specific.

## 6. Cross-cutting notes

- Keep the **universe fixed** for any degree comparison; the measure is not
  comparable across universes.
- **Dedup policy is an estimand choice**: per-expression (weighted) vs
  per-meaning (support) — report both; degree is a function of meaning only.
- The published-figure caveat when writing anything up: reproduction gives
  r = −0.333 vs published −0.3453 (post-publication monotonicity fix);
  Table 6 reproduces to ≤0.2%. The repo's `paper_figure1` = published
  **Figure 2**.
- A verified figure gallery with methodology vignettes from the review lives
  at: https://claude.ai/code/artifact/4ca5c8e6-2314-4ed2-b14d-fb3ba53411e2
- Context for the collaboration: Steinert-Threlkeld requested the
  length-vs-learning result (delivered; r ≈ +0.22–0.23) for a book-chapter
  in a Logic & AI anthology (van Benthem co-editing), and suggested the
  regression designs now implemented in `deeper_analysis.py` §7–8. The
  generalized-switches monotonicity variant and the saturation issue were
  flagged in that exchange as Chris's follow-ups.

## 7. Suggested opening moves for the new session

1. `conda activate altk`; run all three analysis scripts from this directory;
   confirm outputs match `analysis/tables/` (regression test of the port).
2. Fix or re-clone `.git`, branch (`monotonicity-followup`), commit the
   ported stack as the baseline.
3. Q2-T1 (negation-pair test) — highest information per hour.
4. Q1 Phase 0 (add `M` + integer constants; anchor checks) — unblocks
   everything else.
5. Q3 item 1 (real-universe simulation) — reuses Phase 0 machinery.

## 8. STATUS ADDENDUM (2026-07-26 session)

Done this session (details in `REPRODUCE.md` §4c and `analysis/tables/11*/12*`):

- **§7.1 regression test passed** — all three scripts regenerate
  `analysis/tables/` byte-identically in this repo (`ultk-fresh` clone; git
  works here, branch `learn-quant-paper-reproduction`).
- **EXPERIMENT_PLAN.md actually ported** (it was referenced but missing).
- **Q2-T1 DONE** (`scripts/negation_pair_test.py`): 44 complement pairs among
  the trained 2,000; 41 survive functional verification on training-style
  models (3 are complements only on the 256-model universe — mind this
  universe/training-distribution gap). Within pairs, AUC is ~equal
  (mean-arch Δ = +16, p = 0.59); the population directional model predicts
  +182/+558 (all/contrast pairs); observed/predicted ≈ 0.12. **Verdict: label
  symmetry holds empirically; ≥~88% of the directional asymmetry is
  composition/measure-side (H1/H4/H5), not boundary-level (H2/H3 within
  pairs).** LSTM shows a weak non-significant hint (+76, p = 0.16). Also H4
  evidence: the mirror identity up(e)=down(¬e) fails for half the pairs
  (max dev 0.79) — the directional degrees are noisy measurements.
- **Q2-T2 DONE** (`scripts/truth_set_stats.py`): class-conditional input
  statistics on 4,000 training-style models explain R² = 0.54 of AUC
  (directions: 0.13); downward β shrinks 80% (−708 → −138/SD, residual
  p = 0.002) under stats control. Dominant mediator: **class separation**
  (distance between positive/negative mean zone-count vectors), β −1500/SD,
  r = −0.68 with AUC, r = +0.48 with downward degree. **Verdict: the downward
  advantage is mostly generalized-H2 — downward-as-sampled quantifiers have
  more input-separable classes.** H5's "simpler statistical structure"
  is now concrete and measurable as class_sep.

Next in line: T5/generalized-switches (H4 now has direct supporting
evidence), Q1 Phase 0, Q3 item 1. T3 (matched minimal pairs) is partly
de-prioritized: T1 already bounds the boundary-level effect at ~12%.
