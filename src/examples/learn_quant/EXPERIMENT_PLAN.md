# Experiment plan: how the LoT grammar shapes monotonicity

*Drafted 2026-07-24 during the reproduction review. Companion analyses:
`scripts/review_extensions.py` (esp. `9e_polarity_counterbalance`,
`9i_monotonicity_gap`) and the pilot regression in the appendix.*

## 1. Question and motivating results

**Question.** How does the choice of Language-of-Thought grammar — its operator
inventory and how often each operator is used — determine the distribution of
monotonicity degrees among the quantifiers it generates? Which operators (and
which operator *combinations*) push probability mass toward monotone, non-
monotone, or knife-edge quantifiers?

**What we already know (from the review analyses):**

1. The degree measure is **nearly bimodal on formula-generated ("crisp")
   truth sets**: an atom at exactly 0 (order-uninformative closures), an atom
   at 1 (genuinely monotone), a continuum from ~0.2 up, and a sparse zone
   (0, 0.15] reachable only by exact-value ("equality-type") predicates
   (`analysis/tables/9i_monotonicity_gap.txt`).
2. The current grammar's sample is **downward-dominant** (degree attained by a
   downward sense in 83% of expressions; 57% have zero upward degree), and
   this traces to the scarcity of polarity flippers (`not` in 15.8% of
   expressions) against a near-universal `subset_eq` skeleton — not to
   `difference` (`9e_polarity_counterbalance`).
3. A pilot ridge regression of degrees on operator counts (appendix) finds
   real, interpretable effects (`or` destroys monotonicity, `and` and
   `greater_than` increase it, `not` flips up vs down) but tops out at
   R² ≈ 0.09–0.16 — for reasons that are diagnostic (see §3).

**The organizing structural fact.** Monotonicity degree is a deterministic
function of the *truth set* (the meaning). Two expressions with the same
meaning have identical degrees no matter which operators built them. So the
grammar affects monotonicity **only by steering which meanings get sampled**:

> the monotonicity distribution = (grammar's induced prior over meanings)
> pushed through a fixed function.

Operators do not *cause* monotonicity; operator choices move the sampler
through meaning space. This dictates both what to measure (distributions, not
just means) and the dedup policy (§6).

## 2. Design overview: one maximal inventory, randomized production weights

Rather than enumerating discrete alternative grammars, we use:

- **One maximal operator inventory** — the current 11 operators plus the
  polarity-completing primitives absent today: `complement` (M \ X) and/or
  `implies` (co-difference, Xᶜ ∪ Y), optionally `not_equals` and
  `proper_subset`. This superset spans both monotonicity polarities at the
  set level, which the current inventory does not.
- **Randomized production weights** — instead of one fixed generation process,
  each batch of expressions is generated under a different, randomly drawn
  weight vector over the grammar's production rules. Grammar space is treated
  as a continuous simplex; each draw is one "pseudo-grammar".

This unifies the two angles of attack:

- **Angle 1 (expression level):** regress degree outcomes on operator counts
  and structural features, pooled across batches — now with a well-conditioned
  design matrix (§3).
- **Angle 2 (grammar level):** regress *distribution summaries* of each batch
  on its weight vector — a "grammar response surface", where the weights were
  assigned by us at random and contrasts are causal in the
  randomized-experiment sense.

Both come from the same runs. No neural training is involved anywhere in
Phases 0–3 (degrees are cheap to compute), so the whole design is CPU-scale.

## 3. Why randomize weights? (identification, in detail)

This section expands the reasoning, because it is the crux of the design.

### 3.1 What the current pipeline does

The existing pipeline generates expressions from the grammar **under one fixed
process** (depth-capped generation with fixed rule choices), then samples
2,000 of them. Every observed expression is a draw from a *single*
distribution over expressions. Whatever variation in operator counts we see
across the 2,000 — some have three `union`s, some none — is variation *that
this one process happens to produce*.

### 3.2 Two distinct couplings contaminate count-based attribution

**(a) Hard coupling — type constraints (structural).** The grammar is typed
(SET, INT, BOOL). `equals` and `greater_than` take INT arguments, and
`cardinality` is the only SET→INT bridge, so **every** `equals` or
`greater_than` token forces `cardinality` tokens beneath it; `index` likewise
requires an INT. `and`/`or`/`not` take BOOL arguments, which only
`subset_eq`/`equals`/`greater_than` (and their compositions) produce. These
are logical necessities: some regions of "operator-count space" contain no
grammatical expression at all (structural zeroes), and some count differences
are exactly linearly dependent. This is the same phenomenon that made the
operator→learning OLS rank-deficient (rank 10 on 11 counts) and forced the
Ridge.

**(b) Soft coupling — single-process sampling (statistical).** Even where
combinations are grammatical, one fixed process makes counts co-vary: longer
expressions have more of everything; the depth cap and the uniform choice
among licensed rules make high-fan-in types (`SET`) dominate; the
entropy/balance filter then removes a non-random slice of meanings. So
`op_or` and `op_subset_eq` and `func_count` all rise and fall together across
the sample for reasons that have nothing to do with monotonicity.

**What this does and does not break.** It does **not** bias any individual
measurement — each expression's degree is computed exactly from its truth
set. What it breaks is **attribution**: when counts co-move, a regression
cannot tell which operator in a correlated bundle is responsible (credit gets
split arbitrarily / shrunk by the regularizer), and for count combinations
the process never produces, the model's "effect" is pure extrapolation. This
is the standard observational-data problem: the design matrix is whatever the
data-generating process gave us, and here the process is a single grammar
with strong internal correlations. The pilot's modest R² and shrunken
coefficients are symptoms.

### 3.3 The remedy: make operator prevalence an *experimental treatment*

A probabilistic grammar (PCFG view) attaches a probability to each production
rule; when the sampler expands a nonterminal, it chooses among the applicable
rules according to those probabilities. The current setup is the special case
of one fixed probability vector. The proposal:

1. Let **w** be the vector of rule weights (one sub-simplex per nonterminal
   category: the SET-rules simplex, the BOOL-rules simplex, etc.).
2. For each batch *g* = 1…G, **draw w⁽ᵍ⁾ at random** (Dirichlet on each
   sub-simplex; plus a few deliberately extreme corner vectors for coverage —
   Latin-hypercube style).
3. Generate N expressions under w⁽ᵍ⁾ (recursive descent honoring types and
   the depth cap), dedupe, compute degrees.

Now an `or`-heavy pseudo-grammar, a `not`-saturated one, an `equals`-starved
one, etc., all appear in the data **by construction**. Two consequences:

- **Expression level:** the pooled data covers a far larger region of
  count space, and counts decorrelate *to the extent types permit* — the
  soft coupling (b) is broken because prevalence is now driven by the
  experimenter's randomization rather than one process's habits. Regression
  estimates stop being extrapolations from a thin, correlated cloud.
- **Grammar level:** w⁽ᵍ⁾ is randomized, hence independent of everything
  else by design. Any systematic difference in the monotonicity distribution
  between batches with different w is *caused* by w. This is the
  identification one fixed grammar can never provide: it is a randomized
  dose–response experiment where the "dose" is each operator's production
  weight.

**The honest limit: hard coupling survives randomization.** No weight vector
can produce `equals` without `cardinality` below it — that is logic, not
statistics. So expression-level effects are identified at the level of
**minimal grammatical bundles** (e.g., the `equals`∘`cardinality` bundle vs
the `greater_than`∘`cardinality` bundle vs bare `subset_eq`), not of every
operator in isolation. Randomization still helps precisely here: by
independently varying w_equals and w_greater_than, batches arise where
`cardinality` appears mostly under `equals` vs mostly under `greater_than`,
letting the model separate the bundles' effects — impossible under one fixed
grammar where the two contexts co-occur in fixed proportion. At the grammar
level there is no such limit at all: the response surface w → distribution is
identified regardless of downstream coupling, which simply becomes part of
the mechanism being summarized.

### 3.4 Answering the "how is this different from the current repo?" question

The current repo has **one** distribution over expressions; the design has
**G** distributions, indexed by weight vectors we chose at random. The
between-batch contrasts are new information — they are what let us say
"raising `complement`'s weight symmetrizes the upward/downward marginals"
as a causal statement rather than a correlation. And to the sub-question:
yes — certain functions *can only* occur together (type constraints), and
under a single process this skews not the monotonicity measurements
themselves but the **evidence base for attributing** monotonicity to
operators: bundled operators are confounded, and unobserved count regions are
unidentified. Randomizing weights populates the observable regions and turns
prevalence into a treatment; the remaining logical bundles are named and
reported as bundles.

### 3.5 Implementation options for the weighted sampler

- **Option A — PCFG sampler:** recursive-descent generation choosing rules
  with probabilities w, depth-capped, rejection for type dead-ends. Clean,
  requires a new sampler in `generate_expressions.py`.
- **Option B — log-linear tilt of a fixed pool (minimal code):** enumerate or
  sample one large expression pool once (as the repo already does), compute
  each expression's operator-count vector c(e), then for pseudo-grammar θ
  draw expressions with probability ∝ exp(θ·c(e)). This importance-tilting
  approximates rule-weight changes with no new sampler, reusing the existing
  pipeline; θ plays the role of log-weights. Recommended for a first pass;
  its main limitation is that it cannot up-weight expressions the base pool
  never generated (mitigate with a large, deep pool and the new primitives
  included at pool-generation time).

## 4. Sampling protocol

| parameter | value (first pass) | notes |
|---|---|---|
| operator inventory | current 11 + `complement`, `implies` | optionally `not_equals`, `proper_subset` |
| universe | identical to current (M ≤ 10, same referent encoding) | the measure is not comparable across universes — never vary this |
| depth cap | 5 (as current) | fixed across batches |
| G (pseudo-grammars) | 300 Dirichlet draws + ~20 designed corners | corners: current-grammar weights; `not`-heavy; `complement`-heavy; `equals`-heavy; `or`-free; threshold-only |
| N per batch | 200 expressions after dedup | ~60k expressions total |
| dedup | within batch by expression string AND by truth vector | report both weighted and support versions (§6) |
| outcomes per expression | 4 directional degrees, `degree`, truth-set stats (p(true), saturation flags) | all cheap; vectorize truth-table evaluation |
| filters | compute degrees for all; flag (don't drop) entropy-filter failures | report distributions with and without the filter |
| anchors | *all, some, no, most, an-even-number-of* re-derived under every batch that can express them | measure-invariance check: identical degrees everywhere |

## 5. Measurement and models

### 5.1 Expression level (angle 1)

Outcomes are lumpy ([0,1] with atoms at 0 and 1), so use
**zero-one-inflated beta (ZOIB) regression** per outcome
(`degree`, `upward`, `downward`): a logistic model for P(=0), a logistic for
P(=1), and a beta regression on the interior. Predictor sets, compared by
held-out deviance:

- **P1: operator counts** (continuity with the pilot).
- **P2: polarity-calculus features.** For each occurrence of A and B, walk
  its path to the root; each operator-argument slot is monotone (+), antitone
  (−), or non-monotone (blocked); the path polarity is the product, with
  "blocked" absorbing. Features: counts of +, −, and blocked occurrences per
  argument. This is Sánchez-Valencia/van Benthem monotonicity marking turned
  into regressors — it encodes *position*, which raw counts ignore, and is
  the theoretically expected interaction structure in closed form.
- **P3: P2 + residual operator counts** — does inventory add anything once
  composition is accounted for?

If P2 ≫ P1, composition (not inventory) carries the effect — a publishable
point on its own.

### 5.2 Interactions (three tiers)

1. **Analytic:** P2's path polarities *are* the principled interactions
   (operator effects composing through position and negation parity).
2. **Confirmatory:** pairwise count products under a hierarchical
   (strong-heredity) lasso — an interaction enters only if both main effects
   do. Bundle terms from §3.3 (e.g., `equals`∘`cardinality`) are named
   predictors here.
3. **Discovery:** gradient-boosted trees on all features with SHAP
   interaction values, used only to nominate unanticipated pairs for
   promotion into tier 2. Never reported as final estimates.

### 5.3 Grammar level (angle 2)

Unit = batch (weight vector). Outcomes = summaries of the batch's degree
distribution:

- shares: P(degree = 0), P(0 < degree ≤ 0.15) (gap), P(interior), P(degree ≥ 0.99)
- moments: mean upward, mean downward, mean(down − up) (the asymmetry)
- distributional distance: Wasserstein-1 to the current grammar's distribution
- same summaries on the deduped-by-meaning support (§6)

Model: GAM or Gaussian-process regression from log-weights to each summary —
the **grammar response surface**. Then **Sobol variance decomposition** on
the fitted surface yields, per primitive weight, a main-effect index and,
per pair, a second-order index — interaction magnitudes on a common variance
scale. This is the concrete answer to "how do we measure interactions and
magnitudes of primitives": Sobol S_k = share of between-grammar variance in a
summary explained by primitive k alone; S_jk = the extra share explained by
j and k varying together.

### 5.4 Pre-registered headline predictions

1. Raising `complement`/`implies` weight shrinks the downward−upward
   asymmetry toward 0 (from `9e`: the asymmetry traces to flipper scarcity).
2. Raising `equals` weight increases the gap share P(0 < d ≤ 0.15)
   roughly linearly (from `9i`: equality predicates are the only crisp gap
   route).
3. Raising `or` weight lowers mean degree and increases P(interior)
   (pilot: `or` is the strongest monotonicity destroyer).
4. A threshold-only corner grammar concentrates mass at degree ≈ 1.
5. P2 (polarity features) outperforms P1 (counts) substantially for
   `upward`/`downward`; the `not`-parity feature interacts with everything
   in P1 but is a main effect in P2.

**Predictions for Phase 4 (re-running the learning experiments):**

6. **Conditional effects transport; marginal correlations do not.** Basis
   (computed 2026-07-24 on the current data): the degree→AUC effect survives
   controlling all 11 operator counts + length (β −540/SD vs −639 with
   length only; mixed-model p = 2e-38), and the directional split persists
   under the same control (downward −571, p = 1e-39; upward +80, n.s.).
   Since the conditional effect is not an operator-composition artifact, we
   predict the *within-population* regression coefficients for downward
   monotonicity are approximately stable across engineered grammars, while
   the *marginal* Pearson r of degree vs AUC shifts with the population:
   - polarity-symmetrized grammar: aggregate degree–AUC r attenuates
     (perhaps toward −0.15…−0.2 from −0.33), because degree is then attained
     by the near-inert upward sense about half the time; downward–AUC r
     stays near −0.33;
   - equality-rich grammar: marginal r strengthens (adds low-degree,
     hard-to-learn mass);
   - threshold-only grammar: r collapses toward 0 by restriction of range,
     with no change in any learner property.
7. The paper's headline claim survives in sign everywhere but is sharpened:
   "downward-monotone quantifiers are easier to learn" is the
   population-invariant form; "more monotone → easier" in the aggregate-
   degree sense is partly a property of the current LoT's downward-dominant
   prior.

## 6. Dedup policy = two estimands

Degree is a function of meaning (§1), so:

- **Weighted (per-expression) distribution:** what a learner sampling from
  this LoT would encounter. Duplicated meanings count multiple times.
- **Support (per-meaning) distribution:** which meanings the LoT reaches at
  all within the depth cap.

Both are reported for every batch; the difference between them is itself a
grammar property (how much the grammar re-derives the same meanings).

## 7. Phases, deliverables, effort

| phase | work | deliverable | est. effort |
|---|---|---|---|
| 0 | add primitives to `grammar.yml`/`grammar.py`; anchor checks | extended grammar + `anchors_report.txt` | 1–2 days |
| 1 | weighted sampler (Option B first); batch generation; degree computation | `analysis/lot_sweep/expressions.parquet` (~60k rows) | 2–3 days |
| 2 | ZOIB fits P1/P2/P3; interaction tiers | tables + `lot_expression_models.png` | 2–3 days |
| 3 | response surface + Sobol | `grammar_response_surface.png`, sensitivity table | 1–2 days |
| 4 (optional) | train learners on 3 engineered grammars (polarity-symmetric, equality-rich, threshold-only) | learnability comparison; tests whether the downward advantage is grammar-borne | cluster time; separate decision |

## 8. Pitfalls checklist

- [ ] Universe fixed across all batches (measure comparability).
- [ ] Dedup policy reported both ways (§6).
- [ ] Entropy-filter effects reported (flag, don't silently drop).
- [ ] Saturation cases tracked as a labeled category.
- [ ] Bundles (type-forced co-occurrences) named; no causal claims about
      operators within a bundle at expression level.
- [ ] Corner grammars included so the surface is not extrapolated at edges.
- [ ] Seeds fixed; batches reproducible from (seed, w) pairs.

## Appendix: pilot — operators → monotonicity on the current grammar

Ridge (α = 10) of degree outcomes on z-scored operator counts, n = 2,000
unique expressions, 500-resample bootstrap 95% CIs; coefficients are
Δ-outcome per +1 SD of operator count. Significant terms only:

| outcome | R² | monotonicity ↓ | monotonicity ↑ |
|---|---|---|---|
| degree | 0.120 | or −0.055, cardinality −0.037, equals −0.037, index −0.032, union −0.032, subset_eq −0.023 | and +0.035, greater_than +0.028 |
| upward | 0.164 | or −0.093, subset_eq −0.057, intersection −0.041, index −0.037, difference −0.024, cardinality −0.019 | and +0.055, greater_than +0.031, **not +0.029** |
| downward | 0.089 | or −0.047, cardinality −0.036, equals −0.033, union −0.029, index −0.026, **not −0.025**, intersection −0.024, difference −0.017, subset_eq −0.013 | and +0.033, greater_than +0.019 |

Reading: `or` is the strongest monotonicity destroyer; `and` and
`greater_than` push degrees up; `equals` pulls toward the knife-edge low
range; `not` shows the flipper signature (+up, −down). The modest R² is the
motivation for §3 (counts are position-blind and the design matrix is
confounded), not a null result.
