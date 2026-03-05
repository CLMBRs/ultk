# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ULTK (Unnatural Language ToolKit) is a Python library for computational semantic typology research — specifically for "efficient communication" analyses that explain natural language structure in terms of competing pressures: minimizing cognitive complexity vs. maximizing communicative accuracy.

## Commands

```bash
# Install all dependencies (including dev group for tests)
uv sync --group dev

# Run all tests
uv run pytest src/tests/

# Run a single test file
uv run pytest src/tests/test_language.py

# Run a single test by name
uv run pytest src/tests/test_language.py::TestLanguage::test_name

# Format code (Black is enforced via CI on PRs)
black src/
```

Tests are discovered automatically by pytest from `src/tests/`. The CI workflow runs `uv run pytest src/tests/` from the repo root.

## Architecture

### Two Main Modules

**`ultk.language`** — Core data structures for semantic representations:
- `semantics.py`: `Referent` (immutable semantic object), `Universe` (collection of Referents with a prior distribution), `Meaning` (mapping from Universe to arbitrary type T — e.g., booleans for truth values)
- `language.py`: `Expression` (form + meaning pair), `Language` (frozenset of Expressions sharing a Universe). Helper `aggregate_expression_complexity()` bridges language and effcomm.
- `sampling.py`: Generators for all meanings, expressions, and languages from a universe — used to enumerate the full hypothesis space.
- `grammar/`: A probabilistic context-free grammar (PCFG) framework for building expressions as programs in a Language of Thought. `grammar.py` defines `Rule` and `Grammar`/`GrammaticalExpression`; `likelihood.py` provides scoring functions; `inference.py` handles MDL/Bayesian inference.

**`ultk.effcomm`** — Efficient communication analysis tools:
- `agent.py`: RSA (Rational Speech Act) agents — `LiteralSpeaker`, `LiteralListener`, `PragmaticSpeaker`, `PragmaticListener` — represented as weight matrices.
- `informativity.py`: `informativity()` and `communicative_success()` — compute how well a language supports communication (vectorized as `diag(prior) @ S @ R ⊙ U`).
- `tradeoff.py`: Pareto front computation (`pareto_optimal_languages`, `non_dominated_2d`, `dominates`) for simplicity/informativeness trade-off analysis.
- `optimization.py`: `EvolutionaryOptimizer` — iterative algorithm to approximate the Pareto frontier via mutations (`AddExpression`, `RemoveExpression`).
- `sampling.py`: `get_hypothetical_variants()` — generates null-hypothesis languages by permuting speaker weight matrices.
- `analysis.py`: Aggregation utilities for building results DataFrames.

**`ultk.util`**:
- `frozendict.py`: `FrozenDict` — an immutable dict used extensively as keys in frozen dataclasses.
- `io.py`: I/O helpers.

### Key Design Patterns

- Core objects (`Universe`, `Meaning`, `Expression`) are **frozen/immutable** (`@dataclass(frozen=True)` or manual `_frozen` flag), enabling hashing and use as dict keys.
- `Meaning` stores its mapping as a `tuple[T, ...]` indexed parallel to `Universe.referents`, with `_ref_to_idx` for O(1) lookup. Access via `meaning[referent]`.
- `Language` stores expressions as a `frozenset` — order-independent, hashable.
- Grammar rules are defined via Python type annotations; `Rule.from_callable()` introspects function signatures to build rules automatically.

### Examples

`src/examples/` contains complete worked analyses:
- `indefinites/` — efficient communication analysis of indefinite pronouns
- `modals/` — semantic universals for modals
- `learn_quant/` — quantifier learning
