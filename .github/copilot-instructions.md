# ULTK AI Agent Instructions

## Project Overview
ULTK (Unnatural Language ToolKit) is a Python library for **computational semantic typology** research, analyzing natural languages through the lens of efficient communication theory. The core thesis: natural languages optimize a trade-off between simplicity (cognitive complexity) and informativeness (communicative success).

## Architecture

### Two-Module Design
1. **`ultk.language`**: Primitives for semantic modeling
   - `semantics.py`: `Universe`, `Referent`, `Meaning` - the semantic space
   - `language.py`: `Expression`, `Language` - form-meaning mappings
   - `grammar/`: Language of Thought (LoT) grammars with `Rule` and `Grammar` classes for compositional semantics

2. **`ultk.effcomm`**: Efficient communication analysis tools
   - `agent.py`: RSA framework agents (`LiteralSpeaker`, `PragmaticListener`, etc.)
   - `informativity.py`: Measures communicative success between speakers/listeners
   - `optimization.py`: `EvolutionaryOptimizer` for Pareto frontier estimation
   - `sampling.py`: Language generation and permutation utilities
   - `tradeoff.py`: Pareto dominance detection and language measurement

### Data Flow Pattern
Typical workflow: Define Universe → Build Grammar → Generate Expressions → Create Languages → Measure Properties → Optimize Trade-offs

## Key Conventions

### Immutability & Hashing
- `Referent` objects are **frozen after initialization** (raises `AttributeError` on modification)
- `Language.expressions` is a `frozenset` for hashability
- `FrozenDict` (custom class) used for meanings to enable hashing

### Grammar Definition
Two approaches for defining LoT grammars:

1. **YAML format** (see `src/examples/indefinites/grammar.yml`):
```yaml
start: bool
rules:
  - lhs: bool
    rhs: [bool, bool]
    name: "and"
    func: "lambda p1, p2: p1 and p2"
```
Load: `Grammar.from_yaml("path/to/grammar.yml")`

2. **Python module** (type-annotated functions):
```python
@dataclass
class Rule:
    @classmethod
    def from_callable(cls, func: Callable) -> "Rule":
        # Extracts lhs from return annotation, rhs from parameters
```
Load: `Grammar.from_module("module.path")`

### Universe Construction
Use `Universe.from_dataframe(df)` where CSV has:
- Required: `name` column for referent identifiers
- Optional: any additional feature columns become `Referent` properties
- Example: `src/examples/indefinites/referents.csv`

## Examples as Documentation

### Study Structure Template
Each example (`indefinites/`, `modals/`, `learn_quant/`) follows this pattern:

1. **`grammar.py`**: Define domain grammar
2. **`meaning.py`**: Load universe from CSV
3. **`measures.py`**: Wrap ULTK functions for domain-specific complexity/informativity
4. **`scripts/generate_expressions.py`**: Enumerate shortest expressions per meaning
5. **`scripts/estimate_pareto.py`**: Run `EvolutionaryOptimizer.fit()` 
6. **`scripts/measure_natural_languages.py`**: Analyze real language data
7. **`scripts/combine_data.py`**: Merge outputs to CSV
8. **`scripts/analyze.py`**: Statistical analysis and plotting (uses `plotnine`)

### Configuration
- `learn_quant/` uses **Hydra** for config management (see `conf/` dir)
- Run with: `python -m learn_quant.scripts.generate_expressions recipe=3_3_3_xi.yaml`
- Override params: `++universe.m_size=4`

## Development Workflows

### Testing
Run from `src/tests/`: `pytest`
- Uses `pytest` framework
- Test files: `test_grammar.py`, `test_language.py`, `test_semantics.py`, etc.

### Running Examples
**Always run from `src/examples/` directory** using module syntax:
```bash
cd src/examples
python -m indefinites.scripts.generate_expressions
python -m indefinites.scripts.estimate_pareto
```

### Installation
Editable install (development): `pip install -e .` from project root

## Type Patterns

### Generic Type Parameters
`Expression`, `Meaning`, and `Universe` are generic over referent types:
```python
T = TypeVar("T")
class Expression(Generic[T]):
    form: str
    meaning: Meaning[T]
```

### Agent Matrices
Agents use numpy arrays with shape `(num_referents, num_expressions)`:
- Access mappings via `_referent_to_index` and `_expression_to_index` dicts
- Matrix columns can be permuted to generate language variants (`sampling.py`)

## Domain-Specific Notes

### Grammar Enumeration
`Grammar.enumerate_up_to_depth()` parameters:
- `key`: function to extract unique identifier (typically `lambda expr: expr.evaluate(universe)`)
- `comp`: comparison function for duplicates (typically `lambda e1, e2: len(e1) < len(e2)`)
- Returns shortest expression for each unique meaning

### Complexity Measures
Standard approach: `len(expression)` counts composition depth
Alternative: `expression.atom_count()` counts terminal nodes only

### Pragmatic vs Literal Agents
Toggle in `informativity()` function:
```python
informativity(lang, prior, agent_type="literal")   # default
informativity(lang, prior, agent_type="pragmatic")  # RSA agents
```

## Common Pitfalls

1. **File paths**: Examples expect execution from `src/examples/`, not project root
2. **Empty languages**: `Language([])` raises `ValueError` - languages must have ≥1 expression
3. **Universe consistency**: All expressions in a language must share the same `Universe`
4. **YAML grammar functions**: Must be valid Python lambdas as strings (parsed with `eval`)
5. **Referent types**: Terminal rules expect single `Referent` argument, treated as `rhs=None`

## Current State
- Python 3.7+ required, tested on 3.11
- Dependencies: numpy, pandas, plotnine, pyyaml, nltk, pathos
- Documentation: https://clmbr.shane.st/ultk
- Published: SCiL 2025 proceedings
