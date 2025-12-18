# ULTK Grammar Rust Extension

High-performance Rust implementation of ULTK's grammar enumeration, specifically optimizing `Grammar.get_unique_expressions()` which is a performance bottleneck due to combinatorial explosion and many Python function calls.

## Quick Start

```bash
# 1. Build the extension
cd rust-grammar
./build.sh

# 2. Run the benchmark
python benchmark_rust.py

# 3. Try the example
python example_usage.py
```

## Overview

The Python implementation of `get_unique_expressions()` becomes slow for several reasons:

1. **Combinatorial explosion**: Enumerating all grammatical expressions up to depth N grows exponentially
2. **Python function call overhead**: Each expression evaluation requires calling Python lambdas  
3. **Object creation overhead**: Creating many `GrammaticalExpression` objects in Python
4. **Hash/comparison operations**: Frequent dictionary lookups and comparisons

This Rust implementation aims to address these issues through:
- Native compiled code (potentially 10-100x faster than Python loops)
- Zero-copy data structures where possible
- Efficient hash maps (ahash) optimized for small keys
- Minimal allocations through careful memory management

## Current Status

⚠️ **Prototype Implementation** - This is a working proof-of-concept demonstrating the approach.

### What Works
- ✅ Core Rust data structures (`Rule`, `GrammaticalExpression`, `Grammar`)
- ✅ Enumeration logic with caching (`enumerate_at_depth`)
- ✅ PyO3 bindings for Python interoperability
- ✅ Build configuration with maturin
- ✅ Benchmark script for performance comparison

### Known Limitations
1. **Python callback overhead**: The `unique_key` and `compare_func` still call back to Python, adding overhead
2. **Expression evaluation**: The `evaluate()` method needs to call Python functions for rule semantics
3. **Not fully integrated**: Requires some adaptation to use with existing ULTK code
4. **Testing needed**: Comprehensive testing to ensure parity with Python implementation

### Expected Performance
For the indefinites example (6 referents, ~10 rules):
- **Depth 3-4**: 2-5x speedup (limited by Python callbacks)
- **Depth 5+**: 5-10x speedup (enumeration becomes dominant)

For larger grammars and universes, speedups should be more dramatic as the pure enumeration cost dominates.

## Installation

### Prerequisites

1. Install Rust (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Ensure you're in the ULTK environment with dependencies installed

### Build

```bash
cd rust-grammar
./build.sh
```

This will:
- Install maturin (if needed)
- Compile the Rust code in release mode
- Install the Python extension in your environment

## Usage

### Drop-in Replacement (with fallback)

```python
from ultk.language.grammar.grammar import Grammar
from ultk_grammar import use_rust_if_available

# Load grammar as usual
grammar = Grammar.from_yaml("grammar.yml")

# Optionally accelerate with Rust (falls back to Python if not available)
grammar = use_rust_if_available(grammar)

# Use as normal - Rust will be used if available
expressions = grammar.get_unique_expressions(
    depth=5,
    unique_key=lambda expr: expr.evaluate(universe),
    compare_func=lambda e1, e2: len(e1) < len(e2),
)
```

### Direct Usage

```python
from ultk_grammar import RustGrammar

# Convert existing grammar
rust_grammar = RustGrammar.from_python_grammar(python_grammar)

# Or with fallback
rust_grammar = RustGrammar.from_python_grammar(python_grammar, fallback=True)
```

## Files

```
rust-grammar/
├── Cargo.toml              # Rust package configuration
├── pyproject.toml          # Python package configuration
├── build.sh                # Build script
├── README.md               # This file
├── IMPLEMENTATION.md       # Detailed implementation notes
├── src/
│   └── lib.rs             # Main Rust implementation
├── python/
│   └── ultk_grammar/
│       └── __init__.py    # Python wrapper
├── benchmark_rust.py       # Performance comparison script
└── example_usage.py        # Usage example
```

## Benchmarking

Run the benchmark to compare Python vs Rust:

```bash
python benchmark_rust.py
```

This will test both implementations at depths 3, 4, and 5 on the indefinites grammar.

## Architecture

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for detailed architecture notes.

**Key components:**
- `RustGrammar`: Main class with optimized enumeration
- `enumerate_at_depth()`: Core recursive function with caching
- `GrammaticalExpression`: Lightweight expression tree structure
- PyO3 bindings: Bridge between Rust and Python

**Key optimizations:**
- Memoization of `(depth, lhs)` -> expressions
- Arc (shared references) for child expressions
- Fast hashing with ahash
- Pre-allocated vectors where possible

## Future Improvements

To achieve maximum speedups, the following would help:

1. **Native evaluation**: Implement common patterns (boolean logic) directly in Rust
2. **Reduced callbacks**: Move `unique_key` logic into Rust
3. **Parallelization**: Use rayon for parallel enumeration at high depths  
4. **Memory optimization**: Object pools and arena allocation
5. **Incremental evaluation**: Cache partial evaluations across expressions

## Troubleshooting

**Build fails**
- Ensure Rust is installed: `rustc --version`
- Try `cargo build` in rust-grammar directory for detailed errors

**Import error**
- Run `./build.sh` from the rust-grammar directory
- Check you're in the correct Python environment

**Slower than expected**
- Profile with `cProfile` to see where time is spent
- Python callbacks (`unique_key`, `compare_func`) should be very fast
- Try larger depths where enumeration dominates

## Contributing

Contributions welcome! Priority areas:
1. Comprehensive testing
2. Reducing Python callback overhead
3. Native evaluation implementations
4. Parallelization
5. Integration improvements
