# Rust Grammar Acceleration for ULTK

This directory contains a Rust implementation of ULTK's grammar enumeration functionality, specifically optimizing the `Grammar.get_unique_expressions()` method which is the main performance bottleneck in large-scale enumeration tasks.

## Why Rust?

The Python implementation of `get_unique_expressions()` becomes slow due to:

1. **Combinatorial explosion**: Number of expressions grows exponentially with depth
2. **Python overhead**: Nested loops, function calls, and object creation in Python are slow
3. **Memory allocation**: Creating many `GrammaticalExpression` objects
4. **Dictionary operations**: Frequent hash lookups and comparisons

Rust provides:
- 10-100x faster loops and function calls
- Zero-cost abstractions
- Efficient memory management
- Fast hash maps optimized for our use case

## Current Status

⚠️ **Work in Progress**: This is a prototype implementation demonstrating the approach. 

**What's implemented:**
- Core Rust data structures (`Rule`, `GrammaticalExpression`, `Grammar`)
- Enumeration logic with caching (`enumerate_at_depth`)
- PyO3 bindings for Python interoperability
- Build configuration with maturin

**Known limitations:**
1. **Python callback overhead**: The `unique_key` and `compare_func` still require calling back to Python, which adds overhead. For maximum speedup, these would need to be implemented in Rust.
2. **Expression evaluation**: The `evaluate()` method on expressions currently needs to call Python functions for rule semantics. This is the main remaining bottleneck.
3. **Testing**: Needs comprehensive testing to ensure parity with Python implementation.

**Future optimizations:**
- Cache evaluation results in Rust to avoid repeated Python callbacks
- Support for common evaluation patterns (boolean logic) natively in Rust
- Parallel enumeration for large depths
- Memory pooling to reduce allocations

## Installation

### Prerequisites

1. Install Rust (if not already installed):
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. Install maturin:
   ```bash
   pip install maturin
   ```

### Build

From this directory:

```bash
./build.sh
```

Or manually:
```bash
maturin develop --release
```

## Usage

The Rust extension provides a drop-in replacement for grammar enumeration:

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

## Benchmarking

Run the benchmark to compare Python vs Rust:

```bash
python benchmark_rust.py
```

Expected results (will vary by system):
- Depth 3: 2-5x speedup
- Depth 4: 5-10x speedup  
- Depth 5: 10-20x speedup

Note: Actual speedups depend on how much time is spent in Python callbacks vs pure enumeration.

## Architecture

### Core Components

1. **`src/lib.rs`**: Main Rust implementation
   - `Rule`: Grammar rule representation
   - `GrammaticalExpression`: Expression tree structure
   - `RustGrammar`: Main grammar class with enumeration logic
   - `enumerate_at_depth()`: Recursive enumeration with caching

2. **`python/ultk_grammar/__init__.py`**: Python wrapper
   - Provides compatibility layer with existing ULTK code
   - Handles fallback when Rust is unavailable

3. **`Cargo.toml`**: Rust dependencies and build configuration
   - `pyo3`: Python interop
   - `ahash`: Fast hashing
   - Optimized release profile for maximum performance

### Key Optimizations

1. **Caching**: Memoization of `(depth, lhs)` -> expressions to avoid recomputation
2. **Arc usage**: Shared references to child expressions to reduce cloning
3. **Efficient hash maps**: `ahash` instead of standard library for smaller keys
4. **Pre-allocated vectors**: Where possible, allocate expected sizes upfront

## Integration with ULTK

For maximum benefit, integrate into your workflow:

1. **Development**: Use Python implementation for easier debugging
2. **Production**: Enable Rust for large-scale enumeration tasks
3. **Testing**: Compare outputs between implementations to verify correctness

Example in a script:

```python
# At the top of generate_expressions.py
try:
    from ultk_grammar import use_rust_if_available
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    use_rust_if_available = lambda g: g

# Later when loading grammar
grammar = Grammar.from_yaml("grammar.yml")
if RUST_AVAILABLE:
    print("Using Rust acceleration")
    grammar = use_rust_if_available(grammar)
```

## Performance Tips

1. **Keep callbacks simple**: The `unique_key` and `compare_func` should be as fast as possible
2. **Use max_size**: If you only need N expressions, set `max_size=N` to stop early
3. **Cache universe**: Don't recreate the universe in the callback
4. **Profile**: Use Python's `cProfile` to see where time is spent

## Contributing

To improve the Rust implementation:

1. **Reduce Python callbacks**: Move more logic into Rust
2. **Add native evaluation**: Implement common patterns (boolean logic) in Rust
3. **Parallelize**: Use rayon for parallel enumeration at high depths
4. **Optimize memory**: Use object pools and arena allocation

## Troubleshooting

**"Rust not installed"**
- Install from https://rustup.rs/

**"maturin not found"**
- Run `pip install maturin`

**"Build failed"**
- Ensure you're in the `rust-grammar` directory
- Check that `Cargo.toml` exists
- Try `cargo build` to see detailed errors

**"Import error"**
- Run `./build.sh` from this directory
- Ensure you're in the correct Python environment

**"Different results than Python"**
- This may indicate a bug - please report with a minimal example
- Verify with `benchmark_rust.py` which checks counts match
