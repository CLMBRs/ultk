# Integrating Rust Grammar Acceleration into ULTK

This guide shows how to optionally use the Rust grammar extension in your ULTK projects.

## For End Users

If you're using ULTK for research and want faster expression generation:

### Installation

```bash
# From the ULTK root directory
cd rust-grammar
./build.sh
cd ..
```

### Usage in Scripts

Modify your existing scripts to optionally use Rust:

```python
# At the top of your script (e.g., generate_expressions.py)
try:
    from ultk_grammar import use_rust_if_available
    RUST_ENABLED = True
except ImportError:
    RUST_ENABLED = False
    def use_rust_if_available(g): 
        return g

# When loading your grammar
grammar = Grammar.from_yaml("grammar.yml")
grammar = use_rust_if_available(grammar)  # Accelerates if available

# Rest of your code remains the same
expressions = grammar.get_unique_expressions(
    depth=5,
    unique_key=lambda expr: expr.evaluate(universe),
    compare_func=lambda e1, e2: len(e1) < len(e2),
)
```

This pattern:
- Uses Rust acceleration if available
- Falls back to Python implementation if not
- Requires no other code changes

## For ULTK Developers

### Project Structure

The Rust extension is organized as a separate optional component:

```
ultk/
├── src/
│   ├── ultk/              # Main Python library
│   ├── examples/          # Research examples
│   └── tests/             # Tests
└── rust-grammar/          # Optional Rust extension
    ├── src/lib.rs         # Rust implementation
    └── python/            # Python wrapper
```

### Integration Points

The Rust extension interfaces with ULTK at the `Grammar.get_unique_expressions()` method:

**Python side** (`ultk/language/grammar/grammar.py`):
- Defines the API contract
- Provides the reference implementation
- Used when Rust is not available

**Rust side** (`rust-grammar/src/lib.rs`):
- Reimplements core enumeration logic
- Accepts Python callbacks for flexibility
- Falls back gracefully if build fails

### Adding to pyproject.toml (Optional)

To make Rust extension an optional dependency:

```toml
[project.optional-dependencies]
rust = ["ultk-grammar-rust"]

[tool.setuptools.packages.find]
where = ["src", "rust-grammar/python"]
```

Then users can install with:
```bash
pip install ultk[rust]
```

### Testing

To ensure Rust and Python implementations agree:

```python
def test_rust_parity():
    """Test that Rust produces same results as Python."""
    from ultk.language.grammar.grammar import Grammar
    
    grammar_py = Grammar.from_yaml("test_grammar.yml")
    
    try:
        from ultk_grammar import RustGrammar
        grammar_rust = RustGrammar.from_python_grammar(grammar_py)
        
        results_py = grammar_py.get_unique_expressions(...)
        results_rust = grammar_rust.get_unique_expressions(...)
        
        assert len(results_py) == len(results_rust)
        # More detailed comparisons...
    except ImportError:
        pytest.skip("Rust extension not available")
```

### Performance Testing

Add benchmarks to CI/CD:

```yaml
# .github/workflows/benchmark.yml
- name: Benchmark Rust extension
  run: |
    cd rust-grammar
    ./build.sh
    python benchmark_rust.py
```

## For Example Authors

When creating new examples in `src/examples/`:

### Template Script

```python
"""
generate_expressions.py - Generate unique expressions for this domain
"""

from ..grammar import my_grammar
from ..meaning import universe

# Optional Rust acceleration
try:
    from ultk_grammar import use_rust_if_available
    my_grammar = use_rust_if_available(my_grammar)
    print("Using Rust acceleration")
except ImportError:
    print("Using Python implementation")

if __name__ == "__main__":
    expressions = my_grammar.get_unique_expressions(
        depth=5,
        max_size=2 ** len(universe),
        unique_key=lambda expr: expr.evaluate(universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )
    
    print(f"Generated {len(expressions)} unique expressions")
    # ... rest of script
```

### Documentation

In your example's README.md:

```markdown
## Performance

For large-scale generation, you can optionally use Rust acceleration:

\`\`\`bash
cd ../../rust-grammar
./build.sh
cd -
\`\`\`

This can provide 5-20x speedup for complex grammars at high depths.
```

## Configuration Management

### For Hydra-based Examples

In your config file:

```yaml
# conf/expressions.yaml
grammar:
  use_rust: true  # Try to use Rust if available
  fallback: true  # Fall back to Python if Rust unavailable
  
generation:
  depth: 5
  # ...
```

In your script:

```python
@hydra.main(config_path="conf", config_name="expressions")
def main(cfg):
    grammar = load_grammar(cfg.grammar.path)
    
    if cfg.grammar.use_rust:
        try:
            from ultk_grammar import use_rust_if_available
            grammar = use_rust_if_available(grammar)
            logger.info("Using Rust acceleration")
        except ImportError:
            if not cfg.grammar.fallback:
                raise
            logger.warning("Rust not available, using Python")
```

## Deployment Considerations

### When to Use Rust

**Good cases:**
- Large grammars (>20 rules)
- Deep enumeration (depth > 4)
- Large universes (>10 referents)
- Production/publication runs

**May not help:**
- Small grammars
- Shallow enumeration (depth ≤ 3)
- When evaluation dominates (complex `unique_key`)
- Rapid prototyping/debugging

### Distribution

When distributing code that uses Rust acceleration:

1. **Make it optional**: Always provide Python fallback
2. **Document setup**: Include build instructions in README
3. **CI/CD**: Test both paths (with and without Rust)
4. **Binary wheels**: Consider pre-building for common platforms

### Example Makefile

```makefile
# Makefile for ULTK project

.PHONY: install install-rust test benchmark

install:
	pip install -e .

install-rust:
	cd rust-grammar && ./build.sh

test:
	cd src/tests && pytest

benchmark:
	cd rust-grammar && python benchmark_rust.py

all: install install-rust test
```

## Monitoring Performance

Add timing to your scripts:

```python
import time

start = time.time()
expressions = grammar.get_unique_expressions(...)
elapsed = time.time() - start

logger.info(f"Generated {len(expressions)} expressions in {elapsed:.2f}s")
```

Compare runs with and without Rust to measure benefit for your specific use case.

## FAQ

**Q: Will this become required?**
A: No, it will always be optional. The Python implementation is the reference.

**Q: What if Rust build fails?**
A: Code should gracefully fall back to Python implementation.

**Q: How much faster is it?**
A: Depends on your grammar/universe. Benchmark with your specific case. Typically 2-20x.

**Q: Can I use this in published research?**
A: Yes, but document which version you used and that results are identical to Python.

**Q: What about other performance improvements?**
A: Rust is one option. Also consider: caching, smaller grammars, Numba for callbacks.
