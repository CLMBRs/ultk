"""
Benchmark script comparing Python and Rust implementations of grammar enumeration.

This script tests both implementations on the indefinites grammar to measure
the performance improvement from the Rust implementation.
"""

import time
import sys
from pathlib import Path

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from examples.indefinites.grammar import indefinites_grammar
from examples.indefinites.meaning import universe as indefinites_universe


def benchmark_python(depth: int = 5):
    """Benchmark the Python implementation."""
    print(f"Benchmarking Python implementation at depth {depth}...")

    start = time.time()
    expressions_by_meaning = indefinites_grammar.get_unique_expressions(
        depth,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )
    elapsed = time.time() - start

    print(f"  Generated {len(expressions_by_meaning)} unique expressions")
    print(f"  Time: {elapsed:.2f} seconds")
    return elapsed, len(expressions_by_meaning)


def benchmark_rust(depth: int = 5):
    """Benchmark the Rust implementation."""
    try:
        from ultk_grammar import RustGrammar
    except ImportError:
        print("Rust extension not available. Run build.sh first.")
        return None, None

    print(f"Benchmarking Rust implementation at depth {depth}...")

    # Convert to Rust grammar
    rust_grammar = RustGrammar.from_python_grammar(indefinites_grammar)

    start = time.time()
    expressions_by_meaning = rust_grammar.get_unique_expressions(
        depth,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )
    elapsed = time.time() - start

    print(f"  Generated {len(expressions_by_meaning)} unique expressions")
    print(f"  Time: {elapsed:.2f} seconds")
    return elapsed, len(expressions_by_meaning)


def main():
    """Run benchmarks at various depths."""
    print("=" * 60)
    print("ULTK Grammar Enumeration Benchmark")
    print("Comparing Python vs Rust implementations")
    print("=" * 60)
    print()

    depths = [3, 4, 5]

    for depth in depths:
        print(f"\n{'=' * 60}")
        print(f"Depth: {depth}")
        print("=" * 60)

        py_time, py_count = benchmark_python(depth)
        print()

        rust_time, rust_count = benchmark_rust(depth)
        print()

        if rust_time is not None:
            speedup = py_time / rust_time
            print(f"Speedup: {speedup:.2f}x")

            if py_count != rust_count:
                print(
                    f"WARNING: Different counts! Python: {py_count}, Rust: {rust_count}"
                )
        else:
            print("Rust benchmark skipped (extension not available)")

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
