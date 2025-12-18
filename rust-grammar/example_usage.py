"""
Example of using the Rust-accelerated grammar enumeration.

This demonstrates how to integrate the Rust extension with existing ULTK code.
"""

import sys
from pathlib import Path

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from examples.indefinites.grammar import indefinites_grammar
from examples.indefinites.meaning import universe as indefinites_universe
from ultk.language.grammar.grammar import GrammaticalExpression


def use_rust_grammar():
    """Example of using Rust acceleration with the indefinites grammar."""

    # Import the Rust extension
    try:
        from ultk_grammar import use_rust_if_available

        print("✓ Rust extension available")
    except ImportError:
        print("✗ Rust extension not available (using Python fallback)")
        print("  To install: cd rust-grammar && ./build.sh")
        use_rust_if_available = lambda g: g  # fallback identity function

    # Convert to Rust-accelerated grammar (or fallback to Python)
    grammar = use_rust_if_available(indefinites_grammar)

    print(f"\nEnumerating expressions up to depth 4...")
    print(f"Universe size: {len(indefinites_universe)} referents")

    # This will use Rust if available, otherwise Python
    expressions_by_meaning = grammar.get_unique_expressions(
        4,
        max_size=2 ** len(indefinites_universe),
        unique_key=lambda expr: expr.evaluate(indefinites_universe),
        compare_func=lambda e1, e2: len(e1) < len(e2),
    )

    # Filter out trivial meanings
    for meaning in list(expressions_by_meaning.keys()):
        if meaning.is_uniformly_false():
            del expressions_by_meaning[meaning]

    print(f"Generated {len(expressions_by_meaning)} unique expressions")

    # Show a few examples
    print("\nExample expressions:")
    for i, (meaning, expr) in enumerate(list(expressions_by_meaning.items())[:5]):
        print(f"  {i+1}. {expr} (length: {len(expr)})")

    return expressions_by_meaning


if __name__ == "__main__":
    use_rust_grammar()
