#!/usr/bin/env python3
"""
Quick test to verify the Rust extension builds and runs correctly.
Run this after building to ensure everything works.
"""

import sys
from pathlib import Path

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import():
    """Test that the extension can be imported."""
    try:
        from ultk_grammar import RustGrammar, use_rust_if_available

        print("✓ Extension imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import extension: {e}")
        return False


def test_basic_usage():
    """Test basic functionality."""
    try:
        from ultk_grammar import RustGrammar
        from examples.indefinites.grammar import indefinites_grammar

        # Convert to Rust
        rust_grammar = RustGrammar.from_python_grammar(indefinites_grammar)
        print("✓ Grammar conversion successful")

        # Test that we can access attributes
        assert hasattr(rust_grammar, "get_unique_expressions")
        print("✓ Interface looks correct")

        return True
    except Exception as e:
        print(f"✗ Basic usage test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_enumeration():
    """Test actual enumeration."""
    try:
        from ultk_grammar import use_rust_if_available
        from examples.indefinites.grammar import indefinites_grammar
        from examples.indefinites.meaning import universe

        grammar = use_rust_if_available(indefinites_grammar)

        # Small enumeration to verify it works
        expressions = grammar.get_unique_expressions(
            3,  # Shallow depth for quick test
            max_size=100,
            unique_key=lambda expr: expr.evaluate(universe),
            compare_func=lambda e1, e2: len(e1) < len(e2),
        )

        print(f"✓ Enumeration successful: {len(expressions)} expressions generated")
        return True
    except Exception as e:
        print(f"✗ Enumeration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing ULTK Rust Grammar Extension")
    print("=" * 60)
    print()

    tests = [
        ("Import", test_import),
        ("Basic Usage", test_basic_usage),
        ("Enumeration", test_enumeration),
    ]

    results = []
    for name, test_func in tests:
        print(f"Testing {name}...")
        success = test_func()
        results.append((name, success))
        print()

    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {status:4s} - {name}")
        if not success:
            all_passed = False

    print()
    if all_passed:
        print("✓ All tests passed! The extension is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
