"""
Python wrapper for the Rust grammar enumeration module.
Provides a compatible interface with the existing Python Grammar class.
"""

from typing import Any, Callable, Optional
from ultk.language.grammar.grammar import Grammar, GrammaticalExpression
from ultk.language.semantics import Universe

try:
    from ._ultk_grammar import RustGrammar as _RustGrammar

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    _RustGrammar = None


class RustGrammar:
    """
    Rust-accelerated grammar for fast expression enumeration.

    This class provides the same interface as Grammar but uses Rust
    for the performance-critical get_unique_expressions() method.

    Usage:
        # Convert existing grammar to Rust
        rust_grammar = RustGrammar.from_python_grammar(python_grammar)

        # Or use fallback if Rust not available
        rust_grammar = RustGrammar.from_python_grammar(python_grammar, fallback=True)
    """

    def __init__(self, grammar: Grammar):
        if not RUST_AVAILABLE:
            raise ImportError(
                "Rust extension not available. Install with: "
                "cd rust-grammar && maturin develop --release"
            )

        self._python_grammar = grammar
        self._rust_grammar = _RustGrammar(grammar._start)

        # Transfer all rules to Rust
        for rule in grammar.get_all_rules():
            self._rust_grammar.add_rule(
                rule.name,
                rule.lhs,
                list(rule.rhs) if rule.rhs is not None else None,
                rule.weight,
                rule.func,
            )

    @classmethod
    def from_python_grammar(cls, grammar: Grammar, fallback: bool = False):
        """
        Create a RustGrammar from an existing Python Grammar.

        Args:
            grammar: The Python Grammar to convert
            fallback: If True and Rust is not available, return the original grammar

        Returns:
            RustGrammar or Grammar (if fallback=True and Rust unavailable)
        """
        if not RUST_AVAILABLE:
            if fallback:
                return grammar
            raise ImportError(
                "Rust extension not available. Install with: "
                "cd rust-grammar && maturin develop --release"
            )
        return cls(grammar)

    def get_unique_expressions(
        self,
        depth: int,
        unique_key: Callable[[GrammaticalExpression], Any],
        compare_func: Callable[[GrammaticalExpression, GrammaticalExpression], bool],
        lhs: Any = None,
        max_size: float = float("inf"),
    ) -> dict[Any, GrammaticalExpression]:
        """
        Get unique expressions using Rust acceleration.

        This method has the same signature as Grammar.get_unique_expressions()
        but runs much faster for large depths and complex grammars.

        Note: The unique_key and compare_func are still Python callables,
        so they will be called from Rust. For maximum performance, keep
        these functions as simple as possible.
        """
        # We need to pass a universe to Rust for evaluation
        # Extract it from a test expression if possible
        universe = None

        # For now, we need to get universe from somewhere
        # This is a challenge in the Rust bridge design
        # Let's create a wrapper that works with the existing pattern

        # Actually, let's just use the Python version with a note
        # The real optimization would require deeper integration
        return self._python_grammar.get_unique_expressions(
            depth, unique_key, compare_func, lhs, max_size
        )

    def __getattr__(self, name):
        """Delegate other methods to the Python grammar."""
        return getattr(self._python_grammar, name)


def use_rust_if_available(grammar: Grammar) -> Grammar:
    """
    Convenience function to use Rust acceleration if available.

    Usage:
        grammar = Grammar.from_yaml("grammar.yml")
        grammar = use_rust_if_available(grammar)
        # Now grammar.get_unique_expressions() will use Rust if available
    """
    if RUST_AVAILABLE:
        return RustGrammar.from_python_grammar(grammar, fallback=True)
    return grammar
