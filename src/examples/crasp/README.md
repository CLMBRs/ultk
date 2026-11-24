# Minimal C-RASP Programs

This example enumerates programs of C-RASP, the RASP variant defined in [Yang & Chiang (2024), "Counting Like Transformers: Compiling Temporal Counting Logic Into Softmax Transformers"](https://openreview.net/forum?id=FmhPg4UJ9K), in order of size.  For every formal language (restricted to a finite set of strings) that is expressible by a C-RASP program, it finds a program of minimal size.

It is also an example of a grammar with several types (Boolean-valued operations, count-valued operations, and whole programs), and of using a custom `Semantics` to speed up enumeration (see `ultk.language.grammar.enumeration`).

## Contents

- `grammar.py`: the C-RASP grammar, with one rule per operation in Definition 3.1 of the paper, plus a start rule `accept`, which accepts a string iff a Boolean-valued operation is true at its last position.  Non-terminal `"P"` is for Boolean-valued operations and `"C"` for count-valued operations.  The value of an operation at a string is a tuple, with one entry per position.
- `meaning.py`: the universe, whose referents are all non-empty strings up to a maximum length.
- `semantics.py`: `CRASPSemantics`, which computes the values of operations with numpy.  Because C-RASP operations only look at the current and earlier positions, the value of an operation at position `i` of a string `w` is its value at the last position of the prefix `w[:i]`, so it stores one value per distinct prefix, instead of one per position of every string.
- `scripts/enumerate_programs.py`: enumerates programs by size, and reports the minimal programs for some example languages from the paper.

## Usage

From the `src/examples` directory:

```bash
python -m crasp.scripts.enumerate_programs --max_length 6 --max_cost 15
```

With strings up to length 6, this finds all 15,478 languages expressible by programs of size at most 15 in about 20 seconds.  Some of the minimal programs found:

| language | size | minimal program |
|---|---|---|
| more a's than b's | 7 | `accept(not(<=(#(Q_a), #(Q_b))))` |
| a\*b\* | 9 | `accept(<=(#(<=(1, #(Q_b))), #(Q_b)))` |
| a\*b\*a\* | 15 | `accept(<=(#(Q_b), #(and(Q_b, <=(#(<=(1, #(Q_b))), #(Q_b))))))` |
| aⁿbⁿ | 15 | `accept(and(<=(#(Q_a), #(Q_b)), <=(#(<=(1, #(Q_b))), #(Q_a))))` |
| Dyck-1 (a = open, b = close) | 15 | `accept(<=(max(#(Q_a), #(Q_b)), #(and(Q_b, <=(#(Q_b), #(Q_a))))))` |

## Notes

- **Complexity is relative to the universe.**  Two programs are treated as equivalent if they agree on every string in the universe, so the minimal program for a language is minimal among programs that agree with it on strings up to `max_length`.  It need not define the same language on longer strings (although the examples above do).
- **Size counts every node of the program's tree**, including `accept`.  An operation that is used several times in a C-RASP program (e.g. `#(Q_b)` above) is counted every time it is used.
- **Count-valued operations dominate the search.**  The number of distinct count-valued operations grows by a factor of roughly 2.5 with each additional unit of size, much faster than the number of languages, so memory and time grow at about that rate.
