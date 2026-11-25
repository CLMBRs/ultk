# Minimal C-RASP Programs

This example enumerates programs of C-RASP, the RASP variant defined in [Yang & Chiang (2024), "Counting Like Transformers: Compiling Temporal Counting Logic Into Softmax Transformers"](https://openreview.net/forum?id=FmhPg4UJ9K), in order of size.  For every formal language (restricted to a finite set of strings) that is expressible by a C-RASP program, it finds a program of minimal size.

It is also an example of a grammar with several types (Boolean-valued operations, count-valued operations, and whole programs), and of using a custom `MeaningRepresentation` to speed up enumeration (see `ultk.language.grammar.enumeration`).

## Contents

- `grammar.py`: the C-RASP grammar, with one rule per operation in Definition 3.1 of the paper, plus a start rule `accept`, which accepts a string iff a Boolean-valued operation is true at its last position.  Non-terminal `"P"` is for Boolean-valued operations and `"C"` for count-valued operations.  Each rule's function is the only definition of its operation: it is written with the operations of `values.PositionValues`, so the same function computes values for one string or for a whole universe at once.
- `values.py`: the values of operations, and the operations on them (`PositionValues`), with `StringValues` for the values at each position of one string.
- `meaning.py`: the universe, whose referents (`StringReferent`s) are all non-empty strings up to a maximum length.
- `languages.py`: named formal languages (a\*b\*, Dyck-1, majority, ...) as predicates on strings, for looking up their minimal programs.
- `representations/`: faster ways of representing meanings during enumeration.  Because C-RASP operations only look at the current and earlier positions, the value of an operation at position `i` of a string `w` is its value at the last position of the prefix `w[:i]`, so these store one value per distinct prefix, instead of one per position of every string.
    - `prefix.py`: `PrefixRepresentation`, which computes values at every prefix with the grammar's own rule functions, and `PrefixTree`, the prefixes of a universe.
    - `basic.py`: `BasicCRASPRepresentation`, which stores values as plain numpy arrays.
    - `packed.py`: `PackedCRASPRepresentation`, which bit-packs Boolean values and stores counts in the smallest integer type that holds them.  It uses much less memory, which is what limits enumeration over large universes.
- `scripts/`:
    - `enumerate_programs.py`: enumerates programs by size, and reports the minimal programs for the named languages.
    - `benchmark.py`: records the time, memory and number of meanings found at each size, and can save every program found.
    - `find_named_languages.py`: finds the minimal programs for the named languages among saved programs, and checks them on longer strings.
- `outputs/benchmarks/`: results of the benchmarks.
- `BENCHMARKS.md`: how far enumeration goes, with universes of up to 2 million strings, and how the results were produced.

## Usage

From the `src/examples` directory:

```bash
python -m crasp.scripts.enumerate_programs --max_length 6 --max_cost 15
```

With strings up to length 6, this finds all 15,478 languages expressible by programs of size at most 15.  Some of the minimal programs found:

| language | size | minimal program |
|---|---|---|
| more a's than b's | 7 | `accept(not(<=(#(Q_a), #(Q_b))))` |
| a\*b\* | 9 | `accept(<=(#(<=(1, #(Q_b))), #(Q_b)))` |
| a\*b\*a\* | 15 | `accept(<=(#(Q_b), #(and(Q_b, <=(#(<=(1, #(Q_b))), #(Q_b))))))` |
| aⁿbⁿ | 15 | `accept(and(<=(#(Q_a), #(Q_b)), <=(#(<=(1, #(Q_b))), #(Q_a))))` |
| Dyck-1 (a = open, b = close) | 15 | `accept(<=(max(#(Q_a), #(Q_b)), #(and(Q_b, <=(#(Q_b), #(Q_a))))))` |

See `BENCHMARKS.md` for larger universes and more languages.

## Notes

- **Complexity is relative to the universe.**  Two programs are treated as equivalent if they agree on every string in the universe, so the minimal program for a language is minimal among programs that agree with it on strings up to `max_length`.  It need not define the same language on longer strings (although the examples above do).
- **Size counts every node of the program's tree**, including `accept`.  An operation that is used several times in a C-RASP program (e.g. `#(Q_b)` above) is counted every time it is used.
- **Count-valued operations dominate the search.**  The number of distinct count-valued operations grows by a factor of roughly 2.5 with each additional unit of size, much faster than the number of languages, so memory and time grow at about that rate.
