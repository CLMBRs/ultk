# Benchmarks: how far can C-RASP enumeration go?

These benchmarks measure enumerating C-RASP programs in order of size, over universes of *all* binary strings (alphabet {a, b}) up to some maximum length, with `ultk.language.grammar.enumeration.SemanticEnumerator`.

**In short:** directly over all 2,097,150 strings up to length 20, enumeration reaches programs of size 10 in about 2.5 minutes, and then runs out of memory.  But up to the sizes we can reach, longer strings barely distinguish more programs: enumerating over strings up to length 12 reaches size 16 in under 4 minutes, and every named language's minimal program found that way is also correct, and minimal, on all strings up to length 20.

## Contents

- [Setup](#setup)
- [Results](#results)
    - [Strings up to length 20](#strings-up-to-length-20)
    - [Smaller universes](#smaller-universes)
    - [Minimal programs for named languages](#minimal-programs-for-named-languages)
    - [Which programs short strings cannot tell apart](#which-programs-short-strings-cannot-tell-apart)
- [Main findings](#main-findings)
- [Reproducing these results](#reproducing-these-results)

## Setup

- **Grammar**: `grammar.py` (Definition 3.1 of Yang & Chiang 2024, plus `accept`), with every rule costing 1, so a program's size is its number of nodes.
- **Universes**: all binary strings of length 1 to *L*, for *L* = 6, 8, 10, 12 and 20 (126 to 2,097,150 strings).
- **Representations** (`representations/`):
    - `basic`: `BasicCRASPRepresentation`, one numpy value per distinct prefix (`bool` or `int64`).
    - `packed`: `PackedCRASPRepresentation`, the same values with Booleans bit-packed and counts in the smallest integer type that holds them.  It gives exactly the same results as `basic` (checked in `src/tests/test_crasp_example.py`).
- **Machine**: a Linux cloud container with 4 vCPUs (Intel Xeon, 2.3 GHz) and 15 GB of memory, no swap; Python 3.13.18, numpy 2.3.5.  Enumeration is single-threaded.
- **Limits**: each run was limited to 13.5 GB of virtual memory (`ulimit -v`), so running out of memory ends it cleanly.
- **Measurements** (`scripts/benchmark.py`): for each size, the number of distinct meanings of each type found so far, the time spent enumerating (excluding setup, such as building the universe), and the peak memory of the process.  "Languages" counts distinct meanings of whole programs, i.e. distinct sets of accepted strings.
- **Date**: 25 November 2026, on branch `claude/compassionate-knuth-ajfs9e`.

## Results

### Strings up to length 20

All 2,097,150 binary strings of length 1 to 20.  Building the universe takes about 25 seconds and 1.5 GB, not included below.

| Size ≤ | Languages | Count-valued sub-programs | Time, packed | Peak memory, packed | Time, basic | Peak memory, basic |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 9 | 42 | 0.7 s | 1.5 GB | 2.7 s | 2.2 GB |
| 6 | 16 | 92 | 2.1 s | 1.6 GB | 7.5 s | 3.0 GB |
| 7 | 29 | 239 | 6.9 s | 1.9 GB | 20 s | 5.4 GB |
| 8 | 71 | 624 | 19 s | 2.7 GB | 67 s | 11.7 GB |
| 9 | 147 | 1,526 | 56 s | 4.5 GB | out of memory | |
| 10 | 300 | 3,908 | 140 s | 9.3 GB | | |
| 11 | | | out of memory | | | |

Size 11 would need about 10,000 count-valued sub-programs of about 2 MB each, or about 21 GB.

### Smaller universes

All with the packed representation.  The number of languages found at each size:

| Size ≤ | *L* = 6 | *L* = 8 | *L* = 10 | *L* = 12 | *L* = 20 |
|---:|---:|---:|---:|---:|---:|
| 8 | 71 | 71 | 71 | 71 | 71 |
| 9 | 147 | 147 | 147 | 147 | 147 |
| 10 | 300 | 300 | 300 | 300 | 300 |
| 11 | 635 | 635 | 635 | 635 | |
| 12 | 1,475 | 1,516 | 1,516 | 1,516 | |
| 13 | 3,224 | 3,378 | 3,388 | 3,388 | |
| 14 | 7,009 | 7,628 | 7,701 | 7,701 | |
| 15 | | | | 18,256 | |
| 16 | | | | 43,731 | |

Time and peak memory to reach each size:

| Size ≤ | *L* = 6 | *L* = 8 | *L* = 10 | *L* = 12 | *L* = 20 |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.15 s, 0.1 GB | 0.17 s, 0.1 GB | 0.29 s, 0.1 GB | 0.46 s, 0.1 GB | 140 s, 9.3 GB |
| 12 | 1.4 s, 0.1 GB | 1.5 s, 0.1 GB | 1.8 s, 0.1 GB | 3.4 s, 0.3 GB | |
| 14 | 9.7 s, 0.2 GB | 10.8 s, 0.2 GB | 14.1 s, 0.5 GB | 30 s, 1.5 GB | |
| 15 | | | | 81 s, 3.9 GB | |
| 16 | | | | 219 s, 10.1 GB | |
| 17 | | | | out of memory | |

Runs for *L* ≤ 10 were stopped at size 14; *L* = 12 ran until it ran out of memory.

### Minimal programs for named languages

The languages in `languages.py`, found among the 43,731 programs enumerated over strings up to length 12 (`scripts/find_named_languages.py`), and then checked on all 2,097,150 strings up to length 20.  **Every program below is correct on all strings up to length 20, and therefore also minimal for them**: any program correct up to length 20 is also correct up to length 12, so none can be smaller.

| Language | Size | Minimal program |
|---|---:|---|
| ends in b | 2 | `accept(Q_b)` |
| contains a b | 5 | `accept(<=(1, #(Q_b)))` |
| at most one b | 5 | `accept(<=(#(Q_b), 1))` |
| length exactly 1 | 5 | `accept(<=(#(true), 1))` |
| only a's (a\*) | 6 | `accept(not(<=(1, #(Q_b))))` |
| length at least 2 | 6 | `accept(not(<=(#(true), 1)))` |
| at least as many a's as b's | 6 | `accept(<=(#(Q_b), #(Q_a)))` |
| more a's than b's | 7 | `accept(not(<=(#(Q_a), #(Q_b))))` |
| starts with a | 9 | `accept(<=(1, #(not(<=(1, #(Q_b))))))` |
| at least twice as many a's as b's | 9 | `accept(<=(#(Q_b), -(#(Q_a), #(Q_b))))` |
| a\*b\* | 9 | `accept(<=(#(<=(1, #(Q_b))), #(Q_b)))` |
| b\*a\* | 9 | `accept(<=(#(<=(1, #(Q_a))), #(Q_a)))` |
| exactly one b | 10 | `accept(and(<=(1, #(Q_b)), <=(#(Q_b), 1)))` |
| some a before some b | 10 | `accept(not(<=(#(<=(1, #(Q_a))), #(Q_a))))` |
| every prefix has at least as many a's as b's | 10 | `accept(<=(#(true), #(<=(#(Q_b), #(Q_a)))))` |
| equal numbers of a's and b's | 12 | `accept(and(<=(#(Q_a), #(Q_b)), <=(#(Q_b), #(Q_a))))` |
| last symbol is a strict majority | 14 | `accept(not(<=(max(#(Q_a), #(Q_b)), ?(Q_a, #(Q_b), #(Q_a)))))` |
| a\*b\*a\* | 15 | `accept(<=(#(Q_b), #(and(Q_b, <=(#(<=(1, #(Q_b))), #(Q_b))))))` |
| aⁿbⁿ | 15 | `accept(and(<=(#(Q_a), #(Q_b)), <=(#(<=(1, #(Q_b))), #(Q_a))))` |
| Dyck-1 (a = open, b = close) | 15 | `accept(<=(max(#(Q_a), #(Q_b)), #(and(Q_b, <=(#(Q_b), #(Q_a))))))` |
| first symbol equals last symbol | 16 | `accept(<=(#(true), ?(Q_a, #(<=(1, #(Q_a))), #(<=(1, #(Q_b))))))` |
| even number of a's (parity) | — | no program of size ≤ 16 |

Some of these use C-RASP in neat ways:
- **starts with a**: there is at least one position where no b has appeared yet.
- **a\*b\***: the number of positions from the first b on is at most the number of b's, i.e. every position from the first b on is a b.
- **every prefix has at least as many a's as b's**: the number of positions whose prefix has at least as many a's as b's equals the length.
- **first symbol equals last symbol**: the conditional `?` chooses, depending on the last symbol, which count to compare with the length.

### Which programs short strings cannot tell apart

Enumeration treats two programs as equivalent if they accept the same strings in the universe, so a smaller universe can merge programs that a larger one distinguishes.  The 43,731 languages found over strings up to length 12 are only 23,344 on strings up to length 5.  The smallest examples all involve the constant 2, `+(1, 1)`:

| Program | Size | Same as, up to length 5 | Size | First differ on |
|---|---:|---|---:|---|
| `accept(true)` | 2 | `accept(<=(min(#(Q_a), #(Q_b)), +(1, 1)))` | 10 | `aaabbb` |
| `accept(<=(#(Q_a), #(Q_b)))` | 6 | `accept(<=(#(Q_a), min(#(Q_b), +(1, 1))))` | 10 | `aaabbb` |
| `accept(<=(#(Q_a), +(1, 1)))` | 7 | `accept(<=(#(Q_a), max(#(Q_b), +(1, 1))))` | 10 | `aaabbb` |

For example, `min(#a, #b) <= 2` holds of every string up to length 5, but not of `aaabbb`.

## Main findings

1. **Memory, not time, is the limit.**  Almost all memory holds the values of count-valued sub-programs (about 90% of all meanings found), at about one byte per prefix even when packed, and their number grows by a factor of about 2.6 per unit of size.  Enumeration is single-threaded, so more cores would not help; on this 15 GB machine, the limit is size 10 over strings up to length 20, and size 16 over strings up to length 12.
2. **The packed representation is 3 to 4 times faster than the basic one, and uses much less memory**, over strings up to length 20 (e.g. 19 s and 2.7 GB versus 67 s and 11.7 GB to reach size 8, of which 1.5 GB is the universe itself).  It reaches size 10, where the basic one runs out of memory at size 9.
3. **For small programs, long strings add nothing.**  Over strings up to length 6 and up to length 20, enumeration finds exactly the same meanings, of every type, at every size up to 10.  A larger universe can only split meanings apart, never merge them, so equal counts mean the two universes group programs identically.  Up to size 14, strings up to length 10 and up to length 12 give the same languages.
4. **The length needed grows slowly with program size**, roughly with the largest constant a program of that size can build: short strings cannot distinguish comparisons with constants that are larger than their counts can be (see the table above).
5. **Recommended workflow for large universes**: enumerate over a moderate universe (strings up to length 10 or 12), and then check the programs of interest on the large one, as `find_named_languages --verify_length` does.  A program that passes the check is minimal for the large universe too.  This goes about 6 sizes further than enumerating over the large universe directly.
6. **Growth per size**: each extra unit of size multiplies the number of meanings, candidates and time by about 2.5 to 2.7.  Reaching size 17 over strings up to length 12 would need about 25 GB, and size 18 about 65 GB.

## Reproducing these results

From the `src/examples` directory, with the project's environment (e.g. after `uv sync`):

```bash
uv run bash crasp/scripts/run_benchmarks.sh
```

This runs every benchmark above (about 15 minutes on the machine described), writing the results to `outputs/benchmarks/`:

- `length_<L>_<representation>.jsonl`: one line per size, from `scripts/benchmark.py`.  The first line describes the run.  A run that ran out of memory ends without a line for the size it was working on.
- `named_languages.json`: the minimal programs for the named languages, and the merged programs, from `scripts/find_named_languages.py`.
- `programs_length_12.jsonl`: every program found over strings up to length 12 (about 7 MB, so not committed).

Individual runs, e.g.:

```bash
python -m crasp.scripts.benchmark --max_length 12 --max_cost 14 --programs_output programs.jsonl
python -m crasp.scripts.find_named_languages --programs programs.jsonl --verify_length 20
```
