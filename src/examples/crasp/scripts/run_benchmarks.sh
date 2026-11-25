#!/bin/bash
# Reproduce the results in crasp/BENCHMARKS.md.  Run from the src/examples directory:
#     bash crasp/scripts/run_benchmarks.sh
# Each run is limited to 13.5 GB of virtual memory, so that running out of memory ends the run cleanly.
set -x
OUT=crasp/outputs/benchmarks
MEMORY_LIMIT_KB=13500000
mkdir -p $OUT

# all strings up to length 20, with both representations
for REPRESENTATION in basic packed; do
  ( ulimit -v $MEMORY_LIMIT_KB; python -m crasp.scripts.benchmark --max_length 20 --max_cost 12 \
      --representation $REPRESENTATION --output $OUT/length_20_$REPRESENTATION.jsonl )
done

# smaller universes
for LENGTH in 6 8 10; do
  ( ulimit -v $MEMORY_LIMIT_KB; python -m crasp.scripts.benchmark --max_length $LENGTH --max_cost 14 \
      --output $OUT/length_${LENGTH}_packed.jsonl )
done
( ulimit -v $MEMORY_LIMIT_KB; python -m crasp.scripts.benchmark --max_length 12 --max_cost 17 \
    --output $OUT/length_12_packed.jsonl --programs_output $OUT/programs_length_12.jsonl )

# minimal programs for named languages, checked on all strings up to length 20
( ulimit -v $MEMORY_LIMIT_KB; python -m crasp.scripts.find_named_languages --programs $OUT/programs_length_12.jsonl \
    --verify_length 20 --merge_length 5 --output $OUT/named_languages.json )
