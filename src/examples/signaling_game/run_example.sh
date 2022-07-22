#!/bin/sh

python3 main.py \
--num_states 4 \
--num_signals 4 \
--num_rounds 1000 \
--distribution_over_states random \
--learning_rate 0.9 \
--save_languages outputs/example/languages.yml \
--save_weights outputs/example/weights.txt \
--save_distribution outputs/example/dist.png \
--save_accuracy_plot outputs/example/accuracy.png \
--save_complexity_plot outputs/example/complexity.png \
--save_tradeoff_plot outputs/example/tradeoff.png \
