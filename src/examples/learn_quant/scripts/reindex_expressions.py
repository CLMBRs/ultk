import ultk.language.grammar
from ultk.language.grammar import Grammar
from learn_quant.grammar import quantifiers_grammar
from learn_quant.quantifier import QuantifierUniverse
from learn_quant.util import read_expressions, calculate_term_expression_depth
from learn_quant.monotonicity import *
from importlib import reload
import numpy as np
from typing import Iterable
from hydra import compose, initialize
from omegaconf import OmegaConf
import csv

initialize(version_base=None, config_path="../conf", job_name="learn")
cfg = compose(config_name="config", overrides=["recipe=base"])

expressions, _ = read_expressions("../outputs/M4/X4/d5/generated_expressions_xidx.yml", add_indices=False)

#### Create a sample of expressions 
# If depth > 3, random
# Otherwise, include in sample

import random
expressions_sample = []
shuffled_expressions = random.sample(expressions, len(expressions))

for expression in shuffled_expressions:
    if calculate_term_expression_depth(expression.term_expression) <= 3:
        expressions_sample.append(expression)

for expression in shuffled_expressions:
    if calculate_term_expression_depth(expression.term_expression) > 3:
        expressions_sample.append(expression)
    if len(expressions_sample) >= 2000:
        break

# Build a dictionary from the original list of expressions
# mapping term_expression -> original index
expression_index_map = {
    expr.term_expression: i
    for i, expr in enumerate(expressions)
}

with open('expressions_sample_2k.csv', 'w', newline='') as csvfile:
    fieldnames = ['index', 'term_expression', 'original_index']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for idx, expression in enumerate(expressions_sample):
        original_idx = expression_index_map.get(expression.term_expression, None)
        writer.writerow({
            'index': idx,
            'term_expression': expression.term_expression,
            'original_index': original_idx
        })