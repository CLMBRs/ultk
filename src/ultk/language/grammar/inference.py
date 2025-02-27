from typing import TypeVar, Iterable, Callable
from ultk.language.grammar.grammar import Grammar, GrammaticalExpression
from ultk.language.grammar.likelihood import Dataset, all_or_nothing
from math import isnan, isinf, exp
import copy
import random


def log_mh_sample(
    expr: GrammaticalExpression,
    grammar: Grammar,
    data: Dataset,
    likelihood_func: Callable[[Dataset, GrammaticalExpression], float] = all_or_nothing,
) -> GrammaticalExpression:
    """Sample a new GrammaticalExpression from an exsiting one and data using Metropolis Hastings using log probabilities

    Args:
        expr (GrammaticalExpression): the exsiting GrammaticalExpression
        grammar (Grammar): the grammar for generation
        data (Dataset): data used for calculation of acceptance probability
        likelihood_func (Callable[[Dataset, GrammaticalExpression], float], optional): _description_. Defaults to all_or_nothing.

    Returns:
        GrammaticalExpression: newly sampled GrammaticalExpression
    """
    old_tree_prior = grammar.log_prior(expr)
    old_node_count = expr.node_count()
    while True:
        old_tree = copy.deepcopy(expr)
        current_node, parent_node = mh_select(old_tree)
        old_subtree_prior = grammar.log_prior(current_node)
        new_tree, new_node = mh_generate(old_tree, current_node, parent_node, grammar)
        new_tree_prior = grammar.log_prior(new_tree)
        new_node_count = new_tree.node_count()
        new_subtree_prior = grammar.log_prior(new_node)
        mh_accept = (
            (new_tree_prior + likelihood_func(data, new_tree))
            - (old_tree_prior + likelihood_func(data, old_tree))
        ) + (
            (old_subtree_prior - new_node_count) - (new_subtree_prior - old_node_count)
        )
        if not (isnan(mh_accept) or isinf(mh_accept)) and (
            mh_accept >= 0 or random.random() < exp(mh_accept)
        ):
            return new_tree


def mh_sample(
    expr: GrammaticalExpression,
    grammar: Grammar,
    data: Dataset,
    likelihood_func: Callable[[Dataset, GrammaticalExpression], float] = all_or_nothing,
) -> GrammaticalExpression:
    """Sample a new GrammaticalExpression from an exsiting one and data using Metropolis Hastings

    Args:
        expr (GrammaticalExpression): the exsiting GrammaticalExpression
        grammar (Grammar): the grammar for generation
        data (Dataset): data used for calculation of acceptance probability
        likelihood_func (Callable[[Dataset, GrammaticalExpression], float], optional): _description_. Defaults to all_or_nothing.

    Returns:
        GrammaticalExpression: newly sampled GrammaticalExpression
    """
    old_tree_prior = grammar.prior(expr)
    old_node_count = expr.node_count()
    while True:
        old_tree = copy.deepcopy(expr)
        current_node, parent_node = mh_select(old_tree)
        old_subtree_prior = grammar.prior(current_node)
        new_tree, new_node = mh_generate(old_tree, current_node, parent_node, grammar)
        new_tree_prior = grammar.prior(new_tree)
        new_node_count = new_tree.node_count()
        new_subtree_prior = grammar.prior(new_node)
        try:
            mh_accept = min(
                1,
                (
                    (new_tree_prior * likelihood_func(data, new_tree))
                    / (old_tree_prior * likelihood_func(data, old_tree))
                )
                * (
                    (old_subtree_prior / new_node_count)
                    / (new_subtree_prior / old_node_count)
                ),
            )
        except ZeroDivisionError:
            mh_accept = 0
        if random.random() < mh_accept:
            return new_tree


def mh_select(
    old_tree: GrammaticalExpression,
) -> tuple[GrammaticalExpression, GrammaticalExpression]:
    """Select a node for futher change from a GrammaticalExpression

    Args:
        old_tree (GrammaticalExpression): input GrammaticalExpression

    Returns:
        tuple[GrammaticalExpression, GrammaticalExpression]: the node selected for change and its parent node
    """
    linearized_self = []
    parents = []
    stack = [(old_tree, -1)]
    while stack:
        current_node, parent_index = stack.pop()
        linearized_self.append(current_node)
        parents.append(parent_index)
        current_index = len(linearized_self) - 1
        children = current_node.children if current_node.children else []
        for child in children:
            stack.append((child, current_index))
    changing_node = random.choice(range(len(linearized_self)))
    current_node = linearized_self[changing_node]
    parent_node = linearized_self[parents[changing_node]]
    return (current_node, parent_node)


def mh_generate(
    old_tree: GrammaticalExpression,
    current_node: GrammaticalExpression,
    parent_node: GrammaticalExpression,
    grammar: Grammar,
) -> tuple[GrammaticalExpression, GrammaticalExpression]:
    """Generate a new GrammaticalExpression

    Args:
        old_tree (GrammaticalExpression): the original full GrammaticalExpression
        current_node (GrammaticalExpression): the node selected for change
        parent_node (GrammaticalExpression): the parent node for the chaging node
        grammar (Grammar): grammar used for generation

    Returns:
        tuple[GrammaticalExpression, GrammaticalExpression]: the new full GrammaticalExpression and the changed node
    """
    if current_node != old_tree:
        new_children = []
        children = parent_node.children if parent_node.children else ()
        for child in children:
            if child is current_node:
                new_node = grammar.generate(
                    grammar._rules_by_name[current_node.rule_name].lhs
                )
                new_children.append(new_node)
            else:
                new_children.append(child)
        parent_node.replace_children(tuple(new_children))
        new_tree = old_tree
    else:
        new_node = grammar.generate(grammar._rules_by_name[old_tree.rule_name].lhs)
        new_tree = new_node
    return (new_tree, new_node)
