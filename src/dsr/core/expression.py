from dataclasses import dataclass
from typing import List, Tuple

from .grammar import Grammar


@dataclass
class ExprNode:
    token: str
    children: list


def expression_complexity(tokens: List[str]) -> int:
    return len(tokens)


def is_complete_prefix(tokens: List[str], grammar: Grammar) -> bool:
    pending_slots = 1

    for token in tokens:
        if token not in grammar.arity:
            return False

        pending_slots = pending_slots - 1 + grammar.arity[token]

        if pending_slots < 0:
            return False

    return pending_slots == 0


def _parse_prefix(tokens: List[str], grammar: Grammar, index: int = 0) -> Tuple[ExprNode, int]:
    if index >= len(tokens):
        raise ValueError("Unexpected end of prefix sequence.")

    token = tokens[index]

    if token not in grammar.arity:
        raise ValueError(f"Unknown token: {token}")

    arity = grammar.arity[token]

    if arity == 0:
        return ExprNode(token=token, children=[]), index + 1

    children = []
    next_index = index + 1

    for _ in range(arity):
        child, next_index = _parse_prefix(tokens, grammar, next_index)
        children.append(child)

    return ExprNode(token=token, children=children), next_index


def prefix_to_tree(tokens: List[str], grammar: Grammar) -> ExprNode:
    root, next_index = _parse_prefix(tokens, grammar, index=0)

    if next_index != len(tokens):
        raise ValueError("Trailing tokens after a complete expression.")

    return root


def _tree_to_infix(node: ExprNode, grammar: Grammar) -> str:
    arity = grammar.arity[node.token]

    if arity == 0:
        return node.token

    if arity == 1:
        child_str = _tree_to_infix(node.children[0], grammar)
        return f"{node.token}({child_str})"

    if arity == 2:
        left = _tree_to_infix(node.children[0], grammar)
        right = _tree_to_infix(node.children[1], grammar)
        return f"({left} {node.token} {right})"

    raise ValueError(f"Unsupported arity: {arity}")


def prefix_to_infix(tokens: List[str], grammar: Grammar) -> str:
    if len(tokens) == 0:
        return "<empty>"

    tree = prefix_to_tree(tokens, grammar)
    return _tree_to_infix(tree, grammar)


def safe_prefix_to_infix(tokens: List[str], grammar: Grammar) -> str:
    try:
        return prefix_to_infix(tokens, grammar)
    except Exception:
        return "INVALID_EXPRESSION"