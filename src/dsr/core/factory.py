from ..config import GRAMMAR_CONFIG, ENV_CONFIG
from .grammar import Grammar

def build_grammar(num_variables: int) -> Grammar:
    if num_variables < 1:
        raise ValueError("num_variables must be >= 1")
    if num_variables > GRAMMAR_CONFIG["max_num_variables"]:
        raise ValueError("num_variables exceeds configured maximum")

    return Grammar(
        binary_operators=GRAMMAR_CONFIG["binary_operators"],
        unary_operators=GRAMMAR_CONFIG["unary_operators"],
        constants=GRAMMAR_CONFIG["constants"],
        num_variables=num_variables,
        use_eos=ENV_CONFIG["use_eos"],
    )
