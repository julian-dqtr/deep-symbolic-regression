from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TokenSpec:
    name: str
    arity: int
    kind: str  # "binary", "unary", "constant", "variable", "special"


class Grammar:
    def __init__(
        self,
        binary_operators: List[str],
        unary_operators: List[str],
        constants: Dict[str, float],
        num_variables: int,
        use_eos: bool = False,
    ):
        self.tokens: List[TokenSpec] = []
        self.constant_values = dict(constants)

        for op in binary_operators:
            self.tokens.append(TokenSpec(name=op, arity=2, kind="binary"))

        for op in unary_operators:
            self.tokens.append(TokenSpec(name=op, arity=1, kind="unary"))

        for const_name in constants.keys():
            self.tokens.append(TokenSpec(name=const_name, arity=0, kind="constant"))

        for i in range(num_variables):
            self.tokens.append(TokenSpec(name=f"x{i}", arity=0, kind="variable"))

        if use_eos:
            self.tokens.append(TokenSpec(name="<EOS>", arity=0, kind="special"))

        self.action_space = [tok.name for tok in self.tokens]

        if len(self.action_space) != len(set(self.action_space)):
            raise ValueError("Duplicate tokens in grammar")

        self.arity = {tok.name: tok.arity for tok in self.tokens}
        self.kind = {tok.name: tok.kind for tok in self.tokens}
        self.token_to_id = {tok.name: i for i, tok in enumerate(self.tokens)}
        self.id_to_token = {i: tok.name for i, tok in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.action_space)

    def is_terminal(self, token: str) -> bool:
        return self.arity[token] == 0 and self.kind[token] in {"constant", "variable"}

    def is_unary(self, token: str) -> bool:
        return self.arity[token] == 1

    def is_binary(self, token: str) -> bool:
        return self.arity[token] == 2

    def is_special(self, token: str) -> bool:
        return self.kind[token] == "special"