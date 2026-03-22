import numpy as np

from ..config import ENV_CONFIG
from .grammar import Grammar


class PrefixEvaluator:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.eps = ENV_CONFIG["numeric_epsilon"]

    def evaluate(self, tokens, X, y):
        """
        Evaluate a prefix expression on dataset X and compare with target y.

        Args:
            tokens (list[str]): Prefix expression tokens.
            X (np.ndarray): Shape (n_samples, n_features)
            y (np.ndarray): Shape (n_samples,)

        Returns:
            dict with keys:
                - is_valid (bool)
                - nmse (float)
                - complexity (int)
        """
        complexity = len(tokens)

        try:
            y_pred = self._eval_prefix(tokens, X)

            if not isinstance(y_pred, np.ndarray):
                return {
                    "is_valid": False,
                    "nmse": 1.0,
                    "complexity": complexity,
                }

            if y_pred.shape != y.shape:
                return {
                    "is_valid": False,
                    "nmse": 1.0,
                    "complexity": complexity,
                }

            if not np.all(np.isfinite(y_pred)):
                return {
                    "is_valid": False,
                    "nmse": 1.0,
                    "complexity": complexity,
                }

            nmse = self._calculate_nmse(y, y_pred)

            return {
                "is_valid": True,
                "nmse": float(min(nmse, 1.0)),
                "complexity": complexity,
            }

        except Exception:
            return {
                "is_valid": False,
                "nmse": 1.0,
                "complexity": complexity,
            }

    def _calculate_nmse(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        var_true = np.var(y_true)

        if var_true <= self.eps:
            return float(mse)

        return float(mse / var_true)

    def _eval_prefix(self, tokens, X):
        stack = []

        for token in reversed(tokens):
            arity = self.grammar.arity[token]

            if arity == 0:
                stack.append(self._terminal_value(token, X))

            elif arity == 1:
                if len(stack) < 1:
                    raise ValueError("Invalid unary expression")
                a = stack.pop()
                stack.append(self._apply_unary(token, a))

            elif arity == 2:
                if len(stack) < 2:
                    raise ValueError("Invalid binary expression")
                a = stack.pop()
                b = stack.pop()
                stack.append(self._apply_binary(token, a, b))

            else:
                raise ValueError(f"Unsupported arity for token: {token}")

        if len(stack) != 1:
            raise ValueError("Expression did not reduce to a single output")

        return stack[0]

    def _terminal_value(self, token, X):
        if self.grammar.kind[token] == "variable":
            idx = int(token[1:])
            return X[:, idx]

        if self.grammar.kind[token] == "constant":
            value = self.grammar.constant_values[token]
            return np.full(X.shape[0], value, dtype=np.float32)

        raise ValueError(f"Unknown terminal token: {token}")

    def _apply_unary(self, token, a):
        if token == "sin":
            return np.sin(a)
        if token == "cos":
            return np.cos(a)
        if token == "exp":
            return np.exp(np.clip(a, -20, 20))
        if token == "log":
            return np.log(np.abs(a) + self.eps)

        raise ValueError(f"Unknown unary token: {token}")

    def _apply_binary(self, token, a, b):
        if token == "+":
            return a + b
        if token == "-":
            return a - b
        if token == "*":
            return a * b
        if token == "/":
            return a / (b + self.eps)

        raise ValueError(f"Unknown binary token: {token}")