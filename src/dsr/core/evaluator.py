import numpy as np

from ..config import ENV_CONFIG
from .grammar import Grammar


class PrefixEvaluator:
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.eps = ENV_CONFIG["numeric_epsilon"]

    def evaluate(self, tokens, X, y, optimize_constants=True):
        complexity = len(tokens)
        num_consts = tokens.count("const")

        if optimize_constants and num_consts > 0:
            def loss_fn(C):
                try:
                    y_pred = self._eval_prefix(tokens, X, C=C)
                    if not isinstance(y_pred, np.ndarray) or y_pred.shape != y.shape or not np.all(np.isfinite(y_pred)):
                        return 1e9
                    mse = np.mean((y - y_pred) ** 2)
                    var_y = np.var(y)
                    return float(mse / var_y) if var_y > self.eps else float(mse)
                except Exception:
                    return 1e9

            try:
                import scipy.optimize
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = scipy.optimize.minimize(loss_fn, x0=np.ones(num_consts), method="BFGS", options={"maxiter": 10})
                
                best_C = res.x
                y_pred = self._eval_prefix(tokens, X, C=best_C)
                if not isinstance(y_pred, np.ndarray) or y_pred.shape != y.shape or not np.all(np.isfinite(y_pred)):
                    return {"is_valid": False, "nmse": 1.0, "complexity": complexity, "optimized_constants": []}
                
                nmse = self._calculate_nmse(y, y_pred)
                return {
                    "is_valid": True,
                    "nmse": float(min(nmse, 1.0)),
                    "complexity": complexity,
                    "optimized_constants": best_C.tolist()
                }
            except Exception:
                return {"is_valid": False, "nmse": 1.0, "complexity": complexity, "optimized_constants": []}
        else:
            try:
                y_pred = self._eval_prefix(tokens, X)
                if not isinstance(y_pred, np.ndarray) or y_pred.shape != y.shape or not np.all(np.isfinite(y_pred)):
                    return {"is_valid": False, "nmse": 1.0, "complexity": complexity, "optimized_constants": []}
                
                nmse = self._calculate_nmse(y, y_pred)
                return {
                    "is_valid": True,
                    "nmse": float(min(nmse, 1.0)),
                    "complexity": complexity,
                    "optimized_constants": []
                }
            except Exception:
                return {"is_valid": False, "nmse": 1.0, "complexity": complexity, "optimized_constants": []}

    def _calculate_nmse(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        var_true = np.var(y_true)

        if var_true <= self.eps:
            return float(mse)

        return float(mse / var_true)

    def _eval_prefix(self, tokens, X, C=None):
        stack = []
        num_consts = tokens.count("const")
        c_idx = num_consts - 1

        for token in reversed(tokens):
            arity = self.grammar.arity[token]

            if arity == 0:
                if token == "const" and C is not None:
                    stack.append(np.full(X.shape[0], C[c_idx], dtype=np.float32))
                    c_idx -= 1
                else:
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