import numpy as np

from .config import ENV_CONFIG
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
                    y_pred = self._eval_prefix(tokens, X, C=list(C))
                    if (
                        not isinstance(y_pred, np.ndarray)
                        or y_pred.shape != y.shape
                        or not np.all(np.isfinite(y_pred))
                    ):
                        return 1e9
                    mse   = np.mean((y - y_pred) ** 2)
                    var_y = np.var(y)
                    return float(mse / var_y) if var_y > self.eps else float(mse)
                except Exception:
                    return 1e9

            try:
                import scipy.optimize
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = scipy.optimize.minimize(
                        loss_fn,
                        x0=np.ones(num_consts),
                        method="BFGS",
                        options={"maxiter": 10},
                    )

                best_C = list(res.x)
                y_pred = self._eval_prefix(tokens, X, C=best_C)
                if (
                    not isinstance(y_pred, np.ndarray)
                    or y_pred.shape != y.shape
                    or not np.all(np.isfinite(y_pred))
                ):
                    return {
                        "is_valid": False,
                        "nmse": 1.0,
                        "complexity": complexity,
                        "optimized_constants": [],
                    }

                nmse = self._calculate_nmse(y, y_pred)
                return {
                    "is_valid": True,
                    "nmse": float(min(nmse, 1.0)),
                    "complexity": complexity,
                    "optimized_constants": best_C,
                }
            except Exception:
                return {
                    "is_valid": False,
                    "nmse": 1.0,
                    "complexity": complexity,
                    "optimized_constants": [],
                }
        else:
            try:
                y_pred = self._eval_prefix(tokens, X)
                if (
                    not isinstance(y_pred, np.ndarray)
                    or y_pred.shape != y.shape
                    or not np.all(np.isfinite(y_pred))
                ):
                    return {
                        "is_valid": False,
                        "nmse": 1.0,
                        "complexity": complexity,
                        "optimized_constants": [],
                    }

                nmse = self._calculate_nmse(y, y_pred)
                return {
                    "is_valid": True,
                    "nmse": float(min(nmse, 1.0)),
                    "complexity": complexity,
                    "optimized_constants": [],
                }
            except Exception:
                return {
                    "is_valid": False,
                    "nmse": 1.0,
                    "complexity": complexity,
                    "optimized_constants": [],
                }

    def _calculate_nmse(self, y_true, y_pred):
        mse      = np.mean((y_true - y_pred) ** 2)
        var_true = np.var(y_true)
        if var_true <= self.eps:
            return float(mse)
        return float(mse / var_true)

    def _eval_prefix(self, tokens, X, C=None):
        """
        Evaluate a prefix-notation expression on dataset X.

        Parameters
        ----------
        tokens : List[str]  — prefix token sequence
        X      : np.ndarray — shape (n_samples, n_features)
        C      : List[float] | None
            Optimised constant values, one per 'const' token in left-to-right
            order (i.e. the order they appear in the prefix sequence).
            Must have exactly tokens.count('const') elements when provided.

        Implementation note
        -------------------
        We use a standard stack-based prefix evaluator with a forward pass
        (left to right). Constants are consumed from C in left-to-right order
        via a simple index counter, which matches the order in which BFGS
        assigns them (loss_fn iterates tokens in order via this same function).
        The earlier implementation used a reversed pass with a decrementing
        counter, which produced the same result for symmetric expressions but
        could silently mis-assign constants in asymmetric cases.
        """
        if C is not None and len(C) != tokens.count("const"):
            raise ValueError(
                f"Expected {tokens.count('const')} constants, got {len(C)}."
            )

        stack  = []
        c_iter = iter(C) if C is not None else iter([])

        # Standard recursive-descent using an explicit stack.
        # We iterate left-to-right and use a 'pending' counter to know
        # how many operands each operator is still waiting for.
        # Simpler: use the recursive helper below which is easier to verify.
        return self._eval_recursive(tokens, X, c_iter, [0])[0]

    def _eval_recursive(self, tokens, X, c_iter, idx):
        """
        Recursive prefix evaluator.  Returns (value, next_index).
        idx is a one-element list so it acts as a mutable integer reference.
        """
        if idx[0] >= len(tokens):
            raise ValueError("Unexpected end of prefix sequence.")

        token = tokens[idx[0]]
        idx[0] += 1
        arity = self.grammar.arity[token]

        if arity == 0:
            return self._terminal_value(token, X, c_iter), idx[0]

        if arity == 1:
            a, idx[0] = self._eval_recursive(tokens, X, c_iter, idx)
            return self._apply_unary(token, a), idx[0]

        if arity == 2:
            a, idx[0] = self._eval_recursive(tokens, X, c_iter, idx)
            b, idx[0] = self._eval_recursive(tokens, X, c_iter, idx)
            return self._apply_binary(token, a, b), idx[0]

        raise ValueError(f"Unsupported arity {arity} for token: {token!r}")

    def _terminal_value(self, token, X, c_iter):
        if self.grammar.kind[token] == "variable":
            idx = int(token[1:])
            return X[:, idx].astype(np.float32)

        if token == "const":
            try:
                val = next(c_iter)
            except StopIteration:
                raise ValueError("Not enough constant values supplied.")
            return np.full(X.shape[0], val, dtype=np.float32)

        if self.grammar.kind[token] == "constant":
            value = self.grammar.constant_values[token]
            return np.full(X.shape[0], value, dtype=np.float32)

        raise ValueError(f"Unknown terminal token: {token!r}")

    def _apply_unary(self, token, a):
        if token == "sin":
            return np.sin(a)
        if token == "cos":
            return np.cos(a)
        if token == "exp":
            return np.exp(np.clip(a, -20, 20))
        if token == "log":
            return np.log(np.abs(a) + self.eps)
        if token == "sqrt":
            # Protected: sqrt(|a|) avoids domain errors for negative inputs.
            # The agent learns that negative inputs incur a penalty through
            # the NMSE reward — we do not hard-clip the domain here.
            return np.sqrt(np.abs(a))
        raise ValueError(f"Unknown unary token: {token!r}")

    def _apply_binary(self, token, a, b):
        if token == "+":
            return a + b
        if token == "-":
            return a - b
        if token == "*":
            return a * b
        if token == "/":
            return a / (b + self.eps)
        if token == "pow":
            # Protected: pow(|a|, clip(b)) avoids complex numbers and overflow.
            # |a| ensures a non-negative base; b is clipped to [-10, 10] to
            # prevent inf for large exponents (e.g. x^100 on x=2).
            return np.power(np.abs(a) + self.eps, np.clip(b, -10, 10))
        raise ValueError(f"Unknown binary token: {token!r}")