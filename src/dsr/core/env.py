from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from .grammar import Grammar
from .evaluator import PrefixEvaluator
from ..config import ENV_CONFIG


@dataclass
class StepOutput:
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class SymbolicRegressionEnv:
    def __init__(self, X: np.ndarray, y: np.ndarray, grammar: Grammar):
        if X.ndim != 2:
            raise ValueError("X must be (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must be (n_samples,)")

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.grammar = grammar

        self.max_length = ENV_CONFIG["max_length"]
        self.complexity_penalty = ENV_CONFIG["complexity_penalty"]
        self.invalid_reward = ENV_CONFIG["invalid_reward"]

        self.evaluator = PrefixEvaluator(grammar)

        self.tokens: List[str] = []
        self.pending_slots: int = 1
        self.done: bool = False

    def reset(self) -> Dict[str, Any]:
        self.tokens = []
        self.pending_slots = 1
        self.done = False
        return self._get_observation()

    def step(self, action_id: int) -> StepOutput:
        if self.done:
            raise RuntimeError("Episode finished. Call reset().")

        token = self.grammar.id_to_token[action_id]
        arity = self.grammar.arity[token]

        # update expression
        self.tokens.append(token)
        self.pending_slots = self.pending_slots - 1 + arity

        # invalid expression
        if self.pending_slots < 0:
            self.done = True
            return StepOutput(
                self._get_observation(),
                self.invalid_reward,
                True,
                {"reason": "invalid_slots"},
            )

        # complete expression
        if self.pending_slots == 0:
            self.done = True
            reward = self._compute_reward()
            return StepOutput(
                self._get_observation(),
                reward,
                True,
                {"reason": "complete"},
            )

        # too long
        if len(self.tokens) >= self.max_length:
            self.done = True
            return StepOutput(
                self._get_observation(),
                self.invalid_reward,
                True,
                {"reason": "max_length"},
            )

        return StepOutput(
            self._get_observation(),
            0.0,
            False,
            {"reason": "ongoing"},
        )

    def _compute_reward(self) -> float:
        result = self.evaluator.evaluate(self.tokens, self.X, self.y)

        if not result["is_valid"]:
            return self.invalid_reward

        nmse = result["nmse"]
        complexity = len(self.tokens)

        return -nmse - self.complexity_penalty * complexity

    def _get_observation(self) -> Dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "pending_slots": self.pending_slots,
            "length": len(self.tokens),
            # "X": self.X,
            # "y": self.y,
        }
    
    def valid_action_mask(self) -> np.ndarray:
        mask = np.ones(len(self.grammar), dtype=np.float32)

        if self.done or self.pending_slots == 0:
            mask[:] = 0.0

        return mask