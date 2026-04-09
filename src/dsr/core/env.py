from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np

from .grammar import Grammar
from .evaluator import PrefixEvaluator
from .config import ENV_CONFIG


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
        }
    
    def valid_action_mask(self) -> np.ndarray:
        """
        Intelligent action mask that filters grammatically invalid tokens.

        Three rules applied simultaneously:

        1. Done guard: if episode is finished, all actions are masked.

        2. Length constraint: a token is valid only if the expression can
           still be completed within max_length after choosing it.
           Choosing a binary op (+, *, ...) adds 1 pending slot; a unary op
           leaves pending_slots unchanged; a terminal closes 1 slot.
           After choosing token t:
               new_pending = pending_slots - 1 + arity(t)
           The minimum tokens needed to complete new_pending slots is
           new_pending (one terminal per slot). So token t is valid iff:
               len(tokens) + 1 + new_pending <= max_length
           i.e.  new_pending <= max_length - len(tokens) - 1

        3. Completability: if only 1 slot remains (pending_slots == 1),
           only terminals are allowed — any operator would open more slots
           that cannot all be closed before max_length is reached (rule 2
           already handles this, but making it explicit improves clarity).

        Result: the policy never wastes episodes on expressions that are
        guaranteed to be invalid or truncated.
        """
        mask = np.zeros(len(self.grammar), dtype=np.float32)

        if self.done or self.pending_slots == 0:
            return mask  # all zeros

        remaining = self.max_length - len(self.tokens) - 1  # slots after this token

        for i, token in enumerate(self.grammar.action_space):
            arity       = self.grammar.arity[token]
            new_pending = self.pending_slots - 1 + arity

            # Rule 1: new_pending must be non-negative
            if new_pending < 0:
                continue

            # Rule 2: must be completable within remaining length
            # Need at least new_pending more tokens (one terminal per slot)
            if new_pending > remaining:
                continue

            mask[i] = 1.0

        # Safety: if nothing is valid (edge case), allow all terminals
        if mask.sum() == 0:
            for i, token in enumerate(self.grammar.action_space):
                if self.grammar.arity[token] == 0:
                    mask[i] = 1.0

        return mask