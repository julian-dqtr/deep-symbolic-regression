"""
dsr/tests/test_core.py
======================
Unit tests for the RSPG symbolic regression system.

Run with:
    python -m dsr.tests.test_core

Or run a specific test class:
    python -m dsr.tests.test_core TestGrammar
    python -m dsr.tests.test_core TestPrefixEvaluator

Coverage:
  - Grammar construction and token properties
  - PrefixEvaluator: basic ops, constants, const ordering, invalid exprs
  - SymbolicRegressionEnv: step logic, valid_action_mask, reward computation
  - TopKMemory / DiverseTopKMemory: insertion, capacity, diversity
  - collect_batched_episodes: rewards contract (empty until filled by Trainer)
  - MDL reward: monotonicity, invalid guard
  - Expression utilities: is_complete_prefix, prefix_to_infix, safe_prefix_to_infix
"""

import math
import unittest

import numpy as np
import torch

from ..core.grammar import Grammar
from ..core.factory import build_grammar
from ..core.evaluator import PrefixEvaluator
from ..core.env import SymbolicRegressionEnv
from ..core.expression import (
    is_complete_prefix,
    prefix_to_infix,
    safe_prefix_to_infix,
)
from ..core.config import ENV_CONFIG
from ..analysis.memory import TopKMemory, DiverseTopKMemory
from ..core.mdl_reward import mdl_reward, mdl_reward_from_mse, MdlRewardConfig


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_grammar(num_variables: int = 1) -> Grammar:
    return build_grammar(num_variables=num_variables)


def _linear_dataset(n: int = 50):
    """y = 2*x0 + 1 on n samples in [-1, 1]."""
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (n, 1)).astype(np.float32)
    y = (2 * X[:, 0] + 1).astype(np.float32)
    return X, y


def _sin_dataset(n: int = 50):
    """y = sin(x0) on n uniform samples in [-pi, pi]."""
    X = np.linspace(-np.pi, np.pi, n).reshape(-1, 1).astype(np.float32)
    y = np.sin(X[:, 0]).astype(np.float32)
    return X, y


# ===========================================================================
# Grammar
# ===========================================================================

class TestGrammar(unittest.TestCase):

    def setUp(self):
        self.g1 = _make_grammar(1)
        self.g2 = _make_grammar(2)

    def test_vocab_contains_all_token_kinds(self):
        kinds = {self.g1.kind[t] for t in self.g1.action_space}
        self.assertIn("binary",   kinds)
        self.assertIn("unary",    kinds)
        self.assertIn("constant", kinds)
        self.assertIn("variable", kinds)

    def test_arity_binary(self):
        for op in ["+", "-", "*", "/"]:
            self.assertEqual(self.g1.arity[op], 2)

    def test_arity_unary(self):
        for op in ["sin", "cos", "exp", "log"]:
            self.assertEqual(self.g1.arity[op], 1)

    def test_arity_terminal(self):
        self.assertEqual(self.g1.arity["x0"], 0)
        self.assertEqual(self.g1.arity["1.0"], 0)

    def test_no_duplicate_tokens(self):
        action_space = self.g2.action_space
        self.assertEqual(len(action_space), len(set(action_space)))

    def test_id_roundtrip(self):
        for token, idx in self.g1.token_to_id.items():
            self.assertEqual(self.g1.id_to_token[idx], token)

    def test_num_variables_two(self):
        variables = [t for t in self.g2.action_space if self.g2.kind[t] == "variable"]
        self.assertEqual(len(variables), 2)
        self.assertIn("x0", variables)
        self.assertIn("x1", variables)

    def test_invalid_num_variables_raises(self):
        with self.assertRaises(ValueError):
            build_grammar(num_variables=0)


# ===========================================================================
# PrefixEvaluator
# ===========================================================================

class TestPrefixEvaluator(unittest.TestCase):

    def setUp(self):
        self.g = _make_grammar(1)
        self.ev = PrefixEvaluator(self.g)
        self.X, self.y = _linear_dataset()

    def test_single_variable(self):
        result = self.ev._eval_prefix(["x0"], self.X)
        np.testing.assert_allclose(result, self.X[:, 0], rtol=1e-5)

    def test_addition(self):
        """+ x0 1.0  →  x0 + 1"""
        result = self.ev._eval_prefix(["+", "x0", "1.0"], self.X)
        np.testing.assert_allclose(result, self.X[:, 0] + 1.0, rtol=1e-5)

    def test_multiplication(self):
        """* 2.0 x0  →  2*x0"""
        result = self.ev._eval_prefix(["*", "2.0", "x0"], self.X)
        np.testing.assert_allclose(result, 2.0 * self.X[:, 0], rtol=1e-5)

    def test_sin(self):
        X, y = _sin_dataset()
        result = self.ev._eval_prefix(["sin", "x0"], X)
        np.testing.assert_allclose(result, y, rtol=1e-5)

    def test_const_single(self):
        """const with C=[3.0] should return array of 3.0."""
        result = self.ev._eval_prefix(["const"], self.X, C=[3.0])
        np.testing.assert_allclose(result, np.full(len(self.X), 3.0), rtol=1e-5)

    def test_const_ordering(self):
        """
        / const const with C=[6.0, 2.0] must yield 6.0/2.0 = 3.0.
        The left child gets C[0]=6.0, the right child gets C[1]=2.0.
        This test fails if constants are consumed in the wrong order.
        """
        result = self.ev._eval_prefix(["/", "const", "const"], self.X, C=[6.0, 2.0])
        np.testing.assert_allclose(result, np.full(len(self.X), 3.0), rtol=1e-4)

    def test_protected_log_no_nan(self):
        """log(0) must not produce NaN — uses log(|x| + eps)."""
        X = np.zeros((10, 1), dtype=np.float32)
        result = self.ev._eval_prefix(["log", "x0"], X)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_protected_division_no_nan(self):
        """x / 0 must not produce NaN — uses x / (b + eps)."""
        X = np.zeros((10, 1), dtype=np.float32)
        result = self.ev._eval_prefix(["/", "x0", "x0"], X)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_evaluate_valid_near_perfect(self):
        """+ * 2.0 x0 1.0  ≈  2*x0 + 1  ≈  y  → low NMSE."""
        result = self.ev.evaluate(
            ["+", "*", "2.0", "x0", "1.0"], self.X, self.y,
            optimize_constants=False,
        )
        self.assertTrue(result["is_valid"])
        self.assertLess(result["nmse"], 0.01)

    def test_evaluate_incomplete_expression_invalid(self):
        """A single binary op with no children is not evaluable."""
        result = self.ev.evaluate(["+"], self.X, self.y, optimize_constants=False)
        self.assertFalse(result["is_valid"])

    def test_const_wrong_count_raises(self):
        """Supplying the wrong number of constants must raise ValueError."""
        with self.assertRaises(ValueError):
            self.ev._eval_prefix(["const"], self.X, C=[1.0, 2.0])


# ===========================================================================
# SymbolicRegressionEnv
# ===========================================================================

class TestEnv(unittest.TestCase):

    def setUp(self):
        self.g = _make_grammar(1)
        self.X, self.y = _linear_dataset()
        self.env = SymbolicRegressionEnv(self.X, self.y, self.g)

    def test_reset_clears_state(self):
        self.env.reset()
        self.assertEqual(self.env.tokens, [])
        self.assertEqual(self.env.pending_slots, 1)
        self.assertFalse(self.env.done)

    def test_single_terminal_completes_episode(self):
        """x0 alone is a complete expression — episode ends in one step."""
        self.env.reset()
        out = self.env.step(self.g.token_to_id["x0"])
        self.assertTrue(out.done)

    def test_complete_expression_reward_above_invalid(self):
        """A valid expression should get reward > invalid_reward (-1.0)."""
        self.env.reset()
        out = None
        for tok in ["+", "*", "2.0", "x0", "1.0"]:
            out = self.env.step(self.g.token_to_id[tok])
        self.assertTrue(out.done)
        self.assertGreater(out.reward, ENV_CONFIG["invalid_reward"])

    def test_action_mask_all_operators_masked_when_one_slot(self):
        """
        With pending_slots=1 AND max_length=1 (zero tokens left after this one),
        ALL operators are invalid — even unary ones — because after placing any
        operator at least one more token would be needed to complete the tree,
        which exceeds the budget. Only terminals (arity=0) are valid.
        """
        tight_env = SymbolicRegressionEnv(self.X, self.y, self.g)
        tight_env.max_length = 1   # only 1 token allowed total
        tight_env.reset()
        mask = tight_env.valid_action_mask()
        for i, token in enumerate(self.g.action_space):
            if self.g.arity[token] > 0:
                self.assertEqual(
                    mask[i], 0.0,
                    msg=f"Operator {token!r} (arity={self.g.arity[token]}) "
                        f"should be masked when max_length=1",
                )

    def test_action_mask_length_equals_vocab(self):
        self.env.reset()
        mask = self.env.valid_action_mask()
        self.assertEqual(len(mask), len(self.g))

    def test_step_after_done_raises(self):
        self.env.reset()
        self.env.step(self.g.token_to_id["x0"])   # completes
        with self.assertRaises(RuntimeError):
            self.env.step(self.g.token_to_id["x0"])

    def test_x_must_be_2d(self):
        with self.assertRaises(ValueError):
            SymbolicRegressionEnv(self.X[:, 0], self.y, self.g)

    def test_y_must_be_1d(self):
        with self.assertRaises(ValueError):
            SymbolicRegressionEnv(self.X, self.X, self.g)


# ===========================================================================
# Memory
# ===========================================================================

class TestTopKMemory(unittest.TestCase):

    def test_add_and_topk_order(self):
        mem = TopKMemory(capacity=3)
        mem.add(["x0"],           "x0",     reward=-0.1, nmse=0.1, complexity=1)
        mem.add(["x1"],           "x1",     reward=-0.5, nmse=0.5, complexity=1)
        mem.add(["+", "x0", "x0"], "x0+x0", reward=-0.3, nmse=0.3, complexity=3)
        top = mem.topk()
        self.assertAlmostEqual(top[0].reward, -0.1)   # best first

    def test_capacity_evicts_worst(self):
        mem = TopKMemory(capacity=2)
        mem.add(["x0"],           "x0",    reward=-0.5, nmse=0.5, complexity=1)
        mem.add(["x1"],           "x1",    reward=-0.3, nmse=0.3, complexity=1)
        mem.add(["+", "x0", "x1"], "x0+x1", reward=-0.1, nmse=0.1, complexity=3)
        self.assertEqual(len(mem), 2)
        rewards = [item.reward for item in mem.topk()]
        self.assertNotIn(-0.5, rewards)   # worst was evicted

    def test_duplicate_updates_to_better_reward(self):
        mem = TopKMemory(capacity=5)
        mem.add(["x0"], "x0", reward=-0.5, nmse=0.5, complexity=1)
        mem.add(["x0"], "x0", reward=-0.2, nmse=0.2, complexity=1)   # better
        self.assertEqual(len(mem), 1)
        self.assertAlmostEqual(mem.topk()[0].reward, -0.2)

    def test_duplicate_does_not_downgrade(self):
        mem = TopKMemory(capacity=5)
        mem.add(["x0"], "x0", reward=-0.2, nmse=0.2, complexity=1)
        mem.add(["x0"], "x0", reward=-0.9, nmse=0.9, complexity=1)   # worse
        self.assertAlmostEqual(mem.topk()[0].reward, -0.2)   # unchanged


class TestDiverseTopKMemory(unittest.TestCase):

    def test_near_duplicate_rejected_when_not_better(self):
        """Edit distance 1 — second expression rejected because reward is worse."""
        mem = DiverseTopKMemory(capacity=5, min_edit_distance=3)
        mem.add(["x0"], "x0", reward=-0.3, nmse=0.3, complexity=1)
        mem.add(["x1"], "x1", reward=-0.5, nmse=0.5, complexity=1)   # worse, dist=1
        self.assertEqual(len(mem), 1)

    def test_near_duplicate_replaces_when_better(self):
        """Edit distance 1 but better reward — replaces the neighbour."""
        mem = DiverseTopKMemory(capacity=5, min_edit_distance=3)
        mem.add(["x0"], "x0", reward=-0.5, nmse=0.5, complexity=1)
        mem.add(["x1"], "x1", reward=-0.1, nmse=0.1, complexity=1)   # better, dist=1
        self.assertEqual(len(mem), 1)
        self.assertAlmostEqual(mem.topk()[0].reward, -0.1)

    def test_diverse_expressions_coexist(self):
        """Edit distance >= min_edit_distance — both expressions are kept."""
        mem = DiverseTopKMemory(capacity=5, min_edit_distance=3)
        mem.add(["x0"], "x0", reward=-0.3, nmse=0.3, complexity=1)
        mem.add(["sin", "x0", "x0", "x0", "x0"], "sin(...)",
                reward=-0.2, nmse=0.2, complexity=5)
        self.assertEqual(len(mem), 2)


# ===========================================================================
# collect_batched_episodes reward contract
# ===========================================================================

class TestBatchedRolloutContract(unittest.TestCase):
    """
    collect_batched_episodes must return trajectories with empty rewards
    and final_reward=0.0. Trainer fills them after expression evaluation.
    """

    def setUp(self):
        from ..models.policy import SymbolicPolicy
        self.g  = _make_grammar(1)
        self.X, self.y = _linear_dataset()
        self.env = SymbolicRegressionEnv(self.X, self.y, self.g)
        self.policy = SymbolicPolicy(vocab_size=len(self.g))
        # collect_batched_episodes passes x=None, y=None to policy.forward —
        # the policy must have a cached_dataset_embedding or it crashes.
        tensor_X = torch.tensor(self.X, dtype=torch.float32)
        tensor_y = torch.tensor(self.y, dtype=torch.float32)
        self.policy.set_dataset_embedding(tensor_X, tensor_y)
        self.policy.eval()

    def test_rewards_empty_on_return(self):
        from ..training.rollout import collect_batched_episodes
        with torch.no_grad():
            trajs = collect_batched_episodes(
                env_template=self.env,
                policy=self.policy,
                grammar=self.g,
                batch_size=4,
                max_length=10,
                device="cpu",
            )
        for traj in trajs:
            self.assertEqual(
                traj["rewards"], [],
                msg=(
                    "collect_batched_episodes must return empty rewards; "
                    "Trainer is responsible for filling them after evaluation."
                ),
            )
            self.assertEqual(traj["final_reward"], 0.0)

    def test_tokens_and_log_probs_consistent(self):
        from ..training.rollout import collect_batched_episodes
        with torch.no_grad():
            trajs = collect_batched_episodes(
                env_template=self.env,
                policy=self.policy,
                grammar=self.g,
                batch_size=4,
                max_length=10,
                device="cpu",
            )
        for traj in trajs:
            self.assertGreater(len(traj["tokens"]), 0)
            self.assertEqual(len(traj["log_probs"]), len(traj["tokens"]))
            self.assertEqual(len(traj["entropies"]), len(traj["tokens"]))


# ===========================================================================
# MDL reward
# ===========================================================================

class TestMdlReward(unittest.TestCase):

    def test_perfect_fit_better_than_poor_fit(self):
        n      = 100
        y_true = np.ones(n, dtype=np.float32)
        y_perf = np.ones(n, dtype=np.float32)
        y_bad  = np.zeros(n, dtype=np.float32)
        cfg    = MdlRewardConfig()
        r_perf = mdl_reward(["x0"], y_perf, y_true, vocab_size=20, config=cfg)
        r_bad  = mdl_reward(["x0"], y_bad,  y_true, vocab_size=20, config=cfg)
        self.assertGreater(r_perf, r_bad)

    def test_shorter_expression_better_for_identical_fit(self):
        rng    = np.random.default_rng(1)
        y_true = rng.standard_normal(100).astype(np.float32)
        y_pred = y_true.copy()
        cfg    = MdlRewardConfig()
        r_short = mdl_reward(["x0"],      y_pred, y_true, vocab_size=20, config=cfg)
        r_long  = mdl_reward(["x0"] * 10, y_pred, y_true, vocab_size=20, config=cfg)
        self.assertGreater(r_short, r_long)

    def test_nan_prediction_returns_invalid_reward(self):
        cfg    = MdlRewardConfig(invalid_reward=-1.0)
        y_true = np.ones(10, dtype=np.float32)
        y_nan  = np.full(10, np.nan, dtype=np.float32)
        self.assertEqual(
            mdl_reward(["x0"], y_nan, y_true, vocab_size=20, config=cfg),
            -1.0,
        )

    def test_mdl_reward_and_from_mse_agree(self):
        rng    = np.random.default_rng(2)
        y_true = rng.standard_normal(100).astype(np.float32)
        y_pred = (y_true * 1.1).astype(np.float32)
        mse    = float(np.mean((y_true - y_pred) ** 2))
        tokens = ["*", "const", "x0"]
        cfg    = MdlRewardConfig()
        r1 = mdl_reward(tokens, y_pred, y_true, vocab_size=20, config=cfg)
        r2 = mdl_reward_from_mse(tokens, mse, n=100, vocab_size=20, config=cfg)
        self.assertAlmostEqual(r1, r2, places=2)


# ===========================================================================
# Expression utilities
# ===========================================================================

class TestExpressionUtils(unittest.TestCase):

    def setUp(self):
        self.g = _make_grammar(1)

    def test_is_complete_single_terminal(self):
        self.assertTrue(is_complete_prefix(["x0"], self.g))

    def test_is_complete_binary_with_two_terminals(self):
        self.assertTrue(is_complete_prefix(["+", "x0", "1.0"], self.g))

    def test_is_not_complete_missing_operand(self):
        self.assertFalse(is_complete_prefix(["+", "x0"], self.g))

    def test_is_not_complete_empty(self):
        self.assertFalse(is_complete_prefix([], self.g))

    def test_prefix_to_infix_contains_tokens(self):
        infix = prefix_to_infix(["+", "x0", "1.0"], self.g)
        self.assertIn("x0", infix)
        self.assertIn("+", infix)

    def test_safe_prefix_to_infix_empty(self):
        self.assertEqual(safe_prefix_to_infix([], self.g), "<empty>")

    def test_safe_prefix_to_infix_with_optimized_constants(self):
        """
        optimized_constants replaces 'const' placeholders in the infix string.
        We use '+ const x0' which is a complete valid prefix expression.
        """
        result = safe_prefix_to_infix(
            ["+", "const", "x0"], self.g, optimized_constants=[3.14]
        )
        # The constant 3.14 should appear formatted in the result
        self.assertNotEqual(result, "INVALID_EXPRESSION")
        self.assertIn("3.14", result)

    def test_safe_prefix_to_infix_invalid_returns_string(self):
        """An invalid sequence should not raise — returns INVALID_EXPRESSION."""
        result = safe_prefix_to_infix(["+"], self.g)
        self.assertIsInstance(result, str)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)