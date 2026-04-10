import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical

from ..core.factory import build_grammar
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..data.datasets import get_task_suite, get_task_by_name
from ..data.feynman_ground_truth import FEYNMAN_GROUND_TRUTH, classify_quality
from .trainer import Trainer


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

@dataclass
class MCTSNode:
    """
    One node in the MCTS tree = one partial expression state.

    tokens        : prefix tokens chosen so far
    pending_slots : open argument slots still needed (0 = complete)
    parent        : parent node (None for root)
    prior         : policy prior probability P(node | parent) from the policy
    children      : dict from token_id → child MCTSNode
    visit_count   : N(s) — how many times this node was visited
    value_sum     : W(s) — cumulative value from all visits
    """
    tokens:        List[str]
    pending_slots: int
    parent:        Optional["MCTSNode"] = field(default=None, repr=False)
    prior:         float                = 0.0
    children:      Dict[int, "MCTSNode"] = field(default_factory=dict)
    visit_count:   int   = 0
    value_sum:     float = 0.0

    @property
    def is_terminal(self) -> bool:
        return self.pending_slots == 0

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct: float) -> float:
        """
        PUCT score (Polynomial Upper Confidence Trees) as in AlphaZero:
            UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        where Q = mean value of child, P = prior, N(s) = parent visits.
        """
        if self.parent is None:
            return 0.0
        parent_visits = max(self.parent.visit_count, 1)
        exploration   = (c_puct * self.prior
                         * math.sqrt(parent_visits) / (1 + self.visit_count))
        return self.mean_value + exploration


# ---------------------------------------------------------------------------
# MCTS core
# ---------------------------------------------------------------------------

@torch.no_grad()
def _get_policy_priors(
    node:      MCTSNode,
    policy,
    grammar,
    max_length: int,
    device:    str,
) -> Dict[int, float]:
    """
    Query the policy network for action probabilities from this node.
    Returns {token_id: prior_probability} for all valid actions.
    """
    if len(node.tokens) == 0:
        token_ids = torch.empty(0, dtype=torch.long, device=device)
    else:
        token_ids = torch.tensor(
            [grammar.token_to_id[t] for t in node.tokens],
            dtype=torch.long, device=device,
        )

    # Build validity mask
    vocab_size = len(grammar)
    arities    = torch.tensor(
        [grammar.arity[grammar.id_to_token[i]] for i in range(vocab_size)],
        dtype=torch.long, device=device,
    )
    remaining   = max_length - len(node.tokens) - 1
    new_pending = node.pending_slots - 1 + arities
    valid       = (new_pending >= 0) & (new_pending <= remaining)
    action_mask = valid.float()

    if action_mask.sum() == 0:
        return {}

    logits, _ = policy(
        token_ids=token_ids,
        pending_slots=node.pending_slots,
        length=len(node.tokens),
        action_mask=action_mask,
    )

    probs = torch.softmax(logits, dim=-1)
    probs = probs * valid.float()
    total = probs.sum()
    if total > 0:
        probs = probs / total

    return {
        i: probs[i].item()
        for i in range(vocab_size)
        if valid[i].item() and probs[i].item() > 1e-8
    }


@torch.no_grad()
def _rollout(
    node:       MCTSNode,
    policy,
    grammar,
    X:          np.ndarray,
    y:          np.ndarray,
    evaluator:  PrefixEvaluator,
    max_length: int,
    device:     str,
) -> Tuple[float, List[str]]:
    """
    Complete the partial expression by sampling from the policy,
    then evaluate the resulting expression on (X, y).
    Returns (reward, complete_tokens).
    """
    tokens        = list(node.tokens)
    pending_slots = node.pending_slots

    while pending_slots > 0 and len(tokens) < max_length:
        if len(tokens) == 0:
            token_ids = torch.empty(0, dtype=torch.long, device=device)
        else:
            token_ids = torch.tensor(
                [grammar.token_to_id[t] for t in tokens],
                dtype=torch.long, device=device,
            )

        vocab_size = len(grammar)
        arities    = torch.tensor(
            [grammar.arity[grammar.id_to_token[i]] for i in range(vocab_size)],
            dtype=torch.long, device=device,
        )
        remaining   = max_length - len(tokens) - 1
        new_pending = pending_slots - 1 + arities
        valid       = (new_pending >= 0) & (new_pending <= remaining)
        action_mask = valid.float()

        if action_mask.sum() == 0:
            break

        logits, _ = policy(
            token_ids=token_ids,
            pending_slots=pending_slots,
            length=len(tokens),
            action_mask=action_mask,
        )

        dist   = Categorical(logits=logits)
        action = dist.sample().item()
        token  = grammar.id_to_token[action]

        tokens.append(token)
        pending_slots = pending_slots - 1 + grammar.arity[token]

    if pending_slots != 0:
        return -1.0, tokens

    result = evaluator.evaluate(tokens, X, y)
    if not result["is_valid"]:
        return -1.0, tokens

    nmse   = result["nmse"]
    reward = -nmse - 0.01 * len(tokens)
    return float(reward), tokens


def _select(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Traverse the tree using PUCT until a leaf node."""
    while not node.is_leaf and not node.is_terminal:
        best_score = -float("inf")
        best_child = None
        for child in node.children.values():
            score = child.ucb_score(c_puct)
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            break
        node = best_child
    return node


def _expand(node: MCTSNode, priors: Dict[int, float], grammar) -> None:
    """Add all valid children to the node using policy priors."""
    for token_id, prior in priors.items():
        token       = grammar.id_to_token[token_id]
        new_pending = node.pending_slots - 1 + grammar.arity[token]
        child       = MCTSNode(
            tokens=node.tokens + [token],
            pending_slots=new_pending,
            parent=node,
            prior=prior,
        )
        node.children[token_id] = child


def _backprop(node: MCTSNode, value: float) -> None:
    """Propagate the value estimate back up the tree."""
    while node is not None:
        node.visit_count += 1
        node.value_sum   += value
        node             = node.parent


# ---------------------------------------------------------------------------
# Main MCTS search function
# ---------------------------------------------------------------------------

def mcts_search(
    policy,
    grammar,
    X:              np.ndarray,
    y:              np.ndarray,
    evaluator:      PrefixEvaluator,
    num_simulations: int   = 200,
    c_puct:         float  = 1.4,
    max_length:     int    = 30,
    device:         str    = "cpu",
    seed_tokens:    Optional[List[str]] = None,
) -> Dict:
    """
    Run MCTS from the root (empty expression) and return the best expression.

    Parameters
    ----------
    policy          : trained SymbolicPolicy (eval mode)
    grammar         : Grammar object
    X, y            : dataset
    evaluator       : PrefixEvaluator
    num_simulations : number of MCTS iterations (more = better, slower)
    c_puct          : exploration constant (AlphaZero uses 1.0-2.0)
    max_length      : maximum expression length in tokens
    device          : 'cpu' or 'cuda'

    Returns
    -------
    dict with best_nmse, best_expr, best_tokens, all_visited (list of dicts)
    """
    policy.eval()

    root = MCTSNode(tokens=[], pending_slots=1)

    # Warm-start: if a good seed expression is provided (e.g. from sampling),
    # pre-populate the tree with the path to that expression so MCTS explores
    # around it rather than starting from scratch.
    if seed_tokens:
        node = root
        for token in seed_tokens:
            token_id    = grammar.token_to_id.get(token)
            if token_id is None:
                break
            new_pending = node.pending_slots - 1 + grammar.arity[token]
            if token_id not in node.children:
                child = MCTSNode(
                    tokens=node.tokens + [token],
                    pending_slots=new_pending,
                    parent=node,
                    prior=0.5,   # neutral prior for seed path
                )
                node.children[token_id] = child
            node = node.children[token_id]
            # Give the seed path some initial visits so UCB explores around it
            node.visit_count = 5
            node.value_sum   = -0.05 * 5   # approximate seed reward

    # Track all complete expressions evaluated during search
    all_visited: List[Dict] = []
    best_reward = -float("inf")
    best_tokens: List[str]  = []

    for sim_idx in range(num_simulations):
        # 1. SELECT
        node = _select(root, c_puct)

        # 2. EXPAND (if not terminal and not already expanded)
        if not node.is_terminal and node.is_leaf:
            priors = _get_policy_priors(node, policy, grammar, max_length, device)
            if priors:
                _expand(node, priors, grammar)
                # Pick the child with highest prior for rollout
                node = max(node.children.values(), key=lambda c: c.prior)

        # 3. EVALUATE — rollout to complete expression
        if node.is_terminal:
            result = evaluator.evaluate(node.tokens, X, y)
            if result["is_valid"]:
                nmse   = result["nmse"]
                reward = -nmse - 0.01 * len(node.tokens)
                expr   = safe_prefix_to_infix(
                    node.tokens, grammar,
                    result.get("optimized_constants", []),
                )
                all_visited.append({
                    "tokens": node.tokens,
                    "expr":   expr,
                    "nmse":   nmse,
                    "reward": reward,
                })
                if reward > best_reward:
                    best_reward = reward
                    best_tokens = node.tokens
            else:
                reward = -1.0
        else:
            reward, complete_tokens = _rollout(
                node, policy, grammar, X, y, evaluator, max_length, device
            )
            if reward > -1.0 and complete_tokens:
                result = evaluator.evaluate(complete_tokens, X, y)
                if result["is_valid"]:
                    nmse = result["nmse"]
                    expr = safe_prefix_to_infix(
                        complete_tokens, grammar,
                        result.get("optimized_constants", []),
                    )
                    all_visited.append({
                        "tokens": complete_tokens,
                        "expr":   expr,
                        "nmse":   nmse,
                        "reward": reward,
                    })
                    if reward > best_reward:
                        best_reward = reward
                        best_tokens = complete_tokens

        # 4. BACKPROP
        _backprop(node, reward)

    # Final evaluation of best expression found
    best_nmse, best_expr, best_quality = 1.0, "", "Poor"
    if best_tokens:
        result = evaluator.evaluate(best_tokens, X, y)
        if result["is_valid"]:
            best_nmse    = result["nmse"]
            best_expr    = safe_prefix_to_infix(
                best_tokens, grammar,
                result.get("optimized_constants", []),
            )
            best_quality = classify_quality(best_nmse)

    return {
        "best_nmse":    best_nmse,
        "best_expr":    best_expr,
        "best_tokens":  best_tokens,
        "best_quality": best_quality,
        "n_simulations": num_simulations,
        "n_visited":    len(all_visited),
        "root":         root,
        "all_visited":  all_visited,
    }


# ---------------------------------------------------------------------------
# Standalone script
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "difficulty", "true_expression",
    "sampling_nmse", "sampling_expr",
    "mcts_nmse",     "mcts_expr",
    "improvement",   "winner",
    "num_simulations", "n_visited",
]


def main():
    parser = argparse.ArgumentParser(
        description="MCTS guided by trained RSPG policy — AlphaZero-style SR."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["pmlb_feynman_subset", "pmlb_feynman_all"])
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")

    parser.add_argument("--sampling_csv", type=str, default=None,
                        help="CSV from evaluate_expressions.py (sampling baseline)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_simulations", type=int, default=200,
                        help="MCTS iterations per task (more = better, slower)")
    parser.add_argument("--c_puct",         type=float, default=1.4,
                        help="Exploration constant (AlphaZero default ≈ 1.0-2.0)")
    parser.add_argument("--max_length",     type=int,   default=30)
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--num_episodes",   type=int,   default=3000,
                        help="Episodes if no checkpoint found (train from scratch)")
    parser.add_argument("--seed",           type=int,   default=42)

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  simulations: {args.num_simulations}  |  c_puct: {args.c_puct}")

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples) for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    # Load sampling CSV if provided
    sampling_cache: Dict[str, Dict] = {}
    if args.sampling_csv:
        with open(args.sampling_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sampling_cache[row["task"]] = {
                    "nmse": float(row.get("nmse", 1.0)),
                    "expr": row.get("recovered_expression", ""),
                }
        print(f"Loaded {len(sampling_cache)} sampling results from {args.sampling_csv}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, f"mcts_{timestamp}.csv")
    rows        = []

    for idx, task in enumerate(tasks):
        X, y = task.generate()
        gt   = FEYNMAN_GROUND_TRUTH.get(task.name, {})
        print(f"\n[{idx+1}/{len(tasks)}] {task.name}  ({gt.get('difficulty','?')})")

        grammar   = build_grammar(num_variables=task.num_variables)
        evaluator = PrefixEvaluator(grammar)

        # Load checkpoint or train from scratch
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{task.name}.pt")
        if os.path.exists(checkpoint_path):
            print(f"  Loading checkpoint: {checkpoint_path}")
            trainer = Trainer.load_checkpoint(checkpoint_path, device=device)
        else:
            print(f"  No checkpoint found — training from scratch "
                  f"(seed={args.seed}, episodes={args.num_episodes})")
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            trainer = Trainer(
                X=X, y=y,
                num_variables=task.num_variables,
                device=device,
                optimizer_name="rspg",
            )
            trainer.num_episodes = args.num_episodes
            trainer.train(
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_name=task.name,
            )

        policy = trainer.policy
        policy.eval()

        # Restore dataset embedding
        tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
        tensor_y = torch.tensor(y, dtype=torch.float32, device=device)
        policy.set_dataset_embedding(tensor_X, tensor_y)

        # Sampling baseline
        if task.name in sampling_cache:
            sampling_nmse = sampling_cache[task.name]["nmse"]
            sampling_expr = sampling_cache[task.name]["expr"]
        else:
            best_ep = trainer.best_episode
            if best_ep is not None:
                er = evaluator.evaluate(best_ep["tokens"], X, y)
                sampling_nmse = er.get("nmse", 1.0)
                sampling_expr = safe_prefix_to_infix(
                    best_ep["tokens"], grammar,
                    er.get("optimized_constants", []),
                )
            else:
                sampling_nmse, sampling_expr = 1.0, ""

        print(f"  Sampling: NMSE={sampling_nmse:.4f}  expr={sampling_expr[:55]}")

        # MCTS search
        # Use sampling best as warm-start seed for MCTS
        seed_tokens = None
        if sampling_nmse < 0.999 and task.name in sampling_cache:
            # Re-tokenize from the checkpoint best episode if available
            if trainer.best_episode is not None:
                seed_tokens = trainer.best_episode.get("tokens")

        # Skip MCTS if sampling already found perfect expression
        if sampling_nmse < 0.001:
            print(f"  Sampling already perfect (NMSE={sampling_nmse:.6f}) — skipping MCTS")
            result = {
                "best_nmse":     sampling_nmse,
                "best_expr":     sampling_expr,
                "best_quality":  "Perfect",
                "n_simulations": 0,
                "n_visited":     0,
            }
        else:
            if seed_tokens:
                print(f"  Warm-starting MCTS from sampling best: {seed_tokens[:6]}...")
            print(f"  Running MCTS ({args.num_simulations} simulations, c_puct={args.c_puct})...")
            result = mcts_search(
                policy=policy,
                grammar=grammar,
                X=X, y=y,
                evaluator=evaluator,
                num_simulations=args.num_simulations,
                c_puct=args.c_puct,
                max_length=args.max_length,
                device=device,
                seed_tokens=seed_tokens,
            )

        mcts_nmse = result["best_nmse"]
        mcts_expr = result.get("best_expr", sampling_expr)
        improvement = sampling_nmse - mcts_nmse
        winner      = "mcts" if mcts_nmse < sampling_nmse else "sampling"

        print(f"  MCTS:     NMSE={mcts_nmse:.4f}  expr={mcts_expr[:55]}")
        print(f"  Visited {result['n_visited']} unique expressions")
        delta_str = f"Δ+{improvement:.4f} (MCTS wins)" if improvement > 0 else f"Δ{improvement:.4f} (sampling wins)"
        print(f"  {delta_str}")

        row = {
            "task":             task.name,
            "difficulty":       gt.get("difficulty", "Unknown"),
            "true_expression":  gt.get("expr", "unknown"),
            "sampling_nmse":    sampling_nmse,
            "sampling_expr":    sampling_expr,
            "mcts_nmse":        mcts_nmse,
            "mcts_expr":        mcts_expr,
            "improvement":      round(improvement, 6),
            "winner":           winner,
            "num_simulations":  args.num_simulations,
            "n_visited":        result["n_visited"],
        }
        rows.append(row)

        mode = "w" if idx == 0 else "a"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if idx == 0:
                writer.writeheader()
            writer.writerow(row)

    # Summary
    print("\n" + "=" * 70)
    print(f"MCTS SUMMARY  (simulations={args.num_simulations}, c_puct={args.c_puct})")
    print("=" * 70)
    w = max(len(r["task"]) for r in rows) + 2
    print(f"{'Task':<{w}} {'Sampling':>10}  {'MCTS':>10}  {'Δ NMSE':>10}  Winner")
    print("-" * 70)
    for r in rows:
        delta  = r["improvement"]
        marker = "mcts ▲" if delta > 0 else "sampling"
        print(f"{r['task']:<{w}} {r['sampling_nmse']:>10.4f}  "
              f"{r['mcts_nmse']:>10.4f}  {delta:>+10.4f}  {marker}")

    mcts_wins     = sum(1 for r in rows if r["winner"] == "mcts")
    sampling_wins = sum(1 for r in rows if r["winner"] == "sampling")
    n             = len(rows)
    mean_imp      = np.mean([r["improvement"] for r in rows])
    print("=" * 70)
    print(f"  MCTS wins:     {mcts_wins}/{n} ({100*mcts_wins/n:.0f}%)")
    print(f"  Sampling wins: {sampling_wins}/{n} ({100*sampling_wins/n:.0f}%)")
    print(f"  Mean NMSE improvement: {mean_imp:+.4f}")
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()