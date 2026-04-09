"""
beam_search.py
==============
Beam search over a trained RSPG policy for final evaluation.

During training, the policy samples stochastically — it may miss the best
expression it has learned to assign high probability to. Beam search at
inference time fixes this by deterministically exploring the top-k branches
at each decoding step, keeping only the `beam_width` most promising partial
expressions at all times.

This is a standard technique in DSR (Petersen et al., 2021) and sequence
models generally. It runs AFTER training — the policy weights are frozen.

Key design choices
------------------
- Grammar-aware: the beam tracks `pending_slots` to enforce structural
  validity. Beams that would violate the grammar (pending_slots < 0) are
  pruned immediately.
- Score = sum of log-probs along the path (sequence likelihood under policy).
  Final ranking uses NMSE evaluated on data, not the policy score.
- Duplicate suppression: finished beams with identical token sequences are
  deduplicated before NMSE evaluation.
- No environment calls during search: the grammar constraints are tracked
  internally, so beam search is fast even for large beam widths.

Usage
-----
Standalone script:

    python -m dsr.training.beam_search \
        --suite pmlb_feynman_subset \
        --checkpoint path/to/policy.pt \
        --beam_width 50 \
        --max_length 30

Or from Python (e.g. in evaluate_expressions.py):

    from dsr.training.beam_search import beam_search_decode, load_policy

    policy = load_policy(checkpoint_path, grammar, device)
    results = beam_search_decode(
        policy=policy,
        grammar=grammar,
        X=X, y=y,
        evaluator=evaluator,
        beam_width=50,
        max_length=30,
        device=device,
    )
    best = results[0]   # sorted by nmse ascending
    print(best["expr"], best["nmse"])

Output (standalone)
------
results/beam_search_<timestamp>.csv
"""

import argparse
import csv
import heapq
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.factory import build_grammar
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix, is_complete_prefix
from ..models.policy import SymbolicPolicy
from ..data.datasets import get_task_suite, get_task_by_name
from ..data.feynman_ground_truth import FEYNMAN_GROUND_TRUTH, classify_quality


# ---------------------------------------------------------------------------
# Beam state
# ---------------------------------------------------------------------------

@dataclass(order=True)
class BeamState:
    """
    A partial expression under construction.
    Ordered by neg_log_prob (min-heap → highest probability first).
    """
    neg_log_prob: float                        # −Σ log p(token_i) — minimise
    tokens: List[str]    = field(compare=False)
    pending_slots: int   = field(compare=False)  # slots still needed to complete


# ---------------------------------------------------------------------------
# Core beam search
# ---------------------------------------------------------------------------

@torch.no_grad()
def beam_search_decode(
    policy,
    grammar,
    X: np.ndarray,
    y: np.ndarray,
    evaluator: PrefixEvaluator,
    beam_width: int = 50,
    max_length: int = 30,
    device: str = "cpu",
    top_k_results: int = 10,
) -> List[Dict]:
    """
    Run beam search over `policy` and evaluate all completed beams on (X, y).

    Parameters
    ----------
    policy       : trained SymbolicPolicy (weights frozen — call policy.eval() before)
    grammar      : Grammar object
    X, y         : dataset for NMSE evaluation
    evaluator    : PrefixEvaluator instance
    beam_width   : number of beams kept at each step
    max_length   : maximum expression length in tokens
    device       : 'cpu' or 'cuda'
    top_k_results: number of finished expressions to return (sorted by NMSE)

    Returns
    -------
    List of dicts sorted by nmse ascending:
        {tokens, expr, nmse, quality, log_prob, complexity}
    """
    vocab_size = len(grammar)
    arities    = torch.tensor(
        [grammar.arity[grammar.id_to_token[i]] for i in range(vocab_size)],
        device=device, dtype=torch.long,
    )

    # Active beams: min-heap on neg_log_prob
    active: List[BeamState] = [BeamState(
        neg_log_prob=0.0,
        tokens=[],
        pending_slots=1,
    )]

    finished: List[BeamState] = []

    step = 0
    while active and step < max_length:
        step += 1
        candidates: List[BeamState] = []

        for beam in active:
            # Build token_ids tensor from current partial sequence
            if len(beam.tokens) == 0:
                token_ids = torch.empty(0, dtype=torch.long, device=device)
            else:
                token_ids = torch.tensor(
                    [grammar.token_to_id[t] for t in beam.tokens],
                    dtype=torch.long, device=device,
                )

            # Action mask: only allow tokens that keep pending_slots ≥ 0
            # and don't exceed max_length
            remaining      = max_length - len(beam.tokens)
            new_pending    = beam.pending_slots - 1 + arities   # shape (vocab,)
            valid          = (new_pending >= 0) & (new_pending <= remaining)
            action_mask    = valid.float()

            if action_mask.sum() == 0:
                # Dead beam — no valid action
                continue

            logits, _ = policy(
                token_ids=token_ids,
                pending_slots=beam.pending_slots,
                length=len(beam.tokens),
                action_mask=action_mask,
            )

            log_probs = torch.log_softmax(logits, dim=-1)  # (vocab,)

            # Take top-beam_width valid actions
            masked_log_probs = log_probs.clone()
            masked_log_probs[~valid] = -float("inf")

            top_k   = min(beam_width, int(valid.sum().item()))
            top_vals, top_ids = torch.topk(masked_log_probs, top_k)

            for lp, aid in zip(top_vals.tolist(), top_ids.tolist()):
                token         = grammar.id_to_token[aid]
                new_tokens    = beam.tokens + [token]
                new_pending   = beam.pending_slots - 1 + grammar.arity[token]
                new_neg_lp    = beam.neg_log_prob - lp  # subtract because log_prob < 0

                new_beam = BeamState(
                    neg_log_prob=new_neg_lp,
                    tokens=new_tokens,
                    pending_slots=new_pending,
                )

                if new_pending == 0:
                    # Complete expression
                    finished.append(new_beam)
                else:
                    candidates.append(new_beam)

        # Keep only top beam_width active beams
        candidates.sort()
        active = candidates[:beam_width]

    # Also add any still-active beams if they happen to be complete
    for beam in active:
        if beam.pending_slots == 0:
            finished.append(beam)

    if not finished:
        return []

    # Deduplicate by token sequence
    seen    = set()
    unique  = []
    for beam in finished:
        key = " ".join(beam.tokens)
        if key not in seen:
            seen.add(key)
            unique.append(beam)

    # Evaluate all unique finished expressions on (X, y)
    results = []
    for beam in unique:
        eval_result = evaluator.evaluate(beam.tokens, X, y)
        if not eval_result["is_valid"]:
            continue

        nmse       = eval_result["nmse"]
        complexity = len(beam.tokens)
        expr       = safe_prefix_to_infix(
            beam.tokens, grammar,
            eval_result.get("optimized_constants", []),
        )

        # Penalised score used for ranking only (not reported as NMSE).
        # Prevents degenerate high-complexity expressions like
        # sin(sin(sin(...))) that reduce NMSE marginally via numerical
        # exploitation. alpha=0.01 matches the training reward.
        alpha           = 0.01
        penalised_score = nmse + alpha * complexity

        results.append({
            "tokens":          beam.tokens,
            "expr":            expr,
            "nmse":            round(nmse, 6),
            "quality":         classify_quality(nmse),
            "log_prob":        round(-beam.neg_log_prob, 4),
            "complexity":      complexity,
            "penalised_score": round(penalised_score, 6),
        })

    # Sort by penalised score (NMSE + complexity penalty) ascending.
    # Matches the training reward — avoids degenerate long expressions
    # that barely improve NMSE at the cost of interpretability.
    results.sort(key=lambda r: r["penalised_score"])
    return results[:top_k_results]


# ---------------------------------------------------------------------------
# Policy loader
# ---------------------------------------------------------------------------

def load_policy(
    checkpoint_path: str,
    grammar,
    device: str,
) -> SymbolicPolicy:
    """Load a saved policy checkpoint."""
    policy = SymbolicPolicy(vocab_size=len(grammar)).to(device)
    state  = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and wrapped checkpoint dicts
    if isinstance(state, dict) and "policy_state_dict" in state:
        policy.load_state_dict(state["policy_state_dict"])
    else:
        policy.load_state_dict(state)
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Compare beam search vs training best
# ---------------------------------------------------------------------------

def compare_beam_vs_sampling(
    policy,
    grammar,
    X: np.ndarray,
    y: np.ndarray,
    evaluator: PrefixEvaluator,
    sampling_best_nmse: float,
    sampling_best_expr: str,
    beam_width: int = 50,
    max_length: int = 30,
    device: str = "cpu",
) -> Dict:
    """
    Run beam search and compare against the best expression found during
    stochastic sampling training. Returns a summary dict.
    """
    policy.eval()
    beam_results = beam_search_decode(
        policy=policy,
        grammar=grammar,
        X=X, y=y,
        evaluator=evaluator,
        beam_width=beam_width,
        max_length=max_length,
        device=device,
        top_k_results=10,
    )

    if not beam_results:
        return {
            "beam_nmse":    1.0,
            "beam_expr":    "",
            "beam_quality": "Poor",
            "sampling_nmse": sampling_best_nmse,
            "sampling_expr": sampling_best_expr,
            "improvement":  0.0,
            "winner":       "sampling",
        }

    best_beam = beam_results[0]
    improvement = sampling_best_nmse - best_beam["nmse"]
    winner = "beam" if best_beam["nmse"] < sampling_best_nmse else "sampling"

    return {
        "beam_nmse":      best_beam["nmse"],
        "beam_expr":      best_beam["expr"],
        "beam_quality":   best_beam["quality"],
        "beam_top10":     beam_results,
        "sampling_nmse":  sampling_best_nmse,
        "sampling_expr":  sampling_best_expr,
        "improvement":    round(improvement, 6),
        "winner":         winner,
    }


# ---------------------------------------------------------------------------
# Standalone script
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "difficulty", "true_expression",
    "sampling_nmse", "sampling_expr", "sampling_quality",
    "beam_nmse",     "beam_expr",     "beam_quality",
    "improvement", "winner", "beam_width",
]


def main():
    parser = argparse.ArgumentParser(
        description="Beam search evaluation over a trained RSPG policy."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["pmlb_feynman_subset", "pmlb_feynman_all"])
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")

    parser.add_argument("--sampling_csv", type=str, default=None, metavar="PATH",
                        help="CSV from evaluate_expressions.py with sampling results.")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing policy_final.pt checkpoints saved "
                             "by Trainer. One file per task, named <task_name>.pt")
    parser.add_argument("--num_episodes",   type=int,   default=3000,
                        help="Episodes for training if no checkpoint found")
    parser.add_argument("--beam_width",     type=int,   default=50,
                        help="Number of beams (default 50; paper uses 50-200)")
    parser.add_argument("--max_length",     type=int,   default=30)
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4)
    parser.add_argument("--entropy_weight", type=float, default=0.017)
    parser.add_argument("--seed",           type=int,   default=42)

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  beam_width: {args.beam_width}")

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples) for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    # Load sampling results if available
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
    csv_path    = os.path.join(results_dir, f"beam_search_{timestamp}.csv")

    rows = []

    for idx, task in enumerate(tasks):
        X, y = task.generate()
        gt   = FEYNMAN_GROUND_TRUTH.get(task.name, {})
        print(f"\n[{idx+1}/{len(tasks)}] {task.name}  ({gt.get('difficulty','?')})")

        grammar   = build_grammar(num_variables=task.num_variables)
        evaluator = PrefixEvaluator(grammar)

        # --- Load checkpoint or train from scratch ---
        from .trainer import Trainer

        checkpoint_path = os.path.join(
            args.checkpoint_dir, f"{task.name}.pt"
        )

        if os.path.exists(checkpoint_path):
            # Best case: load the exact policy used during sampling training
            print(f"  Loading checkpoint: {checkpoint_path}")
            trainer = Trainer.load_checkpoint(checkpoint_path, device=device)
            trained_policy = trainer.policy
            trained_policy.eval()
            # Restore dataset embedding
            tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
            tensor_y = torch.tensor(y, dtype=torch.float32, device=device)
            trained_policy.set_dataset_embedding(tensor_X, tensor_y)
        else:
            # Fallback: train from scratch with fixed seed
            print(f"  No checkpoint found at {checkpoint_path}")
            print(f"  Training from scratch (seed={args.seed}, "
                  f"episodes={args.num_episodes})...")
            import random
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

            trainer = Trainer(
                X=X, y=y,
                num_variables=task.num_variables,
                device=device,
                optimizer_name="rspg",
            )
            trainer.num_episodes   = args.num_episodes
            trainer.batch_size     = args.batch_size
            trainer.rl_optimizer.learning_rate  = args.learning_rate
            trainer.rl_optimizer.entropy_weight = args.entropy_weight
            trainer.rl_optimizer.optimizer = torch.optim.Adam(
                trainer.policy.parameters(), lr=args.learning_rate
            )
            train_results = trainer.train(
                checkpoint_dir=args.checkpoint_dir,
                checkpoint_name=task.name,
            )
            trained_policy = trainer.policy
            tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
            tensor_y = torch.tensor(y, dtype=torch.float32, device=device)
            trained_policy.set_dataset_embedding(tensor_X, tensor_y)

        # Sampling best — from CSV or from the checkpoint's stored best episode
        if task.name in sampling_cache:
            sampling_nmse = sampling_cache[task.name]["nmse"]
            sampling_expr = sampling_cache[task.name]["expr"]
            print(f"  Sampling (CSV): NMSE={sampling_nmse:.4f}")
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

        print(f"  Sampling best: NMSE={sampling_nmse:.4f}  expr={sampling_expr[:60]}")

        # --- Beam search ---
        print(f"  Running beam search (width={args.beam_width})...")
        comparison = compare_beam_vs_sampling(
            policy=trained_policy,
            grammar=grammar,
            X=X, y=y,
            evaluator=evaluator,
            sampling_best_nmse=sampling_nmse,
            sampling_best_expr=sampling_expr,
            beam_width=args.beam_width,
            max_length=args.max_length,
            device=device,
        )

        print(f"  Beam best:     NMSE={comparison['beam_nmse']:.4f}  expr={comparison['beam_expr'][:60]}")
        delta = comparison['improvement']
        marker = f"Δ−{delta:.4f} ({'beam wins' if delta > 0 else 'sampling wins'})"
        print(f"  {marker}")

        row = {
            "task":             task.name,
            "difficulty":       gt.get("difficulty", "Unknown"),
            "true_expression":  gt.get("expr", "unknown"),
            "sampling_nmse":    sampling_nmse,
            "sampling_expr":    sampling_expr,
            "sampling_quality": classify_quality(sampling_nmse),
            "beam_nmse":        comparison["beam_nmse"],
            "beam_expr":        comparison["beam_expr"],
            "beam_quality":     comparison["beam_quality"],
            "improvement":      comparison["improvement"],
            "winner":           comparison["winner"],
            "beam_width":       args.beam_width,
        }
        rows.append(row)

        mode = "w" if idx == 0 else "a"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if idx == 0:
                writer.writeheader()
            writer.writerow(row)

    # Console summary
    print("\n" + "=" * 70)
    print(f"BEAM SEARCH SUMMARY  (beam_width={args.beam_width})")
    print("=" * 70)
    w = max(len(r["task"]) for r in rows) + 2
    print(f"{'Task':<{w}} {'Sampling':>10}  {'Beam':>10}  {'Δ NMSE':>10}  Winner")
    print("-" * 70)
    for r in rows:
        delta  = r["improvement"]
        marker = "beam ▲" if delta > 0 else "sampling"
        print(f"{r['task']:<{w}} {r['sampling_nmse']:>10.4f}  "
              f"{r['beam_nmse']:>10.4f}  {delta:>+10.4f}  {marker}")

    beam_wins     = sum(1 for r in rows if r["winner"] == "beam")
    sampling_wins = sum(1 for r in rows if r["winner"] == "sampling")
    n = len(rows)
    print("=" * 70)
    print(f"  Beam wins:     {beam_wins}/{n} ({100*beam_wins/n:.0f}%)")
    print(f"  Sampling wins: {sampling_wins}/{n} ({100*sampling_wins/n:.0f}%)")
    mean_improvement = np.mean([r["improvement"] for r in rows])
    print(f"  Mean NMSE improvement from beam search: {mean_improvement:+.4f}")
    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()