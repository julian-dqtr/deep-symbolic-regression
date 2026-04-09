import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..core.factory import build_grammar
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..data.datasets import get_task_suite, get_task_by_name
from ..data.feynman_ground_truth import FEYNMAN_GROUND_TRUTH, classify_quality
from .trainer import Trainer


# ---------------------------------------------------------------------------
# Embedding computation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_dataset_embedding(
    X:      np.ndarray,
    y:      np.ndarray,
    policy,
    device: str,
) -> np.ndarray:
    """
    Compute the DeepSets embedding for a dataset (X, y).
    Uses the policy's dataset_encoder — same encoder used during training.

    Returns a 1D numpy array of shape (dataset_embedding_dim,).
    """
    tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
    tensor_y = torch.tensor(y, dtype=torch.float32, device=device)

    # Build encoder if not already built
    policy._build_dataset_encoder_if_needed(num_features=X.shape[1])
    embedding = policy.encode_dataset(tensor_X, tensor_y)
    return embedding.cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns value in [-1, 1]."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L2 distance between two embedding vectors."""
    return float(np.linalg.norm(a - b))


# ---------------------------------------------------------------------------
# Checkpoint catalogue
# ---------------------------------------------------------------------------

def load_checkpoint_catalogue(
    checkpoint_dir: str,
    device:         str,
    exclude_task:   Optional[str] = None,
) -> List[Dict]:
    """
    Load all .pt checkpoints from checkpoint_dir and compute their
    DeepSets embeddings. Returns a list of catalogue entries:
        {path, task_name, X, y, num_variables, embedding, best_reward}

    exclude_task : skip this task (the one we're warm-starting for,
                   to avoid using its own checkpoint).
    """
    catalogue = []
    if not os.path.isdir(checkpoint_dir):
        return catalogue

    for fname in os.listdir(checkpoint_dir):
        if not fname.endswith(".pt"):
            continue

        task_name = fname[:-3]   # strip .pt
        if task_name == exclude_task:
            continue

        path = os.path.join(checkpoint_dir, fname)
        try:
            state = torch.load(path, map_location=device, weights_only=False)
        except Exception:
            continue

        # Must have stored X, y, and policy weights
        if not all(k in state for k in ["X", "y", "policy_state_dict",
                                         "policy_vocab_size"]):
            continue

        X_ckpt        = state["X"]
        y_ckpt        = state["y"]
        num_variables = state.get("num_variables", X_ckpt.shape[1])
        best_reward   = state.get("best_reward", -1.0)

        # Build a temporary policy to compute the embedding
        from ..models.policy import SymbolicPolicy
        tmp_policy = SymbolicPolicy(vocab_size=state["policy_vocab_size"]).to(device)
        # Build the dataset encoder before loading state_dict — otherwise
        # the encoder weights in the checkpoint have no target to load into.
        tmp_X = torch.tensor(X_ckpt[:2], dtype=torch.float32, device=device)
        tmp_y = torch.tensor(y_ckpt[:2], dtype=torch.float32, device=device)
        tmp_policy._build_dataset_encoder_if_needed(num_features=X_ckpt.shape[1])
        tmp_policy.load_state_dict(state["policy_state_dict"], strict=False)
        tmp_policy.eval()

        try:
            embedding = compute_dataset_embedding(X_ckpt, y_ckpt, tmp_policy, device)
        except Exception:
            continue

        catalogue.append({
            "path":          path,
            "task_name":     task_name,
            "X":             X_ckpt,
            "y":             y_ckpt,
            "num_variables": num_variables,
            "embedding":     embedding,
            "best_reward":   best_reward,
        })

    return catalogue


# ---------------------------------------------------------------------------
# Similarity search
# ---------------------------------------------------------------------------

def find_most_similar_checkpoint(
    X:              np.ndarray,
    y:              np.ndarray,
    num_variables:  int,
    checkpoint_dir: str,
    device:         str,
    exclude_task:   Optional[str] = None,
    metric:         str = "cosine",   # "cosine" or "euclidean"
    top_k:          int = 3,
) -> Optional[Dict]:
    """
    Find the checkpoint whose dataset embedding is most similar to (X, y).

    Parameters
    ----------
    metric      : similarity metric — 'cosine' (default) or 'euclidean'
    top_k       : print the top-k most similar checkpoints for debugging
    exclude_task: task name to exclude (avoid self-comparison)

    Returns the most similar catalogue entry, or None if no checkpoints found.
    """
    catalogue = load_checkpoint_catalogue(checkpoint_dir, device, exclude_task)
    if not catalogue:
        return None

    # Compute embedding of the query task using first catalogue policy
    # (all policies share the same DeepSets architecture)
    from ..models.policy import SymbolicPolicy
    first_state = torch.load(
        catalogue[0]["path"], map_location=device, weights_only=False
    )
    tmp_policy = SymbolicPolicy(
        vocab_size=first_state["policy_vocab_size"]
    ).to(device)
    # Build encoder with query task dimensions (for computing query embedding)
    # Use strict=False so mismatched dataset_encoder keys are ignored
    tmp_policy._build_dataset_encoder_if_needed(num_features=X.shape[1])
    try:
        tmp_policy.load_state_dict(first_state["policy_state_dict"], strict=False)
    except Exception:
        pass
    tmp_policy.eval()

    query_emb = compute_dataset_embedding(X, y, tmp_policy, device)

    # Score each catalogue entry
    scored = []
    for entry in catalogue:
        if metric == "cosine":
            sim   = cosine_similarity(query_emb, entry["embedding"])
            score = sim    # higher = more similar
        else:
            dist  = euclidean_distance(query_emb, entry["embedding"])
            score = -dist  # higher = more similar (negate distance)

        scored.append({**entry, "similarity": score})

    scored.sort(key=lambda e: e["similarity"], reverse=True)

    # Print top-k for transparency
    print(f"  Top-{min(top_k, len(scored))} most similar checkpoints:")
    for rank, entry in enumerate(scored[:top_k]):
        print(f"    {rank+1}. {entry['task_name']:<30} "
              f"similarity={entry['similarity']:+.4f}  "
              f"best_reward={entry['best_reward']:.4f}")

    return scored[0]


# ---------------------------------------------------------------------------
# Warm-start trainer factory
# ---------------------------------------------------------------------------

def warm_start_trainer(
    X:              np.ndarray,
    y:              np.ndarray,
    num_variables:  int,
    checkpoint_dir: str,
    device:         str,
    exclude_task:   Optional[str] = None,
    metric:         str = "cosine",
    **trainer_kwargs,
) -> Tuple[Trainer, Optional[str]]:
    """
    Create a Trainer warm-started from the most similar checkpoint.

    Returns (trainer, source_task_name) where source_task_name is the
    task whose weights were used for initialisation (None if cold start).
    """
    best_match = find_most_similar_checkpoint(
        X=X, y=y,
        num_variables=num_variables,
        checkpoint_dir=checkpoint_dir,
        device=device,
        exclude_task=exclude_task,
        metric=metric,
    )

    if best_match is None:
        print("  No checkpoints found — cold start.")
        trainer = Trainer(
            X=X, y=y,
            num_variables=num_variables,
            device=device,
            **trainer_kwargs,
        )
        return trainer, None

    print(f"  Warm-starting from: {best_match['task_name']} "
          f"(similarity={best_match['similarity']:+.4f})")

    # Build a fresh trainer for the new task — correct grammar & vocab
    grammar   = build_grammar(num_variables=num_variables)
    evaluator = PrefixEvaluator(grammar)

    trainer = Trainer(
        X=X, y=y,
        num_variables=num_variables,
        device=device,
        **trainer_kwargs,
    )

    # Load checkpoint weights — only transfer the parts that are
    # independent of num_variables: LSTM, token_embedding, state_mlp,
    # action_head, value_head. Skip dataset_encoder (vocab-dependent).
    state = torch.load(
        best_match["path"], map_location=device, weights_only=False
    )
    src_state  = state["policy_state_dict"]
    tgt_state  = trainer.policy.state_dict()

    # Keys to transfer: everything except dataset_encoder.*
    transferable = {
        k: v for k, v in src_state.items()
        if not k.startswith("dataset_encoder.")
        and k in tgt_state
        and v.shape == tgt_state[k].shape
    }
    tgt_state.update(transferable)
    trainer.policy.load_state_dict(tgt_state)

    n_transferred = len(transferable)
    n_total       = len(tgt_state)
    print(f"  Transferred {n_transferred}/{n_total} parameter tensors "
          f"(skipped dataset_encoder — incompatible dimensions)")

    # Refresh dataset embedding for the new task
    tensor_X = torch.tensor(
        np.asarray(X, dtype=np.float32), dtype=torch.float32, device=device
    )
    tensor_y = torch.tensor(
        np.asarray(y, dtype=np.float32), dtype=torch.float32, device=device
    )
    trainer.policy.set_dataset_embedding(tensor_X, tensor_y)

    return trainer, best_match["task_name"]


# ---------------------------------------------------------------------------
# Standalone comparison script
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "difficulty", "num_vars", "true_expression",
    "cold_nmse", "cold_expr", "cold_quality",
    "warm_nmse", "warm_expr", "warm_quality",
    "warm_source", "warm_similarity",
    "improvement", "winner",
]


def run_one(
    X, y, num_variables, task_name, num_episodes, device,
    warm: bool, checkpoint_dir: str,
) -> Dict:
    """Run one variant (warm or cold) and return metrics."""
    if warm:
        trainer, source = warm_start_trainer(
            X=X, y=y,
            num_variables=num_variables,
            checkpoint_dir=checkpoint_dir,
            device=device,
            exclude_task=task_name,
            optimizer_name="rspg",
        )
        trainer.num_episodes = num_episodes
    else:
        trainer = Trainer(
            X=X, y=y,
            num_variables=num_variables,
            device=device,
            optimizer_name="rspg",
        )
        trainer.num_episodes = num_episodes
        source = None

    results = trainer.train()
    best_ep = results["best_episode"]
    evaluator = PrefixEvaluator(trainer.grammar)

    nmse, expr = 1.0, ""
    if best_ep is not None:
        er   = evaluator.evaluate(best_ep["tokens"], X, y)
        nmse = er.get("nmse", 1.0)
        expr = safe_prefix_to_infix(
            best_ep["tokens"], trainer.grammar,
            er.get("optimized_constants", []),
        )

    return {
        "nmse":    nmse,
        "expr":    expr,
        "quality": classify_quality(nmse),
        "source":  source,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Warm-start by dataset similarity vs cold start — ablation."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["pmlb_feynman_subset", "pmlb_feynman_all"])

    parser.add_argument("--checkpoint_dir",  type=str, default="checkpoints")
    parser.add_argument("--num_episodes",    type=int, default=2000)
    parser.add_argument("--num_samples",     type=int, default=100)
    parser.add_argument("--metric",          type=str, default="cosine",
                        choices=["cosine", "euclidean"])
    parser.add_argument("--compare_cold",    action="store_true",
                        help="Also run cold start for comparison (doubles runtime)")

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  metric: {args.metric}")

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples) for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, f"warm_start_{timestamp}.csv")
    rows        = []

    for idx, task in enumerate(tasks):
        X, y = task.generate()
        gt   = FEYNMAN_GROUND_TRUTH.get(task.name, {})
        print(f"\n[{idx+1}/{len(tasks)}] {task.name}  ({gt.get('difficulty','?')})")
        print(f"  num_vars={task.num_variables}  true: {gt.get('expr','?')[:50]}")

        # Warm start
        print(f"\n  --- Warm start ---")
        warm_result = run_one(
            X=X, y=y, num_variables=task.num_variables,
            task_name=task.name,
            num_episodes=args.num_episodes,
            device=device,
            warm=True,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Cold start (optional)
        cold_result = {"nmse": float("nan"), "expr": "", "quality": "N/A"}
        if args.compare_cold:
            print(f"\n  --- Cold start ---")
            cold_result = run_one(
                X=X, y=y, num_variables=task.num_variables,
                task_name=task.name,
                num_episodes=args.num_episodes,
                device=device,
                warm=False,
                checkpoint_dir=args.checkpoint_dir,
            )

        improvement = (cold_result["nmse"] - warm_result["nmse"]
                       if not np.isnan(cold_result["nmse"]) else float("nan"))
        winner      = ("warm" if improvement > 0
                       else "cold" if improvement < 0
                       else "tie")

        print(f"\n  Warm: NMSE={warm_result['nmse']:.4f}  "
              f"quality={warm_result['quality']}  "
              f"source={warm_result['source']}")
        if args.compare_cold:
            print(f"  Cold: NMSE={cold_result['nmse']:.4f}  "
                  f"quality={cold_result['quality']}")
            if not np.isnan(improvement):
                marker = "warm ▲" if improvement > 0 else "cold ▲"
                print(f"  Δ={improvement:+.4f}  {marker}")

        row = {
            "task":            task.name,
            "difficulty":      gt.get("difficulty", "Unknown"),
            "num_vars":        task.num_variables,
            "true_expression": gt.get("expr", "unknown"),
            "cold_nmse":       cold_result["nmse"],
            "cold_expr":       cold_result["expr"],
            "cold_quality":    cold_result["quality"],
            "warm_nmse":       warm_result["nmse"],
            "warm_expr":       warm_result["expr"],
            "warm_quality":    warm_result["quality"],
            "warm_source":     warm_result["source"] or "",
            "warm_similarity": "",
            "improvement":     improvement,
            "winner":          winner,
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
    print(f"WARM-START SUMMARY  ({args.num_episodes} episodes)")
    print("=" * 70)
    w = max(len(r["task"]) for r in rows) + 2
    print(f"{'Task':<{w}} {'Warm NMSE':>12}  {'Source task':<30}")
    print("-" * 70)
    for r in rows:
        print(f"{r['task']:<{w}} {r['warm_nmse']:>12.4f}  "
              f"{r['warm_source']:<30}")

    if args.compare_cold:
        warm_nmses = [r["warm_nmse"] for r in rows
                      if not np.isnan(r["warm_nmse"])]
        cold_nmses = [r["cold_nmse"] for r in rows
                      if not np.isnan(r["cold_nmse"])]
        if warm_nmses and cold_nmses:
            print(f"\n  Mean NMSE — Warm: {np.mean(warm_nmses):.4f}  "
                  f"Cold: {np.mean(cold_nmses):.4f}  "
                  f"Δ={np.mean(cold_nmses)-np.mean(warm_nmses):+.4f}")

    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()