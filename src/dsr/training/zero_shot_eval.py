"""
zero_shot_eval.py
=================
Zero-shot generalisation experiment — the key experiment for publication.

The question
------------
"Does a policy trained on N tasks generalise to new tasks it has never seen?"

gplearn cannot answer yes. It restarts from scratch for every new task.
RSPG with a shared multi-task policy can — by conditioning on the new
dataset's DeepSets embedding, the policy immediately generates reasonable
expressions without any additional training.

Experimental setup
------------------
1. TRAIN split  — train the multi-task policy on N_train tasks
2. TEST split   — evaluate zero-shot on N_test held-out tasks

Three methods compared on the test split:
  a) RSPG zero-shot  — shared policy, no training, just beam search
  b) RSPG few-shot   — shared policy + 500 fine-tuning episodes on test task
  c) gplearn         — genetic programming, restarts from scratch (baseline)

This experiment directly demonstrates the generalization advantage of
learning-based symbolic regression over search-based methods.

Recommended split
-----------------
  Train: feynman_I_12_1, feynman_I_13_4, feynman_I_14_3, feynman_I_14_4,
         feynman_I_18_4, feynman_I_18_12, feynman_I_25_13, feynman_I_29_4,
         feynman_I_34_8, feynman_I_39_1
  Test:  feynman_I_10_7, feynman_I_11_19, feynman_I_12_11, feynman_II_27_18

Usage
-----
# Full experiment (train + zero-shot + few-shot + gplearn):
python -m dsr.training.zero_shot_eval \
    --train_tasks feynman_I_12_1 feynman_I_13_4 feynman_I_14_3 \
                  feynman_I_14_4 feynman_I_18_4 feynman_I_18_12 \
    --test_tasks feynman_I_10_7 feynman_I_12_11 \
    --num_train_episodes 10000 \
    --few_shot_episodes 500 \
    --beam_width 100

# Quick test (2 train, 2 test):
python -m dsr.training.zero_shot_eval \
    --train_tasks feynman_I_12_1 feynman_I_14_3 feynman_I_18_4 feynman_I_25_13 \
    --test_tasks feynman_I_10_7 feynman_I_12_11 \
    --num_train_episodes 5000 \
    --few_shot_episodes 300 \
    --beam_width 50

Output
------
results/
  zero_shot_<timestamp>.csv
  plots/
    zero_shot_comparison.png
"""

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..core.factory import build_grammar
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..data.datasets import get_task_by_name
from ..data.feynman_ground_truth import FEYNMAN_GROUND_TRUTH, classify_quality
from .multitask_trainer import MultiTaskTrainer
from .trainer import Trainer
from .beam_search import beam_search_decode


# ---------------------------------------------------------------------------
# gplearn baseline (same as baseline_gplearn.py but standalone)
# ---------------------------------------------------------------------------

def run_gplearn(
    X: np.ndarray,
    y: np.ndarray,
    population_size: int = 1000,
    generations:     int = 20,
    random_state:    int = 42,
) -> Dict:
    """Run gplearn on (X, y) and return best NMSE."""
    try:
        from gplearn.genetic import SymbolicRegressor
        import warnings

        # Custom protected functions
        from gplearn.functions import make_function
        import numpy as _np

        def _gp_log(x):
            return _np.where(x > 0, _np.log(x), 0.0)
        def _gp_exp(x):
            return _np.where(x < 100, _np.exp(x), _np.exp(100.0))
        def _gp_sqrt(x):
            return _np.sqrt(_np.abs(x))

        try:
            # gplearn >= 0.4.2 uses keyword arguments
            gp_log  = make_function(function=_gp_log,  name="log",  arity=1)
            gp_exp  = make_function(function=_gp_exp,  name="exp",  arity=1)
            gp_sqrt = make_function(function=_gp_sqrt, name="sqrt", arity=1)
        except TypeError:
            # older gplearn uses positional arguments
            gp_log  = make_function(_gp_log,  "log",  1)
            gp_exp  = make_function(_gp_exp,  "exp",  1)
            gp_sqrt = make_function(_gp_sqrt, "sqrt", 1)

        est = SymbolicRegressor(
            population_size=population_size,
            generations=generations,
            function_set=("add", "sub", "mul", "div",
                          "sin", "cos", gp_log, gp_exp, gp_sqrt),
            metric="mse",
            random_state=random_state,
            verbose=0,
            n_jobs=1,
        )

        var_y = float(np.var(y))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X, y)

        y_pred = est.predict(X)
        mse    = float(np.mean((y - y_pred) ** 2))
        nmse   = mse / var_y if var_y > 1e-9 else mse
        nmse   = min(nmse, 1.0)
        expr   = str(est._program)

        return {"nmse": nmse, "expr": expr, "quality": classify_quality(nmse)}

    except ImportError:
        return {"nmse": float("nan"), "expr": "gplearn not installed",
                "quality": "N/A"}
    except Exception as e:
        return {"nmse": 1.0, "expr": f"error: {e}", "quality": "Poor"}


# ---------------------------------------------------------------------------
# Zero-shot evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def zero_shot_eval(
    policy,
    grammar,
    X:           np.ndarray,
    y:           np.ndarray,
    evaluator:   PrefixEvaluator,
    beam_width:  int = 50,
    max_length:  int = 30,
    device:      str = "cpu",
) -> Dict:
    """
    Evaluate a trained policy zero-shot on a new (X, y) task.
    No gradient updates — just set the dataset embedding and run beam search.
    """
    policy.eval()

    # Condition the policy on the new dataset
    tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
    tensor_y = torch.tensor(y, dtype=torch.float32, device=device)
    policy.set_dataset_embedding(tensor_X, tensor_y)

    # Beam search — deterministic exploitation of the policy
    results = beam_search_decode(
        policy=policy,
        grammar=grammar,
        X=X, y=y,
        evaluator=evaluator,
        beam_width=beam_width,
        max_length=max_length,
        device=device,
        top_k_results=10,
    )

    if not results:
        return {"nmse": 1.0, "expr": "", "quality": "Poor", "method": "zero_shot"}

    best = results[0]
    return {
        "nmse":    best["nmse"],
        "expr":    best["expr"],
        "quality": best["quality"],
        "method":  "zero_shot",
    }


# ---------------------------------------------------------------------------
# Few-shot fine-tuning
# ---------------------------------------------------------------------------

def few_shot_eval(
    policy,
    grammar,
    X:               np.ndarray,
    y:               np.ndarray,
    num_variables:   int,
    fine_tune_episodes: int = 500,
    beam_width:      int = 50,
    device:          str = "cpu",
    learning_rate:   float = 3.35e-4,
) -> Dict:
    """
    Fine-tune the shared policy on a new task for a small number of episodes,
    then evaluate with beam search.
    This is the few-shot condition — minimal adaptation.
    """
    import copy
    from .risk_seeking_optimizer import RiskSeekingOptimizer
    from .rollout import collect_batched_episodes
    from ..core.env import SymbolicRegressionEnv
    from ..analysis.memory import TopKMemory

    # Clone policy weights — don't modify the shared policy
    ft_policy = copy.deepcopy(policy)
    ft_policy.train()

    optimizer = RiskSeekingOptimizer(ft_policy)
    optimizer.optimizer = torch.optim.Adam(
        ft_policy.parameters(), lr=learning_rate
    )

    evaluator = PrefixEvaluator(grammar)
    env       = SymbolicRegressionEnv(X, y, grammar)
    memory    = TopKMemory(capacity=20)

    # Set dataset embedding
    tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
    tensor_y = torch.tensor(y, dtype=torch.float32, device=device)
    ft_policy.set_dataset_embedding(tensor_X, tensor_y)

    best_reward = float("-inf")
    best_tokens: List[str] = []
    episode_idx = 0
    batch_size  = 64   # small batches for few-shot

    while episode_idx < fine_tune_episodes:
        current_batch = min(batch_size, fine_tune_episodes - episode_idx)
        episodes = collect_batched_episodes(
            env_template=env,
            policy=ft_policy,
            grammar=grammar,
            batch_size=current_batch,
            max_length=30,
            device=device,
        )

        for ep in episodes:
            er = evaluator.evaluate(ep["tokens"], X, y)
            reward = (
                -er["nmse"] - 0.01 * len(ep["tokens"])
                if er["is_valid"] else -1.0
            )
            L = len(ep["tokens"])
            ep["rewards"] = [0.0] * L
            if L > 0:
                ep["rewards"][-1] = reward
            ep["final_reward"] = reward

            if reward > best_reward:
                best_reward = reward
                best_tokens = ep["tokens"]

            episode_idx += 1

        optimizer.update(episodes)

    # Final beam search
    ft_policy.eval()
    ft_policy.set_dataset_embedding(tensor_X, tensor_y)
    results = beam_search_decode(
        policy=ft_policy,
        grammar=grammar,
        X=X, y=y,
        evaluator=evaluator,
        beam_width=beam_width,
        max_length=30,
        device=device,
    )

    if results:
        best = results[0]
        return {
            "nmse":    best["nmse"],
            "expr":    best["expr"],
            "quality": best["quality"],
            "method":  "few_shot",
        }

    # Fallback to sampling best
    if best_tokens:
        er   = evaluator.evaluate(best_tokens, X, y)
        nmse = er.get("nmse", 1.0)
        expr = safe_prefix_to_infix(best_tokens, grammar,
                                    er.get("optimized_constants", []))
        return {"nmse": nmse, "expr": expr,
                "quality": classify_quality(nmse), "method": "few_shot"}

    return {"nmse": 1.0, "expr": "", "quality": "Poor", "method": "few_shot"}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(rows: List[Dict], out_dir: str):
    """Bar chart comparing zero-shot, few-shot, and gplearn on test tasks."""
    test_tasks = sorted(set(r["task"] for r in rows))
    methods    = ["zero_shot", "few_shot", "gplearn"]
    colors     = {"zero_shot": "#D85A30", "few_shot": "#EF9F27", "gplearn": "#378ADD"}
    labels     = {"zero_shot": "RSPG zero-shot", "few_shot": "RSPG few-shot",
                  "gplearn": "gplearn (baseline)"}

    x     = np.arange(len(test_tasks))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(test_tasks) * 3), 5))

    for i, method in enumerate(methods):
        nmses = []
        for task in test_tasks:
            task_rows = [r for r in rows if r["task"] == task
                         and r["method"] == method]
            nmses.append(task_rows[0]["nmse"] if task_rows else float("nan"))

        offset = (i - 1) * width
        bars   = ax.bar(x + offset, nmses, width,
                        color=colors[method], alpha=0.85, label=labels[method])
        for bar, val in zip(bars, nmses):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("feynman_", "") for t in test_tasks],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Best NMSE (lower = better)", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_title("Zero-shot generalisation: RSPG (multi-task) vs gplearn",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    ax.axhline(1.0, color="#aaa", lw=0.8, linestyle="--")

    plt.tight_layout()
    path = os.path.join(out_dir, "zero_shot_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "difficulty", "true_expression",
    "method", "nmse", "expr", "quality",
]


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot generalisation: multi-task RSPG vs gplearn."
    )
    parser.add_argument("--train_tasks", nargs="+", required=True,
                        metavar="TASK",
                        help="Tasks used to train the shared policy")
    parser.add_argument("--test_tasks",  nargs="+", required=True,
                        metavar="TASK",
                        help="Held-out tasks for zero-shot evaluation")
    parser.add_argument("--num_train_episodes", type=int, default=8000,
                        help="Total episodes for multi-task training")
    parser.add_argument("--few_shot_episodes",  type=int, default=300,
                        help="Fine-tuning episodes for few-shot condition")
    parser.add_argument("--beam_width",         type=int, default=50)
    parser.add_argument("--num_samples",        type=int, default=100)
    parser.add_argument("--gp_generations",     type=int, default=20)
    parser.add_argument("--gp_population",      type=int, default=1000)
    parser.add_argument("--seed",               type=int, default=42)
    parser.add_argument("--skip_gplearn",       action="store_true")
    parser.add_argument("--skip_few_shot",      action="store_true")

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Device: {device}")
    print(f"Train tasks ({len(args.train_tasks)}): {args.train_tasks}")
    print(f"Test  tasks ({len(args.test_tasks)}):  {args.test_tasks}")

    # Load all tasks
    train_task_objs = [get_task_by_name(n, num_samples=args.num_samples)
                       for n in args.train_tasks]
    test_task_objs  = [get_task_by_name(n, num_samples=args.num_samples)
                       for n in args.test_tasks]

    # --- Phase 1: Multi-task training ---
    print(f"\n{'='*60}")
    print(f"PHASE 1 — Multi-task training ({args.num_train_episodes} episodes)")
    print(f"{'='*60}")

    train_tasks = []
    for task in train_task_objs:
        X, y = task.generate()
        train_tasks.append({
            "name": task.name, "X": X, "y": y,
            "num_variables": task.num_variables,
        })

    mt_trainer = MultiTaskTrainer(
        tasks=train_tasks,
        device=device,
        num_episodes=args.num_train_episodes,
        task_sampling="prioritized",   # focus on harder tasks
    )
    mt_trainer.train()

    shared_policy = mt_trainer.policy
    shared_grammar = mt_trainer.grammar

    # --- Phase 2: Zero-shot evaluation on test tasks ---
    print(f"\n{'='*60}")
    print(f"PHASE 2 — Zero-shot evaluation on {len(test_task_objs)} held-out tasks")
    print(f"{'='*60}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, f"zero_shot_{timestamp}.csv")
    all_rows    = []

    for task in test_task_objs:
        X, y = task.generate()
        gt   = FEYNMAN_GROUND_TRUTH.get(task.name, {})

        print(f"\n--- {task.name}  ({gt.get('difficulty','?')}) ---")
        print(f"    true: {gt.get('expr','?')}")

        evaluator = PrefixEvaluator(shared_grammar)

        # Zero-shot
        print(f"  [zero-shot] beam_width={args.beam_width}...")
        zs = zero_shot_eval(
            policy=shared_policy,
            grammar=shared_grammar,
            X=X, y=y,
            evaluator=evaluator,
            beam_width=args.beam_width,
            device=device,
        )
        print(f"  zero-shot:  NMSE={zs['nmse']:.4f}  {zs['expr'][:60]}")

        # Few-shot
        fs = {"nmse": float("nan"), "expr": "", "quality": "N/A",
              "method": "few_shot"}
        if not args.skip_few_shot:
            print(f"  [few-shot]  {args.few_shot_episodes} episodes...")
            fs = few_shot_eval(
                policy=shared_policy,
                grammar=shared_grammar,
                X=X, y=y,
                num_variables=task.num_variables,
                fine_tune_episodes=args.few_shot_episodes,
                beam_width=args.beam_width,
                device=device,
            )
            print(f"  few-shot:   NMSE={fs['nmse']:.4f}  {fs['expr'][:60]}")

        # gplearn
        gp = {"nmse": float("nan"), "expr": "", "quality": "N/A",
              "method": "gplearn"}
        if not args.skip_gplearn:
            print(f"  [gplearn]   pop={args.gp_population} "
                  f"gen={args.gp_generations}...")
            gp = run_gplearn(
                X=X, y=y,
                population_size=args.gp_population,
                generations=args.gp_generations,
                random_state=args.seed,
            )
            gp["method"] = "gplearn"
            print(f"  gplearn:    NMSE={gp['nmse']:.4f}  {gp['expr'][:60]}")

        for result in [zs, fs, gp]:
            row = {
                "task":            task.name,
                "difficulty":      gt.get("difficulty", "Unknown"),
                "true_expression": gt.get("expr", "unknown"),
                "method":          result["method"],
                "nmse":            result["nmse"],
                "expr":            result.get("expr", ""),
                "quality":         result.get("quality", "N/A"),
            }
            all_rows.append(row)

    # --- Summary ---
    print(f"\n{'='*70}")
    print("ZERO-SHOT GENERALISATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Task':<25} {'Zero-shot':>12} {'Few-shot':>10} {'gplearn':>10}")
    print("-" * 70)

    for task in test_task_objs:
        name = task.name
        zs_r = next((r for r in all_rows
                     if r["task"]==name and r["method"]=="zero_shot"), {})
        fs_r = next((r for r in all_rows
                     if r["task"]==name and r["method"]=="few_shot"), {})
        gp_r = next((r for r in all_rows
                     if r["task"]==name and r["method"]=="gplearn"), {})

        print(f"{name.replace('feynman_',''):<25} "
              f"{zs_r.get('nmse', float('nan')):>12.4f} "
              f"{fs_r.get('nmse', float('nan')):>10.4f} "
              f"{gp_r.get('nmse', float('nan')):>10.4f}")

    # Aggregate
    methods = ["zero_shot", "few_shot", "gplearn"]
    print("-" * 70)
    for method in methods:
        vals = [r["nmse"] for r in all_rows
                if r["method"] == method and not np.isnan(r["nmse"])]
        if vals:
            print(f"  {'Mean NMSE':<10} {method:<15}: {np.mean(vals):.4f}")

    # Save
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    plot_results(all_rows, plots_dir)

    print(f"\nCSV → {csv_path}")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()