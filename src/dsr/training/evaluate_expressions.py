"""
evaluate_expressions.py
========================
Generates the "True vs Recovered Expression" CSV table —
the central result of any symbolic regression paper.

For each Feynman task:
  1. Trains the RSPG agent (or loads results from a pre-existing CSV).
  2. Evaluates the best recovered expression.
  3. Classifies difficulty: Easy / Medium / Hard.
  4. Classifies quality:    Perfect (NMSE < 0.001) / Good (< 0.05) / Poor.
  5. Saves one clean CSV ready for Excel, LaTeX, or pandas.

Usage
-----
# Train from scratch on the subset (~2h CPU):
python -m dsr.training.evaluate_expressions --suite pmlb_feynman_subset

# Load from an existing compare_optimizers CSV — no re-training:
python -m dsr.training.evaluate_expressions \
    --from_csv results/optimizer_comparison_<timestamp>.csv \
    --optimizer rspg

# Explicit task list:
python -m dsr.training.evaluate_expressions \
    --tasks feynman_I_8_14 feynman_I_10_7 feynman_I_12_1

Output
------
results/true_vs_recovered_<timestamp>.csv

Columns
-------
task, difficulty, num_vars, true_expression, recovered_expression,
nmse, complexity, quality, seed
"""

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from ..data.datasets import get_task_suite, get_task_by_name
from ..data.feynman_ground_truth import (
    FEYNMAN_GROUND_TRUTH,
    NMSE_PERFECT,
    NMSE_GOOD,
    DIFF_ORDER,
    classify_quality,
)
from .trainer import Trainer
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix


FIELDNAMES = [
    "task", "difficulty", "num_vars",
    "true_expression", "recovered_expression",
    "nmse", "complexity", "quality", "seed",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_evaluate(task, num_episodes, batch_size,
                       learning_rate, entropy_weight, device, seed) -> Dict:
    set_seed(seed)
    X, y = task.generate()

    trainer = Trainer(
        X=X, y=y,
        num_variables=task.num_variables,
        device=device,
        optimizer_name="rspg",
    )
    trainer.num_episodes   = num_episodes
    trainer.batch_size     = batch_size
    trainer.learning_rate  = learning_rate
    trainer.entropy_weight = entropy_weight
    trainer.rl_optimizer.optimizer = torch.optim.Adam(
        trainer.policy.parameters(), lr=learning_rate
    )

    results   = trainer.train()
    best_ep   = results["best_episode"]
    evaluator = PrefixEvaluator(trainer.grammar)

    if best_ep is not None:
        eval_result = evaluator.evaluate(best_ep["tokens"], X, y)
        nmse        = eval_result.get("nmse", 1.0)
        recovered   = safe_prefix_to_infix(
            best_ep["tokens"], trainer.grammar,
            eval_result.get("optimized_constants", [])
        )
        complexity = len(best_ep["tokens"])
    else:
        nmse, recovered, complexity = 1.0, "", 0

    gt = FEYNMAN_GROUND_TRUTH.get(task.name, {})
    return {
        "task":                 task.name,
        "difficulty":           gt.get("difficulty", "Unknown"),
        "num_vars":             task.num_variables,
        "true_expression":      gt.get("expr", "unknown"),
        "recovered_expression": recovered,
        "nmse":                 round(nmse, 6),
        "complexity":           complexity,
        "quality":              classify_quality(nmse),
        "seed":                 seed,
    }


def load_from_csv(csv_path: str, optimizer: str) -> List[Dict]:
    """Re-use results already computed by compare_optimizers.py."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("optimizer", optimizer) != optimizer:
                continue
            task_name = row["task"]
            nmse      = float(row.get("best_nmse", 1.0))
            gt        = FEYNMAN_GROUND_TRUTH.get(task_name, {})
            rows.append({
                "task":                 task_name,
                "difficulty":           gt.get("difficulty", "Unknown"),
                "num_vars":             row.get("num_vars", ""),
                "true_expression":      gt.get("expr", "unknown"),
                "recovered_expression": row.get("best_expr", ""),
                "nmse":                 round(nmse, 6),
                "complexity":           row.get("complexity", ""),
                "quality":              classify_quality(nmse),
                "seed":                 row.get("seed", ""),
            })
    return rows


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(rows: List[Dict]):
    order = {d: i for i, d in enumerate(DIFF_ORDER)}
    rows  = sorted(rows, key=lambda r: (order.get(r["difficulty"], 9), r["nmse"]))

    w = max(len(r["task"]) for r in rows) + 2
    print()
    print(f"{'Task':<{w}} {'Diff.':<8} {'NMSE':>8}  {'Quality':<8}  Recovered expression")
    print("-" * 110)
    for r in rows:
        expr = r["recovered_expression"]
        if len(expr) > 55:
            expr = expr[:52] + "..."
        print(f"{r['task']:<{w}} {r['difficulty']:<8} {r['nmse']:>8.4f}  {r['quality']:<8}  {expr}")

    print("\n" + "=" * 60)
    print("Summary by difficulty tier")
    print("=" * 60)
    for diff in DIFF_ORDER:
        subset = [r for r in rows if r["difficulty"] == diff]
        if not subset:
            continue
        n       = len(subset)
        perfect = sum(1 for r in subset if r["quality"] == "Perfect")
        good    = sum(1 for r in subset if r["quality"] == "Good")
        poor    = sum(1 for r in subset if r["quality"] == "Poor")
        print(f"  {diff:<8}: {n:>3} tasks | "
              f"Perfect {perfect:>2} ({100*perfect/n:4.0f}%)  "
              f"Good {good:>2} ({100*good/n:4.0f}%)  "
              f"Poor {poor:>2} ({100*poor/n:4.0f}%)")

    n       = len(rows)
    perfect = sum(1 for r in rows if r["quality"] == "Perfect")
    goodp   = sum(1 for r in rows if r["quality"] in ("Perfect", "Good"))
    print("-" * 60)
    print(f"  {'TOTAL':<8}: {n:>3} tasks | "
          f"Perfect {perfect:>2} ({100*perfect/n:4.1f}%)  "
          f"Good+= {goodp:>2} ({100*goodp/n:4.1f}%)")
    print()


# ---------------------------------------------------------------------------
# CSV output — sorted by difficulty then NMSE
# ---------------------------------------------------------------------------

def save_csv(rows: List[Dict], path: str):
    order = {d: i for i, d in enumerate(DIFF_ORDER)}
    rows  = sorted(rows, key=lambda r: (order.get(r["difficulty"], 9), r["nmse"]))
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="True vs Recovered Expressions — Feynman SR benchmark."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["nguyen", "nguyen_univariate", "nguyen_bivariate", "pmlb_feynman_subset", "pmlb_feynman_all"])
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")
    src.add_argument("--from_csv", type=str, default=None, metavar="PATH",
                     help="Load from an existing compare_optimizers CSV (no re-training)")

    parser.add_argument("--optimizer",      type=str,   default="rspg",
                        choices=["rspg", "reinforce", "ppo"])
    parser.add_argument("--num_episodes",   type=int,   default=3000)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4)
    parser.add_argument("--entropy_weight", type=float, default=0.017)
    parser.add_argument("--seed",           type=int,   default=42)

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv   = os.path.join("results", f"true_vs_recovered_{timestamp}.csv")
    os.makedirs("results", exist_ok=True)

    # --- Collect results ---
    if args.from_csv:
        print(f"Loading from {args.from_csv}  (optimizer={args.optimizer})")
        rows = load_from_csv(args.from_csv, args.optimizer)

    else:
        if args.tasks:
            tasks = [get_task_by_name(n, num_samples=args.num_samples)
                     for n in args.tasks]
        else:
            tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

        rows = []
        for idx, task in enumerate(tasks):
            print(f"\n[{idx+1}/{len(tasks)}] {task.name}")
            row = train_and_evaluate(
                task=task,
                num_episodes=args.num_episodes,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                entropy_weight=args.entropy_weight,
                device=device,
                seed=args.seed,
            )
            rows.append(row)
            print(f"  NMSE={row['nmse']:.4f}  quality={row['quality']}")

            # Write each row immediately — safe against mid-run crashes
            mode = "w" if idx == 0 else "a"
            with open(out_csv, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                if idx == 0:
                    writer.writeheader()
                writer.writerow(row)

    # --- Final sorted CSV + console summary ---
    print_summary(rows)
    save_csv(rows, out_csv)


if __name__ == "__main__":
    main()