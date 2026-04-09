import argparse
import csv
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..core.factory import build_grammar
from ..core.env import SymbolicRegressionEnv
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..models.policy import SymbolicPolicy
from ..analysis.memory import TopKMemory
from ..data.datasets import get_task_suite, get_task_by_name
from ..data.feynman_ground_truth import FEYNMAN_GROUND_TRUTH, classify_quality
from .rollout import collect_batched_episodes, recompute_episode
from .risk_seeking_optimizer import RiskSeekingOptimizer
from .trainer import normalize_device, curriculum_max_length, Trainer


# ---------------------------------------------------------------------------
# Multi-task trainer
# ---------------------------------------------------------------------------

class MultiTaskTrainer:
    """
    Trains a single shared policy across multiple Feynman tasks.

    The policy is conditioned on the current task via its DeepSets embedding —
    at each batch, one task is sampled and the embedding is updated accordingly.

    Parameters
    ----------
    tasks        : list of (task_name, X, y, num_variables) tuples
    device       : 'cpu' or 'cuda'
    num_episodes : total training episodes across all tasks
    batch_size   : episodes per update
    task_sampling: 'uniform' (default) or 'prioritized' (harder tasks more often)
    use_curriculum: apply curriculum on max_length
    """

    def __init__(
        self,
        tasks:           List[Dict],   # [{"name", "X", "y", "num_variables"}]
        device:          str   = "cpu",
        num_episodes:    int   = 10000,
        batch_size:      int   = 256,
        learning_rate:   float = 3.35e-4,
        entropy_weight:  float = 0.017,
        use_curriculum:  bool  = False,
        max_length:      int   = 30,
        task_sampling:   str   = "uniform",
    ):
        self.device       = normalize_device(device)
        self.tasks        = tasks
        self.num_episodes = num_episodes
        self.batch_size   = batch_size
        self.max_length   = max_length
        self.task_sampling = task_sampling
        self.use_curriculum = use_curriculum

        # Shared grammar — uses max_num_variables so vocab is consistent
        # across tasks with different numbers of variables
        from ..core.config import GRAMMAR_CONFIG
        max_vars = max(t["num_variables"] for t in tasks)
        self.grammar = build_grammar(num_variables=max_vars)

        # Shared policy
        self.policy = SymbolicPolicy(vocab_size=len(self.grammar)).to(self.device)

        # Pre-build dataset encoder with max_vars so it never needs to be
        # rebuilt when switching between tasks of different num_variables.
        # input_dim = max_vars + 1 (the +1 is for the y target column)
        self.policy._build_dataset_encoder_if_needed(num_features=max_vars)

        # Shared optimizer
        self.rl_optimizer = RiskSeekingOptimizer(self.policy)
        self.rl_optimizer.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate
        )
        self.rl_optimizer.entropy_weight = entropy_weight

        # Per-task state
        self.task_memories:   Dict[str, TopKMemory]         = {}
        self.task_evaluators: Dict[str, PrefixEvaluator]    = {}
        self.task_envs:       Dict[str, SymbolicRegressionEnv] = {}
        self.task_best:       Dict[str, Dict]               = {}

        for task in tasks:
            name = task["name"]
            X, y = task["X"], task["y"]
            # Each task uses the shared grammar (with max_vars variables)
            # Variables beyond num_variables for the task are simply unused
            self.task_memories[name]   = TopKMemory(capacity=20)
            self.task_evaluators[name] = PrefixEvaluator(self.grammar)
            self.task_envs[name]       = SymbolicRegressionEnv(
                X, y, self.grammar
            )
            self.task_best[name] = {
                "reward": float("-inf"),
                "episode": None,
                "nmse": 1.0,
            }

        # Training history
        self.history = defaultdict(list)

    def _sample_task(self, episode_idx: int) -> Dict:
        """Sample a task for the current batch."""
        if self.task_sampling == "prioritized":
            # Sample harder tasks (higher NMSE) more often
            nmses  = np.array([self.task_best[t["name"]]["nmse"]
                               for t in self.tasks])
            probs  = nmses / nmses.sum() if nmses.sum() > 0 else None
            idx    = np.random.choice(len(self.tasks), p=probs)
        else:
            idx = episode_idx // self.batch_size % len(self.tasks)
        return self.tasks[idx]

    def train(self) -> Dict:
        episode_idx = 0
        update_idx  = 0

        while episode_idx < self.num_episodes:
            current_batch_size = min(
                self.batch_size, self.num_episodes - episode_idx
            )

            # Sample task for this batch
            task     = self._sample_task(episode_idx)
            name     = task["name"]
            X, y     = task["X"], task["y"]
            env      = self.task_envs[name]
            evaluator = self.task_evaluators[name]
            memory   = self.task_memories[name]

            # Update dataset embedding for current task
            tensor_X = torch.tensor(X, dtype=torch.float32, device=self.device)
            tensor_y = torch.tensor(y, dtype=torch.float32, device=self.device)
            self.policy.set_dataset_embedding(tensor_X, tensor_y)

            # Curriculum
            if self.use_curriculum:
                current_max_length = curriculum_max_length(
                    episode_idx=episode_idx,
                    num_episodes=self.num_episodes,
                    max_length=self.max_length,
                )
            else:
                current_max_length = self.max_length

            # Collect batch
            batch_episodes = collect_batched_episodes(
                env_template=env,
                policy=self.policy,
                grammar=self.grammar,
                batch_size=current_batch_size,
                max_length=current_max_length,
                device=self.device,
            )

            # Evaluate and store rewards
            for episode in batch_episodes:
                eval_result = evaluator.evaluate(
                    tokens=episode["tokens"], X=X, y=y
                )
                reward = (
                    -eval_result["nmse"] - 0.01 * len(episode["tokens"])
                    if eval_result["is_valid"] else -1.0
                )

                L = len(episode["tokens"])
                episode["rewards"] = [0.0] * L
                if L > 0:
                    episode["rewards"][-1] = reward
                episode["final_reward"] = reward

                infix = safe_prefix_to_infix(
                    episode["tokens"], self.grammar,
                    eval_result.get("optimized_constants", []),
                )
                memory.add(
                    tokens=episode["tokens"],
                    infix=infix,
                    reward=reward,
                    nmse=eval_result.get("nmse", 1.0),
                    complexity=L,
                    source="sampling",
                )

                if reward > self.task_best[name]["reward"]:
                    self.task_best[name]["reward"]  = reward
                    self.task_best[name]["episode"] = episode
                    self.task_best[name]["nmse"]    = eval_result.get("nmse", 1.0)
                    print(
                        f"  [ep={episode_idx}] {name}: new best "
                        f"NMSE={eval_result.get('nmse',1.0):.4f}  "
                        f"{episode['tokens'][:6]}"
                    )

                episode_idx += 1

            # Memory replay (task-specific)
            memory_episodes = []
            for item in memory.to_rows():
                try:
                    ep = recompute_episode(
                        env=env,
                        policy=self.policy,
                        grammar=self.grammar,
                        tokens=item["tokens"],
                        device=self.device,
                    )
                    memory_episodes.append(ep)
                except Exception:
                    pass

            # RSPG update
            stats = self.rl_optimizer.update(
                batch_episodes, memory_episodes=memory_episodes
            )

            self.history["loss"].append(stats["loss"])
            self.history["entropy"].append(stats["entropy"])
            self.history["final_reward"].append(stats["final_reward"])
            self.history["task"].append(name)

            if update_idx % 10 == 0:
                best_per_task = "  ".join(
                    f"{t['name'].replace('feynman_','')[:10]}="
                    f"{self.task_best[t['name']]['nmse']:.3f}"
                    for t in self.tasks
                )
                print(
                    f"[Update {update_idx:4d} | Ep {episode_idx:5d}/"
                    f"{self.num_episodes}]  task={name[:20]:<20}  "
                    f"reward={stats['final_reward']:.4f}  "
                    f"H={stats['entropy']:.3f}"
                )
                print(f"  Best NMSE: {best_per_task}")

            update_idx += 1

        return {
            "history":   dict(self.history),
            "task_best": self.task_best,
        }

    def evaluate_all(self) -> Dict[str, Dict]:
        """Evaluate the shared policy on all tasks after training."""
        results = {}
        self.policy.eval()

        for task in self.tasks:
            name = task["name"]
            X, y = task["X"], task["y"]

            tensor_X = torch.tensor(X, dtype=torch.float32, device=self.device)
            tensor_y = torch.tensor(y, dtype=torch.float32, device=self.device)
            self.policy.set_dataset_embedding(tensor_X, tensor_y)

            best_ep   = self.task_best[name]["episode"]
            evaluator = self.task_evaluators[name]

            nmse, expr = 1.0, ""
            if best_ep is not None:
                er   = evaluator.evaluate(best_ep["tokens"], X, y)
                nmse = er.get("nmse", 1.0)
                expr = safe_prefix_to_infix(
                    best_ep["tokens"], self.grammar,
                    er.get("optimized_constants", []),
                )

            gt = FEYNMAN_GROUND_TRUTH.get(name, {})
            results[name] = {
                "nmse":       nmse,
                "expr":       expr,
                "quality":    classify_quality(nmse),
                "difficulty": gt.get("difficulty", "Unknown"),
                "true_expr":  gt.get("expr", "unknown"),
            }

        return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(multitask_results, single_results, out_dir):
    task_names = sorted(multitask_results.keys())
    x          = np.arange(len(task_names))
    width      = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(task_names) * 2.5), 5))

    mt_nmses = [multitask_results[t]["nmse"]  for t in task_names]
    st_nmses = [single_results.get(t, {}).get("nmse", float("nan"))
                for t in task_names]

    bars1 = ax.bar(x - width/2, mt_nmses, width,
                   color="#D85A30", alpha=0.85, label="Multi-task (shared policy)")
    bars2 = ax.bar(x + width/2, st_nmses, width,
                   color="#378ADD", alpha=0.85, label="Single-task (per-task)")

    for bars, vals in [(bars1, mt_nmses), (bars2, st_nmses)]:
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("feynman_", "") for t in task_names],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Best NMSE (lower = better)", fontsize=10)
    ax.set_title("Multi-task vs single-task training", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(out_dir, "multitask_vs_single_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "difficulty", "true_expression",
    "multitask_nmse", "multitask_expr", "multitask_quality",
    "single_nmse",    "single_expr",    "single_quality",
    "improvement", "winner",
]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-task training: one shared policy across all tasks."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["pmlb_feynman_subset", "pmlb_feynman_all"])

    parser.add_argument("--num_episodes",   type=int,   default=5000,
                        help="Total episodes across ALL tasks")
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4)
    parser.add_argument("--entropy_weight", type=float, default=0.017)
    parser.add_argument("--task_sampling",  type=str,   default="uniform",
                        choices=["uniform", "prioritized"])
    parser.add_argument("--use_curriculum", action="store_true")
    parser.add_argument("--compare_single", action="store_true",
                        help="Also run single-task baselines for comparison. "
                             "Uses the same total episode budget per task.")
    parser.add_argument("--single_csv", type=str, default=None,
                        help="Pre-existing single-task CSV to avoid re-training")
    parser.add_argument("--seed", type=int, default=42)

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  task_sampling: {args.task_sampling}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.tasks:
        task_objs = [get_task_by_name(n, num_samples=args.num_samples)
                     for n in args.tasks]
    else:
        task_objs = get_task_suite(name=args.suite,
                                   num_samples=args.num_samples)

    # Prepare task dicts — policy.py handles padding internally
    max_vars = max(t.num_variables for t in task_objs)
    tasks = []
    for task in task_objs:
        X, y = task.generate()
        tasks.append({
            "name":          task.name,
            "X":             X,
            "y":             y,
            "num_variables": task.num_variables,
        })

    print(f"\nTasks ({len(tasks)}):")
    for t in tasks:
        gt = FEYNMAN_GROUND_TRUTH.get(t["name"], {})
        print(f"  {t['name']:<30}  vars={t['num_variables']}  "
              f"true: {gt.get('expr','?')[:40]}")

    # --- Multi-task training ---
    print(f"\n{'='*60}")
    print(f"MULTI-TASK TRAINING  ({args.num_episodes} total episodes)")
    print(f"{'='*60}")

    mt_trainer = MultiTaskTrainer(
        tasks=tasks,
        device=device,
        num_episodes=args.num_episodes,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        use_curriculum=args.use_curriculum,
        task_sampling=args.task_sampling,
    )
    mt_trainer.train()
    mt_results = mt_trainer.evaluate_all()

    # --- Single-task baselines ---
    single_results: Dict[str, Dict] = {}

    if args.single_csv:
        with open(args.single_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                single_results[row["task"]] = {
                    "nmse":    float(row.get("nmse", 1.0)),
                    "expr":    row.get("recovered_expression", ""),
                    "quality": row.get("quality", "Poor"),
                }
        print(f"\nLoaded single-task results from {args.single_csv}")

    elif args.compare_single:
        # Episodes per task = total / n_tasks (fair budget comparison)
        episodes_per_task = args.num_episodes // len(tasks)
        print(f"\n{'='*60}")
        print(f"SINGLE-TASK BASELINES  ({episodes_per_task} episodes/task)")
        print(f"{'='*60}")

        for task in tasks:
            name = task["name"]
            print(f"\n  {name}")
            trainer = Trainer(
                X=task["X"], y=task["y"],
                num_variables=task["num_variables"],
                device=device,
                optimizer_name="rspg",
            )
            trainer.num_episodes   = episodes_per_task
            trainer.batch_size     = args.batch_size
            trainer.rl_optimizer.learning_rate  = args.learning_rate
            trainer.rl_optimizer.entropy_weight = args.entropy_weight
            trainer.rl_optimizer.optimizer = torch.optim.Adam(
                trainer.policy.parameters(), lr=args.learning_rate
            )
            results   = trainer.train()
            best_ep   = results["best_episode"]
            evaluator = PrefixEvaluator(trainer.grammar)

            nmse, expr = 1.0, ""
            if best_ep is not None:
                er   = evaluator.evaluate(best_ep["tokens"], task["X"], task["y"])
                nmse = er.get("nmse", 1.0)
                expr = safe_prefix_to_infix(
                    best_ep["tokens"], trainer.grammar,
                    er.get("optimized_constants", []),
                )
            single_results[name] = {
                "nmse":    nmse,
                "expr":    expr,
                "quality": classify_quality(nmse),
            }

    # --- Results ---
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, f"multitask_{timestamp}.csv")

    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    w = max(len(t["name"]) for t in tasks) + 2
    print(f"{'Task':<{w}} {'Multi-task':>12}  {'Single':>10}  {'Δ':>8}  Winner")
    print("-" * 70)

    rows = []
    for task in tasks:
        name = task["name"]
        gt   = FEYNMAN_GROUND_TRUTH.get(name, {})
        mt_r = mt_results[name]
        st_r = single_results.get(name, {"nmse": float("nan"),
                                          "expr": "", "quality": "N/A"})

        improvement = (st_r["nmse"] - mt_r["nmse"]
                       if not np.isnan(st_r["nmse"]) else float("nan"))
        winner      = ("multitask" if improvement > 0
                       else "single" if improvement < 0
                       else "tie")
        marker      = "MT ▲" if improvement > 0 else ("ST ▲" if improvement < 0 else "=")

        print(f"{name:<{w}} {mt_r['nmse']:>12.4f}  "
              f"{st_r['nmse']:>10.4f}  "
              f"{improvement:>+8.4f}  {marker}")

        row = {
            "task":             name,
            "difficulty":       gt.get("difficulty", "Unknown"),
            "true_expression":  gt.get("expr", "unknown"),
            "multitask_nmse":   mt_r["nmse"],
            "multitask_expr":   mt_r["expr"],
            "multitask_quality": mt_r["quality"],
            "single_nmse":      st_r["nmse"],
            "single_expr":      st_r["expr"],
            "single_quality":   st_r["quality"],
            "improvement":      improvement,
            "winner":           winner,
        }
        rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    mt_nmses = [r["multitask_nmse"] for r in rows]
    st_nmses = [r["single_nmse"] for r in rows if not np.isnan(r["single_nmse"])]
    print(f"\n  Multi-task mean NMSE: {np.mean(mt_nmses):.4f}")
    if st_nmses:
        print(f"  Single-task mean NMSE: {np.mean(st_nmses):.4f}")
        print(f"  Δ mean: {np.mean(st_nmses)-np.mean(mt_nmses):+.4f}")

    if single_results:
        plot_comparison(mt_results, single_results, plots_dir)

    print(f"\nCSV → {csv_path}")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()