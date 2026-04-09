"""
ablation_study.py
=================
Ablation study for the memory-augmented RSPG symbolic regression system.

Four variants tested on each Feynman task:

  full          — RSPG + Top-K memory replay + BFGS constant opt. + DeepSets
  no_memory     — RSPG without Top-K memory replay (memory_episodes=[])
  no_bfgs       — RSPG without BFGS constant optimisation (optimize_constants=False)
  no_deepsets   — RSPG with dataset embedding replaced by a fixed zero vector

Each variant uses the same seed, same number of episodes, same hyperparameters.
This isolates the contribution of each individual component.

Usage
-----
# Quick test — 3 tasks, 1000 episodes, 1 seed (~30 min CPU):
python -m dsr.training.ablation_study \
    --tasks feynman_I_8_14 feynman_I_10_7 feynman_I_12_1 \
    --num_episodes 1000 --num_seeds 1

# Recommended for paper — subset, 3000 episodes, 3 seeds (~overnight):
python -m dsr.training.ablation_study \
    --suite pmlb_feynman_subset \
    --num_episodes 3000 --num_seeds 3

Output
------
results/
  ablation_<timestamp>.csv          <- per-task per-variant metrics
  plots/
    ablation_barplot.png            <- mean best reward per variant (bar chart)
    ablation_convergence_<task>.png <- reward curves, 4 variants overlaid
    ablation_nmse_heatmap.png       <- NMSE grid: tasks × variants
"""

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ..data.datasets import get_task_suite, get_task_by_name
from .trainer import Trainer
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..analysis.memory import TopKMemory
from .rollout import recompute_episode, collect_batched_episodes


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANTS = {
    "full":        "RSPG + Memory + BFGS + DeepSets  (full system)",
    "no_memory":   "RSPG − Memory replay",
    "no_bfgs":     "RSPG − BFGS constant optimisation",
    "no_deepsets": "RSPG − DeepSets encoder  (zero embedding)",
}

VARIANT_STYLES = {
    "full":        {"color": "#D85A30", "label": "Full system", "lw": 2.2, "zorder": 4},
    "no_memory":   {"color": "#378ADD", "label": "− Memory",    "lw": 1.5, "zorder": 3},
    "no_bfgs":     {"color": "#639922", "label": "− BFGS",      "lw": 1.5, "zorder": 2},
    "no_deepsets": {"color": "#7F77DD", "label": "− DeepSets",  "lw": 1.5, "zorder": 1},
}


# ---------------------------------------------------------------------------
# Patched Trainer subclass for each ablation variant
# ---------------------------------------------------------------------------

class AblationTrainer(Trainer):
    """
    Extends Trainer with ablation flags.

    use_memory   : if False, skip Top-K replay injection entirely
    use_bfgs     : if False, call evaluator with optimize_constants=False
    use_deepsets : if False, replace dataset embedding with a zero vector
    """

    def __init__(self, *args,
                 use_memory: bool = True,
                 use_bfgs: bool = True,
                 use_deepsets: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_memory    = use_memory
        self.use_bfgs      = use_bfgs
        self.use_deepsets  = use_deepsets

        # No-DeepSets: replace the cached dataset embedding with zeros
        # The embedding is computed once in __init__ via set_dataset_embedding.
        # We simply overwrite it with a zero tensor of the same shape.
        if not use_deepsets:
            with torch.no_grad():
                zero_emb = torch.zeros(
                    self.policy.dataset_embedding_dim,
                    device=self.device
                )
                self.policy.cached_dataset_embedding = zero_emb

    def train(self):
        episode_idx = 0
        while episode_idx < self.num_episodes:
            current_batch_size = min(self.batch_size, self.num_episodes - episode_idx)

            batch_episodes = collect_batched_episodes(
                env_template=self.env,
                policy=self.policy,
                grammar=self.grammar,
                batch_size=current_batch_size,
                max_length=30,
                device=self.device,
            )

            for episode in batch_episodes:
                # BFGS ablation: disable constant optimisation
                eval_result = self.evaluator.evaluate(
                    tokens=episode["tokens"],
                    X=self.X,
                    y=self.y,
                    optimize_constants=self.use_bfgs,
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

                self.memory.add(
                    tokens=episode["tokens"],
                    infix=infix,
                    reward=reward,
                    nmse=eval_result["nmse"],
                    complexity=L,
                    source="sampling",
                )

                if reward > self.best_reward:
                    self.best_reward  = reward
                    self.best_episode = episode

                episode_idx += 1

            # Memory ablation: skip replay injection if use_memory=False
            memory_episodes = []
            if self.use_memory:
                for item in self.memory.to_rows():
                    try:
                        ep = recompute_episode(
                            env=self.env,
                            policy=self.policy,
                            grammar=self.grammar,
                            tokens=item["tokens"],
                            device=self.device,
                        )
                        memory_episodes.append(ep)
                    except Exception:
                        pass

            stats = self.rl_optimizer.update(
                batch_episodes,
                memory_episodes=memory_episodes,   # empty list = no replay
            )

            self.history["loss"].append(stats["loss"])
            self.history["policy_loss"].append(stats["policy_loss"])
            self.history["value_loss"].append(stats.get("value_loss", 0.0))
            self.history["entropy"].append(stats["entropy"])
            self.history["final_reward"].append(stats["final_reward"])

        return {
            "history":      self.history,
            "best_reward":  self.best_reward,
            "best_episode": self.best_episode,
            "memory":       self.memory,
        }


# ---------------------------------------------------------------------------
# Single variant run
# ---------------------------------------------------------------------------

def run_variant(
    X: np.ndarray,
    y: np.ndarray,
    num_variables: int,
    variant: str,
    num_episodes: int,
    batch_size: int,
    learning_rate: float,
    entropy_weight: float,
    device: str,
    seed: int,
) -> Dict:
    set_seed(seed)

    flags = {
        "full":        dict(use_memory=True,  use_bfgs=True,  use_deepsets=True),
        "no_memory":   dict(use_memory=False, use_bfgs=True,  use_deepsets=True),
        "no_bfgs":     dict(use_memory=True,  use_bfgs=False, use_deepsets=True),
        "no_deepsets": dict(use_memory=True,  use_bfgs=True,  use_deepsets=False),
    }[variant]

    trainer = AblationTrainer(
        X=X, y=y,
        num_variables=num_variables,
        device=device,
        optimizer_name="rspg",
        **flags,
    )
    trainer.num_episodes   = num_episodes
    trainer.batch_size     = batch_size
    trainer.rl_optimizer.learning_rate  = learning_rate
    trainer.rl_optimizer.entropy_weight = entropy_weight
    trainer.rl_optimizer.optimizer = torch.optim.Adam(
        trainer.policy.parameters(), lr=learning_rate
    )

    results   = trainer.train()
    best_ep   = results["best_episode"]
    evaluator = PrefixEvaluator(trainer.grammar)

    best_nmse, best_expr = 1.0, ""
    if best_ep is not None:
        eval_result = evaluator.evaluate(
            best_ep["tokens"], X, y,
            optimize_constants=flags["use_bfgs"],
        )
        best_nmse = eval_result.get("nmse", 1.0)
        best_expr = safe_prefix_to_infix(
            best_ep["tokens"], trainer.grammar,
            eval_result.get("optimized_constants", []),
        )

    return {
        "variant":         variant,
        "seed":            seed,
        "best_reward":     results["best_reward"],
        "best_nmse":       best_nmse,
        "best_expr":       best_expr,
        "history_reward":  results["history"]["final_reward"],
        "history_entropy": results["history"]["entropy"],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def smooth(values: List[float], window: int = 10) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_convergence(task_name: str, task_runs: List[Dict], out_dir: str):
    """Reward + entropy curves, 4 variants overlaid, independent y-axes."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Ablation — Task: {task_name}", fontsize=13, fontweight="bold")

    metrics = [
        ("history_reward",  "Batch reward (mean)", axes[0]),
        ("history_entropy", "Entropy",             axes[1]),
    ]

    for key, ylabel, ax in metrics:
        for variant, style in VARIANT_STYLES.items():
            runs = [r for r in task_runs if r["variant"] == variant]
            if not runs or all(len(r[key]) == 0 for r in runs):
                continue

            min_len = min(len(r[key]) for r in runs)
            curves  = np.array([np.array(r[key][:min_len], dtype=float)
                                 for r in runs])
            mean_c  = curves.mean(axis=0)
            std_c   = curves.std(axis=0)
            x       = np.arange(min_len)

            ax.plot(x, smooth(mean_c),
                    color=style["color"], label=style["label"],
                    lw=style["lw"], zorder=style["zorder"])
            ax.fill_between(
                x,
                smooth(mean_c - std_c),
                smooth(mean_c + std_c),
                color=style["color"], alpha=0.12, zorder=0,
            )

        ax.set_xlabel("Batch update", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.autoscale(axis="y", tight=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ablation_convergence_{task_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_barplot(summary: List[Dict], out_dir: str):
    """Mean best reward ± std per variant, aggregated across all tasks and seeds."""
    variants = list(VARIANTS.keys())
    means, stds = [], []

    for v in variants:
        rewards = [r["best_reward"] for r in summary if r["variant"] == v]
        means.append(np.mean(rewards) if rewards else 0.0)
        stds.append(np.std(rewards)   if rewards else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    x      = np.arange(len(variants))
    colors = [VARIANT_STYLES[v]["color"] for v in variants]
    labels = [VARIANT_STYLES[v]["label"] for v in variants]

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.55,
                  error_kw={"lw": 1.2, "capthick": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Best reward (mean ± std across tasks & seeds)")
    ax.set_title("Ablation study — component contribution", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_nmse_heatmap(summary: List[Dict], task_names: List[str], out_dir: str):
    """NMSE heatmap: tasks (rows) × variants (cols). Green = better."""
    variants = list(VARIANTS.keys())
    labels   = [VARIANT_STYLES[v]["label"] for v in variants]

    data = np.ones((len(task_names), len(variants)))
    for i, task in enumerate(task_names):
        for j, v in enumerate(variants):
            runs = [r for r in summary
                    if r["task"] == task and r["variant"] == v]
            if runs:
                data[i, j] = np.mean([r["best_nmse"] for r in runs])

    fig_h = max(4, len(task_names) * 0.45 + 1)
    fig, ax = plt.subplots(figsize=(max(6, len(variants) * 2.5), fig_h))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels(
        [t.replace("feynman_", "") for t in task_names], fontsize=7
    )
    ax.set_title("Ablation NMSE per task & variant  (green = better)", fontsize=10)

    for i in range(len(task_names)):
        for j in range(len(variants)):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if val > 0.7 else "black")

    plt.colorbar(im, ax=ax, shrink=0.6, label="NMSE")
    plt.tight_layout()
    path = os.path.join(out_dir, "ablation_nmse_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(summary: List[Dict], task_names: List[str]):
    variants = list(VARIANTS.keys())

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY — mean best reward across seeds")
    print("=" * 80)
    header = f"{'Task':<28}" + "".join(
        f"{VARIANT_STYLES[v]['label']:>14}" for v in variants
    )
    print(header)
    print("-" * 80)

    for task in task_names:
        row = f"{task.replace('feynman_', ''):<28}"
        for v in variants:
            runs = [r for r in summary
                    if r["task"] == task and r["variant"] == v]
            mean_r = np.mean([r["best_reward"] for r in runs]) if runs else float("nan")
            row += f"{mean_r:>14.4f}"
        print(row)

    print("=" * 80)
    print("DELTA vs full system (mean best reward)")
    print("-" * 80)
    full_rewards = [r["best_reward"] for r in summary if r["variant"] == "full"]
    full_mean    = np.mean(full_rewards) if full_rewards else 0.0
    print(f"  {'full':<20}  {full_mean:+.4f}  (baseline)")

    for v in ["no_memory", "no_bfgs", "no_deepsets"]:
        runs    = [r["best_reward"] for r in summary if r["variant"] == v]
        v_mean  = np.mean(runs) if runs else 0.0
        delta   = v_mean - full_mean
        label   = VARIANT_STYLES[v]["label"]
        marker  = "▼" if delta < 0 else "▲"
        print(f"  {label:<20}  {v_mean:+.4f}  (Δ {delta:+.4f} {marker})")

    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "variant", "seed",
    "best_reward", "best_nmse", "best_expr",
]


def run_ablation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples)
                 for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    task_names = [t.name for t in tasks]
    variants   = list(VARIANTS.keys())
    seeds      = list(range(args.num_seeds))
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, f"ablation_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    summary  = []
    total    = len(tasks) * len(variants) * len(seeds)
    done     = 0

    for task in tasks:
        X, y      = task.generate()
        task_runs = []

        for variant in variants:
            for seed in seeds:
                done += 1
                print(f"\n[{done}/{total}]  task={task.name}  "
                      f"variant={variant}  seed={seed}")
                print(f"  ({VARIANTS[variant]})")

                run = run_variant(
                    X=X, y=y,
                    num_variables=task.num_variables,
                    variant=variant,
                    num_episodes=args.num_episodes,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    entropy_weight=args.entropy_weight,
                    device=device,
                    seed=seed,
                )
                run["task"] = task.name

                task_runs.append(run)
                row = {k: run[k] for k in FIELDNAMES}
                summary.append(row)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writerow(row)

                print(f"  best_reward={run['best_reward']:.4f}  "
                      f"best_nmse={run['best_nmse']:.4f}  "
                      f"expr={run['best_expr'][:55]}")

        plot_convergence(task.name, task_runs, plots_dir)

    plot_barplot(summary, plots_dir)
    plot_nmse_heatmap(summary, task_names, plots_dir)
    print_summary(summary, task_names)

    print(f"CSV   → {csv_path}")
    print(f"Plots → {plots_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Ablation study — contribution of each system component."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["nguyen", "nguyen_univariate", "nguyen_bivariate", "pmlb_feynman_subset", "pmlb_feynman_all"])

    parser.add_argument("--num_episodes",   type=int,   default=2000)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--num_seeds",      type=int,   default=3)
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4)
    parser.add_argument("--entropy_weight", type=float, default=0.017)

    args = parser.parse_args()
    run_ablation(args)


if __name__ == "__main__":
    main()