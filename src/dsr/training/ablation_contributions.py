"""
ablation_contributions.py
=========================
Ablation study for the three original contributions:
  1. Curriculum learning on expression complexity (max_length progressive ramp)
  2. Diverse Top-K memory (edit-distance diversity enforcement)
  3. Prioritized Experience Replay (surprise-weighted memory replay)

Five variants tested on each Feynman task, all with RSPG + BFGS + DeepSets:

  full                  — RSPG + curriculum + diverse memory + prioritized replay
  no_curriculum         — without curriculum (flat max_length=30)
  no_diverse_memory     — without diversity filter (standard TopKMemory)
  no_prioritized_replay — without prioritized replay (uniform memory replay)
  baseline              — RSPG only, no contributions (original system)

All variants use the same seed, episodes, and hyperparameters.
This isolates the contribution of each new component.

Usage
-----
# Quick test — 2 tasks, 1000 episodes, 1 seed (~20 min GPU):
python -m dsr.training.ablation_contributions \
    --tasks feynman_I_10_7 feynman_I_12_11 \
    --num_episodes 1000 --num_seeds 1

# Recommended for paper — subset, 3000 episodes, 3 seeds:
python -m dsr.training.ablation_contributions \
    --suite pmlb_feynman_subset \
    --num_episodes 3000 --num_seeds 3

Output
------
results/
  ablation_contributions_<timestamp>.csv
  plots/
    contrib_barplot.png
    contrib_convergence_<task>.png
    contrib_nmse_heatmap.png
"""

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..data.datasets import get_task_suite, get_task_by_name
from .trainer import Trainer
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix


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
    "full":                  "Curriculum + Diverse + PER  (full)",
    "no_curriculum":         "− Curriculum",
    "no_diverse_memory":     "− Diverse memory",
    "no_prioritized_replay": "− Prioritized replay",
    "baseline":              "Baseline  (original system)",
}

VARIANT_STYLES = {
    "full":                  {"color": "#D85A30", "label": "Full (all 3 contributions)", "lw": 2.2, "zorder": 5},
    "no_curriculum":         {"color": "#378ADD", "label": "− Curriculum",               "lw": 1.5, "zorder": 4},
    "no_diverse_memory":     {"color": "#639922", "label": "− Diverse memory",           "lw": 1.5, "zorder": 3},
    "no_prioritized_replay": {"color": "#EF9F27", "label": "− Prioritized replay",       "lw": 1.5, "zorder": 2},
    "baseline":              {"color": "#7F77DD", "label": "Baseline (original)",        "lw": 1.5, "zorder": 1},
}

# Maps variant name → Trainer kwargs
VARIANT_FLAGS = {
    "full": {
        "use_curriculum":         True,
        "use_diverse_memory":     False,   # Prioritized replaces diverse in full system
        "use_prioritized_memory": True,
    },
    "no_curriculum": {
        "use_curriculum":         False,
        "use_diverse_memory":     False,
        "use_prioritized_memory": True,
    },
    "no_diverse_memory": {
        "use_curriculum":         True,
        "use_diverse_memory":     False,
        "use_prioritized_memory": True,
    },
    "no_prioritized_replay": {
        "use_curriculum":         True,
        "use_diverse_memory":     True,
        "use_prioritized_memory": False,
    },
    "baseline": {
        "use_curriculum":         False,
        "use_diverse_memory":     False,
        "use_prioritized_memory": False,
    },
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

    flags = VARIANT_FLAGS[variant]

    trainer = Trainer(
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
        eval_result = evaluator.evaluate(best_ep["tokens"], X, y)
        best_nmse   = eval_result.get("nmse", 1.0)
        best_expr   = safe_prefix_to_infix(
            best_ep["tokens"], trainer.grammar,
            eval_result.get("optimized_constants", []),
        )

    # Diversity stats — only available for DiverseTopKMemory
    diversity = {}
    if hasattr(trainer.memory, "diversity_stats"):
        diversity = trainer.memory.diversity_stats()

    return {
        "variant":                   variant,
        "seed":                      seed,
        "best_reward":               results["best_reward"],
        "best_nmse":                 best_nmse,
        "best_expr":                 best_expr,
        "history_reward":            results["history"]["final_reward"],
        "history_entropy":           results["history"]["entropy"],
        "history_max_length":        results["history"].get("max_length", []),
        "memory_mean_pairwise_dist": diversity.get("mean_pairwise_distance", 0.0),
        "memory_min_pairwise_dist":  diversity.get("min_pairwise_distance", 0),
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
    """Reward curve + curriculum schedule, 4 variants overlaid."""
    has_curriculum = any(len(r["history_max_length"]) > 0 for r in task_runs)
    ncols = 3 if has_curriculum else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4))
    fig.suptitle(f"Contributions ablation — {task_name}", fontsize=12, fontweight="bold")

    for variant, style in VARIANT_STYLES.items():
        runs = [r for r in task_runs if r["variant"] == variant]
        if not runs:
            continue

        for key, ax in zip(
            ["history_reward", "history_entropy"],
            [axes[0], axes[1]],
        ):
            if all(len(r[key]) == 0 for r in runs):
                continue
            min_len = min(len(r[key]) for r in runs)
            curves  = np.array([np.array(r[key][:min_len], dtype=float) for r in runs])
            mean_c  = curves.mean(axis=0)
            std_c   = curves.std(axis=0)
            x       = np.arange(min_len)

            ax.plot(x, smooth(mean_c),
                    color=style["color"], label=style["label"],
                    lw=style["lw"], zorder=style["zorder"])
            ax.fill_between(
                x, smooth(mean_c - std_c), smooth(mean_c + std_c),
                color=style["color"], alpha=0.12, zorder=0,
            )

        # Curriculum ramp (only for variants that use it)
        if has_curriculum and len(runs[0]["history_max_length"]) > 0:
            ml = runs[0]["history_max_length"]
            x  = np.arange(len(ml))
            axes[2].plot(x, ml,
                         color=style["color"], label=style["label"],
                         lw=style["lw"], zorder=style["zorder"])

    labels_and_titles = [
        ("Batch reward (mean)", axes[0]),
        ("Entropy",             axes[1]),
    ]
    if has_curriculum:
        labels_and_titles.append(("Max expression length", axes[2]))

    for ylabel, ax in labels_and_titles:
        ax.set_xlabel("Batch update", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.autoscale(axis="y", tight=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"contrib_convergence_{task_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_barplot(summary: List[Dict], out_dir: str):
    """Mean best reward ± std per variant, aggregated."""
    variants = list(VARIANTS.keys())
    means, stds = [], []
    for v in variants:
        rewards = [r["best_reward"] for r in summary if r["variant"] == v]
        means.append(np.mean(rewards) if rewards else 0.0)
        stds.append(np.std(rewards)   if rewards else 0.0)

    fig, ax = plt.subplots(figsize=(9, 4))
    x      = np.arange(len(variants))
    colors = [VARIANT_STYLES[v]["color"] for v in variants]
    labels = [VARIANT_STYLES[v]["label"] for v in variants]

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.55,
                  error_kw={"lw": 1.2, "capthick": 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Best reward (mean ± std)")
    ax.set_title("Contribution ablation — aggregated", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "contrib_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_nmse_heatmap(summary: List[Dict], task_names: List[str], out_dir: str):
    variants = list(VARIANTS.keys())
    labels   = [VARIANT_STYLES[v]["label"] for v in variants]

    data = np.ones((len(task_names), len(variants)))
    for i, task in enumerate(task_names):
        for j, v in enumerate(variants):
            runs = [r for r in summary if r["task"] == task and r["variant"] == v]
            if runs:
                data[i, j] = np.mean([r["best_nmse"] for r in runs])

    fig_h = max(4, len(task_names) * 0.45 + 1)
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 2.5), fig_h))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels([t.replace("feynman_", "") for t in task_names], fontsize=7)
    ax.set_title("NMSE per task & variant  (green = better)", fontsize=10)

    for i in range(len(task_names)):
        for j in range(len(variants)):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.7 else "black")

    plt.colorbar(im, ax=ax, shrink=0.6, label="NMSE")
    plt.tight_layout()
    path = os.path.join(out_dir, "contrib_nmse_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(summary: List[Dict], task_names: List[str]):
    variants = list(VARIANTS.keys())

    print("\n" + "=" * 85)
    print("CONTRIBUTION ABLATION — mean best reward across seeds")
    print("=" * 85)
    header = f"{'Task':<25}" + "".join(
        f"{VARIANT_STYLES[v]['label'][:18]:>20}" for v in variants
    )
    print(header)
    print("-" * 85)
    for task in task_names:
        row = f"{task.replace('feynman_', ''):<25}"
        for v in variants:
            runs   = [r for r in summary if r["task"] == task and r["variant"] == v]
            mean_r = np.mean([r["best_reward"] for r in runs]) if runs else float("nan")
            row   += f"{mean_r:>20.4f}"
        print(row)

    print("=" * 85)
    print("DELTA vs baseline")
    print("-" * 85)
    base_rewards = [r["best_reward"] for r in summary if r["variant"] == "baseline"]
    base_mean    = np.mean(base_rewards) if base_rewards else 0.0
    print(f"  {'Baseline':<30}  {base_mean:+.4f}  (reference)")

    for v in ["no_diverse_memory", "no_curriculum", "no_prioritized_replay", "full"]:
        runs   = [r["best_reward"] for r in summary if r["variant"] == v]
        v_mean = np.mean(runs) if runs else 0.0
        delta  = v_mean - base_mean
        marker = "▲" if delta > 0 else "▼"
        label  = VARIANT_STYLES[v]["label"]
        print(f"  {label:<30}  {v_mean:+.4f}  (Δ {delta:+.4f} {marker})")

    print()
    # Diversity stats for variants with DiverseTopKMemory
    print("MEMORY DIVERSITY STATS (mean pairwise edit distance)")
    print("-" * 85)
    for v in ["no_prioritized_replay"]:
        runs  = [r for r in summary if r["variant"] == v]
        dists = [r["memory_mean_pairwise_dist"] for r in runs
                 if r["memory_mean_pairwise_dist"] > 0]
        if dists:
            label = VARIANT_STYLES[v]["label"]
            print(f"  {label:<40}  mean dist = {np.mean(dists):.2f} tokens")
    for v in ["full", "no_curriculum", "no_diverse_memory", "baseline"]:
        label = VARIANT_STYLES[v]["label"]
        print(f"  {label:<40}  (PrioritizedTopKMemory or standard TopKMemory)")
    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "variant", "seed",
    "best_reward", "best_nmse", "best_expr",
    "memory_mean_pairwise_dist", "memory_min_pairwise_dist",
]


def run_ablation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples) for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    task_names = [t.name for t in tasks]
    variants   = list(VARIANTS.keys())
    seeds      = list(range(args.num_seeds))
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, f"ablation_contributions_{timestamp}.csv")
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
                print(f"\n[{done}/{total}]  task={task.name}  variant={variant}  seed={seed}")
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
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                print(f"  best_reward={run['best_reward']:.4f}  "
                      f"best_nmse={run['best_nmse']:.4f}  "
                      f"mem_dist={run['memory_mean_pairwise_dist']:.2f}  "
                      f"expr={run['best_expr'][:50]}")

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
        description="Ablation study for curriculum learning + diverse memory contributions."
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