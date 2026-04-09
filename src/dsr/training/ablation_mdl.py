"""
ablation_mdl.py
===============
Ablation study comparing MDL reward vs the standard linear complexity penalty.

Two variants tested on each Feynman task, all else equal:

  linear  — reward = -NMSE - 0.01 * len(tokens)          (current system)
  mdl     — reward = -(L_model + L_data) / n              (MDL principle)

The MDL reward provides a theoretically grounded complexity measure rooted
in information theory (Rissanen, 1978), replacing the arbitrary alpha=0.01
penalty with a dataset-aware description length criterion.

Hypothesis
----------
MDL should produce expressions with better complexity/fit trade-off:
  - Prefer shorter expressions when NMSE difference is small
  - Accept longer expressions only when they provide substantial fit gain
  - Be more consistent across datasets of different sizes

Usage
-----
# Quick test — 2 tasks, 1 seed:
python -m dsr.training.ablation_mdl \
    --tasks feynman_I_10_7 feynman_I_12_1 feynman_I_12_11 \
    --num_episodes 2000 --num_seeds 1

# Full ablation — subset, 3 seeds:
python -m dsr.training.ablation_mdl \
    --suite pmlb_feynman_subset \
    --num_episodes 5000 --num_seeds 3

Output
------
results/
  ablation_mdl_<timestamp>.csv
  plots/
    mdl_vs_linear_barplot.png
    mdl_vs_linear_complexity.png   <- expression length distribution
    mdl_vs_linear_convergence_<task>.png
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
from ..core.mdl_reward import compare_rewards
from ..data.feynman_ground_truth import FEYNMAN_GROUND_TRUTH, classify_quality


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
    "linear": "Linear penalty  (−NMSE − 0.01·|e|)",
    "mdl":    "MDL reward  (−(L_model + L_data) / n)",
}

VARIANT_STYLES = {
    "linear": {"color": "#378ADD", "label": "Linear (baseline)", "lw": 1.8, "zorder": 1},
    "mdl":    {"color": "#D85A30", "label": "MDL (ours)",        "lw": 2.2, "zorder": 2},
}

VARIANT_FLAGS = {
    "linear": {"use_mdl_reward": False},
    "mdl":    {"use_mdl_reward": True},
}


# ---------------------------------------------------------------------------
# Single variant run
# ---------------------------------------------------------------------------

def run_variant(
    X, y, num_variables, variant,
    num_episodes, batch_size,
    learning_rate, entropy_weight,
    device, seed,
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

    best_nmse, best_expr, best_complexity = 1.0, "", 0
    reward_comparison = {}

    if best_ep is not None:
        eval_result  = evaluator.evaluate(best_ep["tokens"], X, y)
        best_nmse    = eval_result.get("nmse", 1.0)
        best_expr    = safe_prefix_to_infix(
            best_ep["tokens"], trainer.grammar,
            eval_result.get("optimized_constants", []),
        )
        best_complexity = len(best_ep["tokens"])

        # Side-by-side reward comparison for analysis
        var_y = float(np.var(y)) if np.var(y) > 1e-9 else 1.0
        mse   = best_nmse * var_y
        reward_comparison = compare_rewards(
            tokens=best_ep["tokens"],
            nmse=best_nmse,
            mse=mse,
            n=len(y),
            vocab_size=len(trainer.grammar),
        )

    return {
        "variant":          variant,
        "seed":             seed,
        "best_nmse":        best_nmse,
        "best_expr":        best_expr,
        "best_complexity":  best_complexity,
        "quality":          classify_quality(best_nmse),
        "history_reward":   results["history"]["final_reward"],
        "history_entropy":  results["history"]["entropy"],
        "linear_reward":    reward_comparison.get("linear_reward", 0.0),
        "mdl_reward_val":   reward_comparison.get("mdl_reward", 0.0),
        "L_model_bits":     reward_comparison.get("L_model_bits", 0.0),
        "L_data_bits":      reward_comparison.get("L_data_bits", 0.0),
        "L_total_bits":     reward_comparison.get("L_total_bits", 0.0),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def smooth(values, window=10):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_convergence(task_name, task_runs, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"MDL vs Linear — {task_name}", fontsize=12, fontweight="bold")

    for variant, style in VARIANT_STYLES.items():
        runs = [r for r in task_runs if r["variant"] == variant]
        if not runs:
            continue
        for key, ax, label in [
            ("history_reward",  axes[0], "Batch reward"),
            ("history_entropy", axes[1], "Entropy"),
        ]:
            min_len = min(len(r[key]) for r in runs)
            curves  = np.array([np.array(r[key][:min_len], dtype=float) for r in runs])
            mean_c  = curves.mean(axis=0)
            std_c   = curves.std(axis=0)
            x       = np.arange(min_len)
            ax.plot(x, smooth(mean_c), color=style["color"],
                    label=style["label"], lw=style["lw"], zorder=style["zorder"])
            ax.fill_between(x, smooth(mean_c - std_c), smooth(mean_c + std_c),
                            color=style["color"], alpha=0.12)
            ax.set_ylabel(label, fontsize=9)
            ax.set_xlabel("Batch update", fontsize=9)
            ax.autoscale(axis="y")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"mdl_vs_linear_convergence_{task_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_barplot(summary, out_dir):
    """Mean NMSE per variant, per task."""
    task_names = sorted(set(r["task"] for r in summary))
    variants   = list(VARIANTS.keys())
    x          = np.arange(len(task_names))
    width      = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(task_names) * 2.5), 5))

    for ax, metric, ylabel in [
        (axes[0], "best_nmse",       "Best NMSE (lower = better)"),
        (axes[1], "best_complexity", "Best expression length (tokens)"),
    ]:
        for i, (variant, style) in enumerate(VARIANT_STYLES.items()):
            means, stds = [], []
            for task in task_names:
                vals = [r[metric] for r in summary
                        if r["task"] == task and r["variant"] == variant]
                means.append(np.mean(vals) if vals else 0.0)
                stds.append(np.std(vals)   if vals else 0.0)

            bars = ax.bar(x + i * width - width / 2, means,
                          width=width, color=style["color"],
                          alpha=0.85, label=style["label"],
                          yerr=stds, capsize=3,
                          error_kw={"lw": 1.0})
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (max(stds) * 0.05 if stds else 0.01),
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([t.replace("feynman_", "") for t in task_names],
                           fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title("Best NMSE per task", fontsize=10)
    axes[1].set_title("Expression complexity (tokens)", fontsize=10)

    fig.suptitle("MDL vs Linear reward — ablation", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "mdl_vs_linear_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(summary, task_names):
    print("\n" + "=" * 75)
    print("MDL vs LINEAR REWARD — mean metrics across seeds")
    print("=" * 75)

    w = max(len(t) for t in task_names) + 2
    header = f"{'Task':<{w}}" + "".join(
        f"{VARIANT_STYLES[v]['label']:>20}" for v in VARIANTS
    )
    print(f"{header}   (metric: NMSE)")
    print("-" * 75)

    for task in task_names:
        row = f"{task.replace('feynman_',''):<{w}}"
        for v in VARIANTS:
            runs   = [r for r in summary if r["task"] == task and r["variant"] == v]
            mean_n = np.mean([r["best_nmse"] for r in runs]) if runs else float("nan")
            row   += f"{mean_n:>20.4f}"
        print(row)

    print("=" * 75)
    print("COMPLEXITY — mean expression length across seeds")
    print("-" * 75)
    for task in task_names:
        row = f"{task.replace('feynman_',''):<{w}}"
        for v in VARIANTS:
            runs  = [r for r in summary if r["task"] == task and r["variant"] == v]
            mean_c = np.mean([r["best_complexity"] for r in runs]) if runs else float("nan")
            row   += f"{mean_c:>20.1f}"
        print(row)

    print("=" * 75)
    print("DELTA MDL vs Linear (NMSE)")
    print("-" * 75)
    lin_nmses = [r["best_nmse"] for r in summary if r["variant"] == "linear"]
    mdl_nmses = [r["best_nmse"] for r in summary if r["variant"] == "mdl"]
    if lin_nmses and mdl_nmses:
        delta = np.mean(mdl_nmses) - np.mean(lin_nmses)
        marker = "▲ MDL better" if delta < 0 else "▼ Linear better"
        print(f"  Mean NMSE: Linear={np.mean(lin_nmses):.4f}  "
              f"MDL={np.mean(mdl_nmses):.4f}  Δ={delta:+.4f}  {marker}")

    lin_cx = [r["best_complexity"] for r in summary if r["variant"] == "linear"]
    mdl_cx = [r["best_complexity"] for r in summary if r["variant"] == "mdl"]
    if lin_cx and mdl_cx:
        delta_cx = np.mean(mdl_cx) - np.mean(lin_cx)
        marker_cx = "▲ MDL shorter" if delta_cx < 0 else "▼ MDL longer"
        print(f"  Mean length: Linear={np.mean(lin_cx):.1f}  "
              f"MDL={np.mean(mdl_cx):.1f}  Δ={delta_cx:+.1f}  {marker_cx}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "variant", "seed",
    "best_nmse", "best_complexity", "best_expr", "quality",
    "linear_reward", "mdl_reward_val",
    "L_model_bits", "L_data_bits", "L_total_bits",
]


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: MDL reward vs linear complexity penalty."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["pmlb_feynman_subset", "pmlb_feynman_all"])

    parser.add_argument("--num_episodes",   type=int,   default=2000)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--num_seeds",      type=int,   default=3)
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4)
    parser.add_argument("--entropy_weight", type=float, default=0.017)

    args      = parser.parse_args()
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples) for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    task_names  = [t.name for t in tasks]
    seeds       = list(range(args.num_seeds))
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, f"ablation_mdl_{timestamp}.csv")

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()

    summary   = []
    total     = len(tasks) * len(VARIANTS) * len(seeds)
    done      = 0

    for task in tasks:
        X, y      = task.generate()
        task_runs = []

        for variant in VARIANTS:
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
                    csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)

                print(f"  NMSE={run['best_nmse']:.4f}  "
                      f"len={run['best_complexity']}  "
                      f"quality={run['quality']}  "
                      f"expr={run['best_expr'][:55]}")
                if run["L_total_bits"]:
                    print(f"  L_model={run['L_model_bits']:.1f}b  "
                          f"L_data={run['L_data_bits']:.1f}b  "
                          f"L_total={run['L_total_bits']:.1f}b")

        plot_convergence(task.name, task_runs, plots_dir)

    plot_barplot(summary, plots_dir)
    print_summary(summary, task_names)

    print(f"CSV   → {csv_path}")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()
