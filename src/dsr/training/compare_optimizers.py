import argparse
import os
import random
import csv
import json
from copy import deepcopy
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

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
# Single training run
# ---------------------------------------------------------------------------

def run_one(
    X: np.ndarray,
    y: np.ndarray,
    num_variables: int,
    optimizer_name: str,
    num_episodes: int,
    batch_size: int,
    learning_rate: float,
    entropy_weight: float,
    device: str,
    seed: int,
) -> Dict:
    """Train one Trainer instance and return metrics."""
    set_seed(seed)

    trainer = Trainer(
        X=X,
        y=y,
        num_variables=num_variables,
        device=device,
        optimizer_name=optimizer_name,
    )
    trainer.num_episodes = num_episodes
    trainer.batch_size = batch_size
    trainer.learning_rate = learning_rate
    trainer.entropy_weight = entropy_weight

    # Re-instantiate Adam with the correct lr (Trainer already built one,
    # but lr may differ from the config default)
    trainer.rl_optimizer.optimizer = torch.optim.Adam(
        trainer.policy.parameters(), lr=learning_rate
    )

    results = trainer.train()

    # Evaluate best episode properly
    best_expr = ""
    best_nmse = 1.0
    if results["best_episode"] is not None:
        best_tokens = results["best_episode"]["tokens"]
        evaluator = PrefixEvaluator(trainer.grammar)
        eval_result = evaluator.evaluate(best_tokens, X, y)
        best_nmse = eval_result.get("nmse", 1.0)
        best_expr = safe_prefix_to_infix(
            best_tokens, trainer.grammar, eval_result.get("optimized_constants", [])
        )

    return {
        "optimizer": optimizer_name,
        "seed": seed,
        "best_reward": results["best_reward"],
        "best_nmse": best_nmse,
        "best_expr": best_expr,
        # Full curves for plotting (one value per batch update)
        "history_reward": results["history"]["final_reward"],
        "history_loss": results["history"]["loss"],
        "history_entropy": results["history"]["entropy"],
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

OPTIMIZER_STYLES = {
    "rspg":      {"color": "#D85A30", "label": "RSPG (ours)",  "lw": 2.0, "zorder": 3},
    "reinforce": {"color": "#378ADD", "label": "REINFORCE",    "lw": 1.5, "zorder": 2},
    "ppo":       {"color": "#639922", "label": "PPO",          "lw": 1.5, "zorder": 1},
}


def smooth(values: List[float], window: int = 10) -> np.ndarray:
    """Simple moving average for readability."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_convergence(task_name: str, all_runs: List[Dict], out_dir: str):
    """
    One figure per task: reward curve (mean ± std across seeds) for each optimizer.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Task: {task_name}", fontsize=13, fontweight="bold")

    metrics = [
        ("history_reward",  "Batch reward (mean)",   axes[0]),
        ("history_loss",    "Loss",                  axes[1]),
        ("history_entropy", "Entropy",               axes[2]),
    ]

    for key, ylabel, ax in metrics:
        for opt_name, style in OPTIMIZER_STYLES.items():
            runs = [r for r in all_runs if r["optimizer"] == opt_name]
            if not runs:
                continue

            # Align curves to the shortest run length (different seeds may differ by 1)
            min_len = min(len(r[key]) for r in runs)
            curves = np.array([r[key][:min_len] for r in runs])

            x = np.arange(min_len)
            mean = smooth(curves.mean(axis=0))
            std  = curves.std(axis=0)

            ax.plot(x, mean, color=style["color"], label=style["label"],
                    lw=style["lw"], zorder=style["zorder"])
            ax.fill_between(x, smooth(curves.mean(axis=0) - std),
                            smooth(curves.mean(axis=0) + std),
                            color=style["color"], alpha=0.12, zorder=0)

        ax.set_xlabel("Batch update")
        ax.set_ylabel(ylabel)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    axes[0].legend(frameon=False, fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"convergence_{task_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_summary_barplot(summary: List[Dict], out_dir: str):
    """
    Aggregated bar chart: mean best reward ± std per optimizer across all tasks.
    """
    optimizers = list(OPTIMIZER_STYLES.keys())
    means, stds = [], []

    for opt in optimizers:
        rewards = [r["best_reward"] for r in summary if r["optimizer"] == opt]
        means.append(np.mean(rewards) if rewards else 0.0)
        stds.append(np.std(rewards)  if rewards else 0.0)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(optimizers))
    colors = [OPTIMIZER_STYLES[o]["color"] for o in optimizers]
    labels = [OPTIMIZER_STYLES[o]["label"] for o in optimizers]

    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, alpha=0.85, width=0.5,
                  error_kw={"lw": 1.2, "capthick": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Best reward (mean ± std across tasks & seeds)")
    ax.set_title("Optimizer comparison — aggregated", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    # Annotate bars with the mean value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, "summary_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_nmse_heatmap(summary: List[Dict], task_names: List[str], out_dir: str):
    """
    Heatmap: tasks (rows) × optimizers (cols), cell = best NMSE.
    Gives a quick visual of which optimizer wins on which task.
    """
    optimizers = list(OPTIMIZER_STYLES.keys())
    labels     = [OPTIMIZER_STYLES[o]["label"] for o in optimizers]

    # Average across seeds per (task, optimizer)
    data = np.ones((len(task_names), len(optimizers)))
    for i, task in enumerate(task_names):
        for j, opt in enumerate(optimizers):
            runs = [r for r in summary if r["task"] == task and r["optimizer"] == opt]
            if runs:
                data[i, j] = np.mean([r["best_nmse"] for r in runs])

    fig, ax = plt.subplots(figsize=(max(5, len(optimizers) * 2),
                                    max(4, len(task_names) * 0.45 + 1)))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(optimizers)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(task_names)))
    ax.set_yticklabels([t.replace("feynman_", "") for t in task_names], fontsize=7)
    ax.set_title("Best NMSE per task & optimizer  (green = better)", fontsize=10)

    # Annotate cells
    for i in range(len(task_names)):
        for j in range(len(optimizers)):
            ax.text(j, i, f"{data[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if data[i, j] < 0.7 else "white")

    plt.colorbar(im, ax=ax, shrink=0.6, label="NMSE")
    plt.tight_layout()
    path = os.path.join(out_dir, "nmse_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_comparison(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Collect tasks ---
    if args.tasks:
        tasks = [get_task_by_name(name, num_samples=args.num_samples) for name in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    task_names   = [t.name for t in tasks]
    optimizers   = ["rspg", "reinforce", "ppo"]
    seeds        = list(range(args.num_seeds))
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_dir  = "results"
    plots_dir    = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    csv_path = os.path.join(results_dir, f"optimizer_comparison_{timestamp}.csv")

    # CSV header
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task", "optimizer", "seed",
            "best_reward", "best_nmse", "best_expr",
        ])

    all_runs = []   # flat list of result dicts, used for plotting
    summary  = []   # same but without the heavy history arrays (for heatmap)

    total = len(tasks) * len(optimizers) * len(seeds)
    done  = 0

    for task in tasks:
        X, y = task.generate()
        task_runs = []

        for opt_name in optimizers:
            for seed in seeds:
                done += 1
                print(f"\n[{done}/{total}] task={task.name}  optimizer={opt_name}  seed={seed}")

                run = run_one(
                    X=X, y=y,
                    num_variables=task.num_variables,
                    optimizer_name=opt_name,
                    num_episodes=args.num_episodes,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    entropy_weight=args.entropy_weight,
                    device=device,
                    seed=seed,
                )
                run["task"] = task.name

                task_runs.append(run)
                all_runs.append(run)
                summary.append({k: v for k, v in run.items()
                                 if not k.startswith("history_")})

                # Write row immediately so a crash mid-run doesn't lose data
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        task.name, opt_name, seed,
                        f"{run['best_reward']:.6f}",
                        f"{run['best_nmse']:.6f}",
                        run["best_expr"],
                    ])

                print(f"  best_reward={run['best_reward']:.4f}  "
                      f"best_nmse={run['best_nmse']:.4f}  "
                      f"expr={run['best_expr']}")

        # Per-task convergence plot (all optimizers × seeds)
        plot_convergence(task.name, task_runs, plots_dir)

    # Aggregated plots
    plot_summary_barplot(summary, plots_dir)
    plot_nmse_heatmap(summary, task_names, plots_dir)

    # Print console summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — mean best reward across seeds")
    print("=" * 70)
    header = f"{'Task':<30}" + "".join(f"{OPTIMIZER_STYLES[o]['label']:>14}" for o in optimizers)
    print(header)
    print("-" * 70)
    for task_name in task_names:
        row = f"{task_name.replace('feynman_', ''):<30}"
        for opt in optimizers:
            runs = [r for r in summary if r["task"] == task_name and r["optimizer"] == opt]
            if runs:
                mean_r = np.mean([r["best_reward"] for r in runs])
                row += f"{mean_r:>14.4f}"
            else:
                row += f"{'N/A':>14}"
        print(row)
    print("=" * 70)
    print(f"\nCSV saved  → {csv_path}")
    print(f"Plots saved → {plots_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare RSPG / REINFORCE / PPO on Feynman symbolic regression tasks."
    )

    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument(
        "--tasks", nargs="+", default=None,
        metavar="TASK",
        help="Explicit list of task names, e.g. feynman_I_8_14 feynman_I_10_7",
    )
    task_group.add_argument(
        "--suite", type=str, default="pmlb_feynman_subset",
        choices=["nguyen", "nguyen_univariate", "nguyen_bivariate", "pmlb_feynman_subset", "pmlb_feynman_all"],
        help="Named task suite (ignored if --tasks is given)",
    )

    parser.add_argument("--num_episodes",   type=int,   default=2000,
                        help="Episodes per run (default 2000 for a quick comparison)")
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--num_samples",    type=int,   default=100,
                        help="Data points per Feynman task")
    parser.add_argument("--num_seeds",      type=int,   default=3,
                        help="Independent seeds per (task, optimizer) combination")
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4,
                        help="Adam learning rate (Optuna best)")
    parser.add_argument("--entropy_weight", type=float, default=0.017,
                        help="Entropy regularization weight (Optuna best)")

    args = parser.parse_args()
    run_comparison(args)


if __name__ == "__main__":
    main()