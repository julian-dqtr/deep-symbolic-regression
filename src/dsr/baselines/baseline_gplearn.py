"""
baseline_gplearn.py
===================
Compares gplearn (genetic programming baseline) against your RSPG system
on the Feynman SR benchmark.

gplearn is the standard open-source SR baseline used in DSR (Petersen et al., 2021)
and most subsequent papers. Including it anchors your results in the literature
and makes the comparison honest — even if RSPG loses on some tasks.
"""

import argparse
import csv
import os
import random
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from gplearn.genetic import SymbolicRegressor
    from gplearn.functions import make_function
except ImportError:
    raise ImportError(
        "gplearn is not installed.\n"
        "Run:  pip install gplearn\n"
        "Docs: https://gplearn.readthedocs.io"
    )

# gplearn does not include 'exp' or 'log' as built-in string names.
# Define them as protected custom functions matching your evaluator:
#   exp: clipped to [-20, 20] to avoid overflow (same as evaluator.py)
#   log: protected log(|x| + eps)                (same as evaluator.py)
_EPS = 1e-6
_gp_exp = make_function(
    function=lambda x: np.exp(np.clip(x, -20, 20)),
    name="exp",
    arity=1,
)
_gp_log = make_function(
    function=lambda x: np.log(np.abs(x) + _EPS),
    name="log",
    arity=1,
)

from ..data.datasets import get_task_suite, get_task_by_name
from ..data.feynman_ground_truth import (
    FEYNMAN_GROUND_TRUTH,
    NMSE_PERFECT,
    NMSE_GOOD,
    DIFF_ORDER,
    classify_quality,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def compute_nmse(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """Normalised MSE, clipped to [0, 1]."""
    if not np.all(np.isfinite(y_pred)):
        return 1.0
    mse    = np.mean((y_true - y_pred) ** 2)
    var_y  = np.var(y_true)
    nmse   = float(mse / var_y) if var_y > eps else float(mse)
    return float(min(nmse, 1.0))


# ---------------------------------------------------------------------------
# gplearn runner
# ---------------------------------------------------------------------------

def run_gplearn(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    population_size: int,
    generations: int,
    n_jobs: int,
) -> Dict:
    """
    Fit a gplearn SymbolicRegressor and return metrics.

    gplearn uses genetic programming — no gradient, no neural network.
    It is the canonical open-source SR baseline cited in DSR and most
    subsequent papers.

    Function set mirrors your grammar:
      binary : add, sub, mul, div
      unary  : sin, cos, exp, log (protected, same as your evaluator)
    """
    set_seed(seed)

    est = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        tournament_size=20,
        stopping_criteria=0.0,
        const_range=(-5.0, 5.0),
        init_depth=(2, 6),
        init_method="half and half",
        function_set=("add", "sub", "mul", "div", "sin", "cos", _gp_exp, _gp_log),
        metric="mse",
        parsimony_coefficient=0.001,
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        warm_start=False,
        n_jobs=n_jobs,
        verbose=0,
        random_state=seed,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            est.fit(X, y)
            y_pred = est.predict(X)
            nmse   = compute_nmse(y, y_pred)
            expr   = str(est._program)
        except Exception as e:
            nmse = 1.0
            expr = f"ERROR: {e}"

    return {
        "method":  "gplearn",
        "nmse":    round(nmse, 6),
        "quality": classify_quality(nmse),
        "expr":    expr,
    }


# ---------------------------------------------------------------------------
# Load RSPG results from an existing CSV
# ---------------------------------------------------------------------------

def load_rspg_csv(csv_path: str) -> Dict[str, Dict]:
    """
    Load RSPG results from a true_vs_recovered CSV produced by
    evaluate_expressions.py. Returns {task_name: {nmse, quality, expr}}.
    """
    results = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row.get("task", "")
            if not task:
                continue
            nmse = float(row.get("nmse", 1.0))
            results[task] = {
                "method":  "rspg",
                "nmse":    round(nmse, 6),
                "quality": classify_quality(nmse),
                "expr":    row.get("recovered_expression", ""),
            }
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METHOD_STYLES = {
    "rspg":    {"color": "#D85A30", "label": "RSPG (ours)"},
    "gplearn": {"color": "#378ADD", "label": "gplearn (GP baseline)"},
}

QUALITY_COLORS = {
    "Perfect": "#639922",
    "Good":    "#EF9F27",
    "Poor":    "#E24B4A",
}


def plot_barplot(rows: List[Dict], out_dir: str):
    """Side-by-side mean NMSE per method, per difficulty tier."""
    fig, axes = plt.subplots(1, len(DIFF_ORDER), figsize=(14, 4), sharey=True)
    fig.suptitle("RSPG vs gplearn — mean NMSE by difficulty", fontsize=12, fontweight="bold")

    for ax, diff in zip(axes, DIFF_ORDER):
        subset = [r for r in rows if r["difficulty"] == diff]
        if not subset:
            ax.set_title(diff, fontsize=10)
            ax.set_visible(False)
            continue

        for xi, method in enumerate(["rspg", "gplearn"]):
            nmses  = [r[f"{method}_nmse"] for r in subset if f"{method}_nmse" in r]
            mean_n = np.mean(nmses) if nmses else 1.0
            std_n  = np.std(nmses)  if nmses else 0.0
            style  = METHOD_STYLES[method]
            bar = ax.bar(xi, mean_n, yerr=std_n, capsize=4,
                         color=style["color"], alpha=0.85, width=0.55,
                         error_kw={"lw": 1.0})
            ax.text(xi, mean_n + std_n + 0.02, f"{mean_n:.3f}",
                    ha="center", fontsize=9)

        ax.set_title(f"{diff}\n(n={len(subset)})", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["RSPG", "gplearn"], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("NMSE" if diff == DIFF_ORDER[0] else "")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    patches = [mpatches.Patch(color=v["color"], label=v["label"])
               for v in METHOD_STYLES.values()]
    fig.legend(handles=patches, loc="lower center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    path = os.path.join(out_dir, "gplearn_vs_rspg_barplot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_winner_heatmap(rows: List[Dict], out_dir: str):
    """
    One row per task. Color = winner (RSPG coral / gplearn blue / tie gray).
    """
    order = {d: i for i, d in enumerate(DIFF_ORDER)}
    rows  = sorted(rows, key=lambda r: (order.get(r["difficulty"], 9), r["rspg_nmse"]))

    n   = len(rows)
    fig_h = max(4, n * 0.32 + 1.0)
    fig, ax = plt.subplots(figsize=(6, fig_h))
    ax.axis("off")
    ax.set_title("Winner per task (RSPG vs gplearn)", fontsize=11, pad=12)

    cols   = ["Task", "Diff.", "RSPG NMSE", "gplearn NMSE", "Winner"]
    widths = [0.32,   0.12,   0.18,         0.18,           0.16]
    y_h    = 1.0
    x      = 0.02
    for col, w in zip(cols, widths):
        ax.text(x, y_h, col, fontsize=8, fontweight="bold",
                va="top", transform=ax.transAxes)
        x += w

    ax.axhline(y=1.0 - 0.4 / fig_h, color="#aaa", linewidth=0.6,
               xmin=0.01, xmax=0.99)

    row_h = (1.0 - 1.1 / fig_h) / max(n, 1)
    for i, row in enumerate(rows):
        y    = 1.0 - (i + 1.5) * row_h
        bg   = "#F8F8F4" if i % 2 == 0 else "white"
        ax.axhspan(y - row_h * 0.5, y + row_h * 0.5,
                   xmin=0.005, xmax=0.995, color=bg, zorder=0)

        rspg_n = row.get("rspg_nmse", 1.0)
        gp_n   = row.get("gplearn_nmse", 1.0)
        diff   = rspg_n - gp_n

        if abs(diff) < 0.01:
            winner, w_color = "Tie", "#888780"
        elif diff < 0:
            winner, w_color = "RSPG ▲", "#993C1D"
        else:
            winner, w_color = "gplearn ▲", "#185FA5"

        vals   = [
            row["task"].replace("feynman_", ""),
            row["difficulty"],
            f"{rspg_n:.4f}",
            f"{gp_n:.4f}",
            winner,
        ]
        colors = ["#222", "#555", "#333", "#333", w_color]
        x = 0.02
        for val, w, color in zip(vals, widths, colors):
            ax.text(x, y, val, fontsize=7, va="center",
                    transform=ax.transAxes, color=color)
            x += w

    plt.tight_layout()
    path = os.path.join(out_dir, "gplearn_vs_rspg_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(rows: List[Dict]):
    order = {d: i for i, d in enumerate(DIFF_ORDER)}
    rows  = sorted(rows, key=lambda r: (order.get(r["difficulty"], 9), r["rspg_nmse"]))

    w = max(len(r["task"]) for r in rows) + 2
    print()
    print(f"{'Task':<{w}} {'Diff.':<8} {'RSPG NMSE':>10}  {'gplearn NMSE':>13}  Winner")
    print("-" * (w + 40))

    rspg_wins, gp_wins, ties = 0, 0, 0
    for r in rows:
        rn    = r.get("rspg_nmse",    1.0)
        gn    = r.get("gplearn_nmse", 1.0)
        diff  = rn - gn
        if abs(diff) < 0.01:
            winner = "Tie"
            ties  += 1
        elif diff < 0:
            winner = "RSPG ◀"
            rspg_wins += 1
        else:
            winner = "gplearn ◀"
            gp_wins  += 1
        print(f"{r['task']:<{w}} {r['difficulty']:<8} {rn:>10.4f}  {gn:>13.4f}  {winner}")

    n = len(rows)
    print()
    print("=" * 60)
    print(f"  RSPG wins  : {rspg_wins}/{n} ({100*rspg_wins/n:.0f}%)")
    print(f"  gplearn    : {gp_wins}/{n}  ({100*gp_wins/n:.0f}%)")
    print(f"  Ties       : {ties}/{n}   ({100*ties/n:.0f}%)")

    for diff in DIFF_ORDER:
        subset = [r for r in rows if r["difficulty"] == diff]
        if not subset:
            continue
        rspg_mean = np.mean([r.get("rspg_nmse", 1.0) for r in subset])
        gp_mean   = np.mean([r.get("gplearn_nmse", 1.0) for r in subset])
        winner    = "RSPG" if rspg_mean < gp_mean else "gplearn"
        print(f"  {diff:<8}: RSPG={rspg_mean:.4f}  gplearn={gp_mean:.4f}  → {winner}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "difficulty", "num_vars", "true_expression",
    "rspg_nmse", "rspg_quality", "rspg_expr",
    "gplearn_nmse", "gplearn_quality", "gplearn_expr",
    "winner",
]


def save_csv(rows: List[Dict], path: str):
    order = {d: i for i, d in enumerate(DIFF_ORDER)}
    rows  = sorted(rows, key=lambda r: (order.get(r["difficulty"], 9), r["rspg_nmse"]))
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
        description="Compare gplearn (GP baseline) vs RSPG on Feynman tasks."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["pmlb_feynman_subset", "pmlb_feynman_all"])
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")

    parser.add_argument("--rspg_csv", type=str, default=None, metavar="PATH",
                        help="Path to a true_vs_recovered CSV from evaluate_expressions.py. "
                             "If provided, RSPG results are loaded instead of re-trained.")
    parser.add_argument("--num_samples",     type=int, default=100)
    parser.add_argument("--seed",            type=int, default=42)
    # gplearn settings
    parser.add_argument("--gp_population",   type=int, default=1000,
                        help="gplearn population size (default 1000, paper uses 1000-5000)")
    parser.add_argument("--gp_generations",  type=int, default=20,
                        help="gplearn generations (default 20 for speed; use 50+ for paper)")
    parser.add_argument("--gp_n_jobs",       type=int, default=-1,
                        help="gplearn parallel jobs (-1 = all cores)")

    args = parser.parse_args()

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples) for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    # Load RSPG results if CSV provided
    rspg_cache = load_rspg_csv(args.rspg_csv) if args.rspg_csv else {}

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    csv_path    = os.path.join(results_dir, f"gplearn_comparison_{timestamp}.csv")

    rows = []

    for idx, task in enumerate(tasks):
        X, y = task.generate()
        gt   = FEYNMAN_GROUND_TRUTH.get(task.name, {})
        print(f"\n[{idx+1}/{len(tasks)}] {task.name}  ({gt.get('difficulty','?')})")

        # --- RSPG ---
        if task.name in rspg_cache:
            rspg = rspg_cache[task.name]
            print(f"  RSPG  (from CSV): NMSE={rspg['nmse']:.4f}  quality={rspg['quality']}")
        else:
            print("  RSPG results not found in CSV — set --rspg_csv to load them.")
            rspg = {"nmse": float("nan"), "quality": "N/A", "expr": ""}

        # --- gplearn ---
        print(f"  gplearn: pop={args.gp_population}, gen={args.gp_generations} ...",
              flush=True)
        gp = run_gplearn(
            X=X, y=y,
            seed=args.seed,
            population_size=args.gp_population,
            generations=args.gp_generations,
            n_jobs=args.gp_n_jobs,
        )
        print(f"  gplearn: NMSE={gp['nmse']:.4f}  quality={gp['quality']}")
        print(f"           expr={gp['expr'][:70]}")

        # Winner
        rn, gn = rspg["nmse"], gp["nmse"]
        if np.isnan(rn):
            winner = "unknown"
        elif abs(rn - gn) < 0.01:
            winner = "tie"
        elif rn < gn:
            winner = "rspg"
        else:
            winner = "gplearn"

        row = {
            "task":            task.name,
            "difficulty":      gt.get("difficulty", "Unknown"),
            "num_vars":        task.num_variables,
            "true_expression": gt.get("expr", "unknown"),
            "rspg_nmse":       rspg["nmse"],
            "rspg_quality":    rspg["quality"],
            "rspg_expr":       rspg["expr"],
            "gplearn_nmse":    gp["nmse"],
            "gplearn_quality": gp["quality"],
            "gplearn_expr":    gp["expr"],
            "winner":          winner,
        }
        rows.append(row)

        # Write immediately
        mode = "w" if idx == 0 else "a"
        with open(csv_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if idx == 0:
                writer.writeheader()
            writer.writerow(row)

    print_summary(rows)
    save_csv(rows, csv_path)
    plot_barplot(rows, plots_dir)
    plot_winner_heatmap(rows, plots_dir)

    print(f"\nCSV   → {csv_path}")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()