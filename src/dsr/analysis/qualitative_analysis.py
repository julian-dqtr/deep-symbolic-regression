"""
qualitative_analysis.py
=======================
Qualitative analysis of successes and failures on the Feynman SR benchmark.

Answers the question asked by the professor:
  "Why do some formulas get recovered and others not?"

Analyses three dimensions:
  1. Number of variables     — more variables = harder?
  2. Expression depth        — deeply nested functions harder to recover?
  3. Function families       — which operators (sin, exp, sqrt, ...) cause failures?

Input: a true_vs_recovered CSV produced by evaluate_expressions.py.

Usage
-----
python -m dsr.analysis.qualitative_analysis \
    --csv results/true_vs_recovered_<timestamp>.csv

Output
------
results/
  qualitative_analysis_<timestamp>.csv   <- per-task feature table
  plots/
    qual_by_num_vars.png     <- NMSE vs number of variables
    qual_by_depth.png        <- NMSE vs expression depth
    qual_by_operator.png     <- success rate per operator family
    qual_success_table.png   <- annotated table of best/worst cases
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from ..data.feynman_ground_truth import FEYNMAN_GROUND_TRUTH, classify_quality, DIFF_ORDER


# ---------------------------------------------------------------------------
# Feature extraction from ground-truth expressions
# ---------------------------------------------------------------------------

OPERATORS = ["sin", "cos", "exp", "log", "sqrt", "arcsin", "arccos", "tanh"]
BINARY_OPS = ["+", "-", "*", "/"]


def expression_depth(expr: str) -> int:
    """
    Compute the true nesting depth of a symbolic expression string.

    Strategy: depth = maximum parenthesis nesting level, where each function
    call (sin, cos, exp, log, sqrt, ...) contributes one level.

    This is more accurate than simply counting all '(' characters because
    arithmetic grouping parentheses (e.g. in '(x0 * x1)') also open a
    depth level — which is the correct interpretation: a binary operation
    applied to two sub-expressions has depth = 1 + max(depth(left), depth(right)).

    For the purposes of this qualitative analysis, 'depth' is used as a
    proxy for expression complexity. The parenthesis-depth heuristic is
    consistent with this interpretation.

    Examples
    --------
    'x0'                   → depth 0   (terminal)
    'x0 * x1'              → depth 1   (one binary op)
    'sin(x0)'              → depth 1   (one unary call)
    'sin(exp(x0))'         → depth 2   (nested unary calls)
    'x0 * sin(x1 + x2)'   → depth 2   (binary op containing unary call)
    """
    max_depth = 0
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == ")":
            depth -= 1
    return max_depth


def operators_present(expr: str) -> List[str]:
    """Return list of non-trivial operators present in the expression."""
    found = []
    for op in OPERATORS:
        if re.search(rf"\b{op}\b", expr):
            found.append(op)
    return found


def has_relativistic_structure(expr: str) -> bool:
    """Detect sqrt(1 - v^2/c^2) — Lorentz factor structure."""
    return "sqrt" in expr and ("v**2" in expr or "u**2" in expr)


def has_exponential(expr: str) -> bool:
    return "exp" in expr


def has_nested_trig(expr: str) -> bool:
    """Detect sin(cos(...)) or cos(sin(...)) patterns."""
    return bool(re.search(r"(sin|cos)\([^)]*(?:sin|cos)", expr))


def extract_features(task_name: str, true_expr: str, num_vars: int) -> Dict:
    """Extract interpretable features from a task."""
    depth = expression_depth(true_expr)
    ops   = operators_present(true_expr)
    n_ops = len(ops)

    return {
        "task":          task_name,
        "num_vars":      num_vars,
        "expr_depth":    depth,
        "n_special_ops": n_ops,
        "has_exp":       int(has_exponential(true_expr)),
        "has_trig":      int(any(op in ops for op in ["sin", "cos", "tanh"])),
        "has_log":       int("log" in ops),
        "has_sqrt":      int("sqrt" in ops),
        "has_arcsin":    int("arcsin" in ops or "arccos" in ops),
        "has_lorentz":   int(has_relativistic_structure(true_expr)),
        "operators":     ", ".join(ops) if ops else "none",
    }


# ---------------------------------------------------------------------------
# Load results CSV
# ---------------------------------------------------------------------------

def load_results(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["task"]
            gt        = FEYNMAN_GROUND_TRUTH.get(task_name, {})
            true_expr = gt.get("expr", row.get("true_expression", ""))
            num_vars  = int(row.get("num_vars", 0) or len(gt.get("vars", [])))
            nmse      = float(row.get("nmse", 1.0))

            features = extract_features(task_name, true_expr, num_vars)
            features.update({
                "difficulty":            gt.get("difficulty", row.get("difficulty", "Unknown")),
                "true_expression":       true_expr,
                "recovered_expression":  row.get("recovered_expression", ""),
                "nmse":                  nmse,
                "quality":               classify_quality(nmse),
            })
            rows.append(features)
    return rows


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

QUALITY_COLORS = {
    "Perfect": "#639922",
    "Good":    "#EF9F27",
    "Poor":    "#E24B4A",
}


def plot_nmse_vs_num_vars(rows: List[Dict], out_dir: str):
    """Scatter: NMSE vs number of variables, colored by quality."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for q, color in QUALITY_COLORS.items():
        subset = [r for r in rows if r["quality"] == q]
        if not subset:
            continue
        xs = [r["num_vars"] + np.random.uniform(-0.15, 0.15) for r in subset]
        ys = [r["nmse"] for r in subset]
        ax.scatter(xs, ys, c=color, label=q, s=40, alpha=0.8, zorder=3)

    # Median per num_vars
    by_nv = defaultdict(list)
    for r in rows:
        by_nv[r["num_vars"]].append(r["nmse"])
    for nv, nmses in sorted(by_nv.items()):
        ax.hlines(np.median(nmses), nv - 0.3, nv + 0.3,
                  colors="#333", lw=1.5, zorder=4)

    ax.axhline(0.001, color=QUALITY_COLORS["Perfect"],
               linestyle="--", lw=0.8, label="Perfect threshold")
    ax.axhline(0.05, color=QUALITY_COLORS["Good"],
               linestyle="--", lw=0.8, label="Good threshold")

    ax.set_xlabel("Number of variables", fontsize=10)
    ax.set_ylabel("NMSE (log scale)", fontsize=10)
    ax.set_yscale("log")
    ax.set_xticks(sorted(by_nv.keys()))
    ax.set_title("NMSE vs number of variables", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    path = os.path.join(out_dir, "qual_by_num_vars.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_nmse_vs_depth(rows: List[Dict], out_dir: str):
    """Scatter: NMSE vs expression depth."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for q, color in QUALITY_COLORS.items():
        subset = [r for r in rows if r["quality"] == q]
        if not subset:
            continue
        xs = [r["expr_depth"] + np.random.uniform(-0.2, 0.2) for r in subset]
        ys = [r["nmse"] for r in subset]
        ax.scatter(xs, ys, c=color, label=q, s=40, alpha=0.8, zorder=3)

    by_d = defaultdict(list)
    for r in rows:
        by_d[r["expr_depth"]].append(r["nmse"])
    for d, nmses in sorted(by_d.items()):
        ax.hlines(np.median(nmses), d - 0.35, d + 0.35,
                  colors="#333", lw=1.5, zorder=4)

    ax.axhline(0.001, color=QUALITY_COLORS["Perfect"],
               linestyle="--", lw=0.8)
    ax.axhline(0.05, color=QUALITY_COLORS["Good"],
               linestyle="--", lw=0.8)

    ax.set_xlabel("Expression nesting depth", fontsize=10)
    ax.set_ylabel("NMSE (log scale)", fontsize=10)
    ax.set_yscale("log")
    ax.set_xticks(sorted(by_d.keys()))
    ax.set_title("NMSE vs expression depth", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    path = os.path.join(out_dir, "qual_by_depth.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_operator_success_rate(rows: List[Dict], out_dir: str):
    """
    Bar chart: for each operator type, what % of tasks containing it
    were recovered as Perfect or Good?
    """
    op_features = [
        ("has_exp",     "exp"),
        ("has_trig",    "sin/cos/tanh"),
        ("has_log",     "log"),
        ("has_sqrt",    "sqrt"),
        ("has_arcsin",  "arcsin/arccos"),
        ("has_lorentz", "Lorentz factor\nsqrt(1-v²/c²)"),
    ]

    labels, success_rates, counts = [], [], []
    for feat, label in op_features:
        subset = [r for r in rows if r[feat] == 1]
        if not subset:
            continue
        good = sum(1 for r in subset if r["quality"] in ("Perfect", "Good"))
        labels.append(label)
        success_rates.append(100 * good / len(subset))
        counts.append(len(subset))

    no_ops = [r for r in rows if r["n_special_ops"] == 0]
    if no_ops:
        good = sum(1 for r in no_ops if r["quality"] in ("Perfect", "Good"))
        labels.append("No special\noperators")
        success_rates.append(100 * good / len(no_ops))
        counts.append(len(no_ops))

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(labels))
    colors = [
        "#E24B4A" if r < 30 else "#EF9F27" if r < 60 else "#639922"
        for r in success_rates
    ]

    bars = ax.bar(x, success_rates, color=colors, alpha=0.85, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Recovery rate  (Perfect + Good) %", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title("Success rate by operator family in true expression", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    for bar, rate, n in zip(bars, success_rates, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{rate:.0f}%\n(n={n})",
            ha="center", va="bottom", fontsize=8,
        )

    plt.tight_layout()
    path = os.path.join(out_dir, "qual_by_operator.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_best_and_worst(rows: List[Dict], out_dir: str, n: int = 5):
    """
    Side-by-side table of the n best and n worst recovered expressions.
    The clearest qualitative evidence for the paper.
    """
    sorted_rows = sorted(rows, key=lambda r: r["nmse"])
    best  = sorted_rows[:n]
    worst = sorted_rows[-n:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, n * 0.7 + 1.5)))
    fig.suptitle(
        "Qualitative analysis — best and worst recovered expressions",
        fontsize=12, fontweight="bold",
    )

    for ax, subset, title in [
        (axes[0], best,  f"Top {n} successes"),
        (axes[1], worst, f"Top {n} failures"),
    ]:
        ax.axis("off")
        ax.set_title(
            title, fontsize=11, pad=10,
            color=QUALITY_COLORS["Perfect"] if "success" in title
            else QUALITY_COLORS["Poor"],
        )

        col_labels = ["Task", "True expression", "Recovered", "NMSE", "Q"]
        col_w      = [0.18, 0.32, 0.32, 0.10, 0.06]

        y = 1.0
        for label, w in zip(col_labels, col_w):
            ax.text(
                sum(col_w[:col_labels.index(label)]) + 0.01, y,
                label, fontsize=8, fontweight="bold",
                va="top", transform=ax.transAxes,
            )
        ax.axhline(
            y=1.0 - 0.3 / (n + 1.5), color="#aaa",
            lw=0.6, xmin=0.01, xmax=0.99,
        )

        row_h = (1.0 - 0.8 / (n + 1.5)) / max(n, 1)
        for i, row in enumerate(subset):
            y   = 1.0 - (i + 1.5) * row_h
            bg  = "#F8F8F4" if i % 2 == 0 else "white"
            ax.axhspan(
                y - row_h * 0.5, y + row_h * 0.5,
                xmin=0.005, xmax=0.995, color=bg, zorder=0,
            )

            short  = row["task"].replace("feynman_", "")
            true_e = (
                row["true_expression"][:28] + "…"
                if len(row["true_expression"]) > 28
                else row["true_expression"]
            )
            rec_e  = (
                row["recovered_expression"][:28] + "…"
                if len(row["recovered_expression"]) > 28
                else row["recovered_expression"]
            )
            q_col  = QUALITY_COLORS.get(row["quality"], "#888")

            vals   = [short, true_e, rec_e, f"{row['nmse']:.4f}", row["quality"]]
            colors = ["#222", "#444", "#444", "#333", q_col]

            x_pos = 0.01
            for val, w, color in zip(vals, col_w, colors):
                ax.text(
                    x_pos, y, val, fontsize=7, va="center",
                    transform=ax.transAxes, color=color, clip_on=True,
                )
                x_pos += w

    plt.tight_layout()
    path = os.path.join(out_dir, "qual_success_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(rows: List[Dict]):
    print("\n" + "=" * 70)
    print("QUALITATIVE ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n1. Recovery rate by number of variables")
    print("-" * 50)
    by_nv = defaultdict(list)
    for r in rows:
        by_nv[r["num_vars"]].append(r)
    for nv in sorted(by_nv.keys()):
        subset = by_nv[nv]
        good   = sum(1 for r in subset if r["quality"] in ("Perfect", "Good"))
        n      = len(subset)
        print(
            f"  {nv} variable{'s' if nv > 1 else ' '}: "
            f"{good}/{n} recovered  ({100*good/n:.0f}%)  "
            f"median NMSE={np.median([r['nmse'] for r in subset]):.3f}"
        )

    print("\n2. Recovery rate by expression depth")
    print("-" * 50)
    by_d = defaultdict(list)
    for r in rows:
        by_d[r["expr_depth"]].append(r)
    for d in sorted(by_d.keys()):
        subset = by_d[d]
        good   = sum(1 for r in subset if r["quality"] in ("Perfect", "Good"))
        n      = len(subset)
        print(
            f"  Depth {d}: {good}/{n} recovered  ({100*good/n:.0f}%)  "
            f"median NMSE={np.median([r['nmse'] for r in subset]):.3f}"
        )

    print("\n3. Recovery rate by operator family")
    print("-" * 50)
    op_features = [
        ("has_exp",     "exp"),
        ("has_trig",    "sin/cos/tanh"),
        ("has_log",     "log"),
        ("has_sqrt",    "sqrt"),
        ("has_arcsin",  "arcsin/arccos"),
        ("has_lorentz", "Lorentz factor"),
    ]
    for feat, label in op_features:
        subset = [r for r in rows if r[feat] == 1]
        if not subset:
            continue
        good = sum(1 for r in subset if r["quality"] in ("Perfect", "Good"))
        n    = len(subset)
        print(f"  {label:<20}: {good}/{n} recovered  ({100*good/n:.0f}%)")

    no_ops = [r for r in rows if r["n_special_ops"] == 0]
    if no_ops:
        good = sum(1 for r in no_ops if r["quality"] in ("Perfect", "Good"))
        n    = len(no_ops)
        print(f"  {'No special ops':<20}: {good}/{n} recovered  ({100*good/n:.0f}%)")

    print("\n4. Top 3 successes")
    print("-" * 50)
    for r in sorted(rows, key=lambda x: x["nmse"])[:3]:
        print(
            f"  {r['task']:<25}  NMSE={r['nmse']:.4f}  "
            f"{r['recovered_expression'][:50]}"
        )

    print("\n5. Top 3 failures")
    print("-" * 50)
    for r in sorted(rows, key=lambda x: -x["nmse"])[:3]:
        print(
            f"  {r['task']:<25}  NMSE={r['nmse']:.4f}  "
            f"true: {r['true_expression'][:40]}"
        )
    print()


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "task", "difficulty", "num_vars", "expr_depth",
    "n_special_ops", "operators",
    "has_exp", "has_trig", "has_log", "has_sqrt", "has_arcsin", "has_lorentz",
    "true_expression", "recovered_expression", "nmse", "quality",
]


def save_csv(rows: List[Dict], path: str):
    os.makedirs(
        os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
    )
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Qualitative analysis of RSPG symbolic regression results."
    )
    parser.add_argument(
        "--csv", type=str, required=True, metavar="PATH",
        help="Path to true_vs_recovered CSV from evaluate_expressions.py",
    )
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="Number of best/worst cases to show (default 5)",
    )

    args        = parser.parse_args()
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Loading results from {args.csv}")
    rows = load_results(args.csv)
    print(f"Loaded {len(rows)} tasks")

    print_summary(rows)

    plot_nmse_vs_num_vars(rows, plots_dir)
    plot_nmse_vs_depth(rows, plots_dir)
    plot_operator_success_rate(rows, plots_dir)
    plot_best_and_worst(rows, plots_dir, n=args.top_n)

    out_csv = os.path.join(results_dir, f"qualitative_analysis_{timestamp}.csv")
    save_csv(rows, out_csv)

    print(f"CSV   → {out_csv}")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()