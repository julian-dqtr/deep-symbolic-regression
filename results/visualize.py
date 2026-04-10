"""
DSR Equation Visualizer
=======================
Visualize the expression tree of a DSR result.

Usage:
    # Random Excellent or Very Good equation from the latest CSV
    python visualize.py

    # Specify a CSV file explicitly
    python visualize.py --csv results_pmlb_feynman_all_50000.csv

    # Display a specific task by name
    python visualize.py --task feynman_I_12_1

    # Combine both
    python visualize.py --csv results_pmlb_feynman_all_50000.csv --task feynman_I_18_12
"""

import os
import sys
import ast
import glob
import argparse
import random
import csv as csv_mod
from pathlib import Path

# Allow absolute imports from project root regardless of cwd
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from src.dsr.analysis.visualizer import ASTVisualizer
from src.dsr.core.factory import build_grammar

# Quality thresholds (same as analyse_results.py)
THRESHOLDS = {"Excellent": 0.01, "Very Good": 0.05}


# ── Helpers ────────────────────────────────────────────────────────────────────

def classify(nmse: float) -> str:
    if nmse < THRESHOLDS["Excellent"]:
        return "Excellent"
    elif nmse < THRESHOLDS["Very Good"]:
        return "Very Good"
    return "Other"


def ast_to_prefix(node) -> list[str]:
    """Recursively convert a Python AST expression node to prefix token list."""
    if isinstance(node, ast.Expression):
        return ast_to_prefix(node.body)
    elif isinstance(node, ast.BinOp):
        op_map = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        return [op_map.get(type(node.op), str(type(node.op)))] + \
               ast_to_prefix(node.left) + ast_to_prefix(node.right)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return ["-", "0"] + ast_to_prefix(node.operand)
    elif isinstance(node, ast.Call):
        return [node.func.id] + ast_to_prefix(node.args[0])
    elif isinstance(node, ast.Name):
        return [node.id]
    elif isinstance(node, ast.Constant):
        return [str(node.value)]
    else:
        return [str(node)]


def load_csv(path: str) -> list[dict]:
    """Load a results CSV and return a list of row dicts with quality labels."""
    rows = []
    with open(path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            if not row.get("task_name"):
                continue
            try:
                nmse   = float(row["best_train_nmse"])
                reward = float(row["best_train_reward"])
                expr   = row.get("best_train_expr", "").strip()
            except (ValueError, KeyError):
                continue
            if not expr:
                continue
            rows.append({
                "task":    row["task_name"],
                "nmse":    nmse,
                "reward":  reward,
                "expr":    expr,
                "quality": classify(nmse),
            })
    return rows


def resolve_csv(csv_arg: str | None) -> str:
    """Return the path to the CSV to use (explicit arg or newest match)."""
    if csv_arg:
        # Accept bare filename (relative to results/) or full path
        p = Path(csv_arg)
        if not p.is_absolute():
            p = Path(script_dir) / p
        if not p.exists():
            print(f"[ERROR] CSV not found: {p}")
            sys.exit(1)
        return str(p)

    # Auto-detect newest results_*.csv in the results folder
    pattern = os.path.join(script_dir, "results_*.csv")
    files   = glob.glob(pattern)
    if not files:
        print("[ERROR] No results CSV found in results/. Use --csv to specify one.")
        sys.exit(1)
    return max(files, key=os.path.getctime)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize an expression tree from DSR results."
    )
    parser.add_argument(
        "--csv",
        metavar="FILE",
        default=None,
        help="Results CSV file to use (filename or full path). "
             "Defaults to the most recent results_*.csv in results/.",
    )
    parser.add_argument(
        "--task",
        metavar="TASK_NAME",
        default=None,
        help="Exact task name to display (e.g. feynman_I_12_1). "
             "Defaults to a random Excellent or Very Good equation.",
    )
    args = parser.parse_args()

    # ── Resolve CSV ────────────────────────────────────────────────────────────
    csv_path = resolve_csv(args.csv)
    print(f"Reading: {csv_path}")
    rows = load_csv(csv_path)

    if not rows:
        print("[ERROR] CSV is empty or has no valid rows.")
        sys.exit(1)

    # ── Select equation ────────────────────────────────────────────────────────
    if args.task:
        # Explicit task requested
        matches = [r for r in rows if r["task"] == args.task]
        if not matches:
            # Try partial / case-insensitive match
            matches = [r for r in rows if args.task.lower() in r["task"].lower()]
        if not matches:
            available = [r["task"] for r in rows]
            print(f"[ERROR] Task '{args.task}' not found.")
            print(f"Available tasks ({len(available)}):")
            for t in sorted(available):
                print(f"  {t}")
            sys.exit(1)
        row = matches[0]
        if len(matches) > 1:
            print(f"[INFO] Multiple matches — showing first: {row['task']}")
    else:
        # Default: random among Excellent + Very Good
        top_rows = [r for r in rows if r["quality"] in ("Excellent", "Very Good")]
        if not top_rows:
            print("[WARN] No Excellent/Very Good equations found; picking random row.")
            top_rows = rows
        row = random.choice(top_rows)
        print(f"[INFO] Randomly selected '{row['quality']}' equation: {row['task']}")

    expr_str  = row["expr"]
    task_name = row["task"]
    nmse      = row["nmse"]
    quality   = row["quality"]

    # ── Display info ───────────────────────────────────────────────────────────
    print()
    print(f"  Task    : {task_name}")
    print(f"  Quality : {quality}")
    print(f"  NMSE    : {nmse:.4e}")
    print(f"  Reward  : {row['reward']:+.4f}")
    print(f"  Expr    : {expr_str}")
    print()

    # ── Parse → prefix tokens ──────────────────────────────────────────────────
    grammar = build_grammar(num_variables=10)
    try:
        tree   = ast.parse(expr_str, mode="eval")
        tokens = ast_to_prefix(tree)
        print(f"  Prefix  : {tokens}")
    except Exception as e:
        print(f"[ERROR] Could not parse expression: {e}")
        sys.exit(1)

    # ── Visualize ──────────────────────────────────────────────────────────────
    out_path = os.path.join(script_dir, f"equation_{task_name}.png")
    vis = ASTVisualizer()
    vis.draw_tree(
        tokens,
        grammar,
        title=f"{task_name}  [{quality}]  NMSE={nmse:.2e}\n{expr_str}",
        filename=out_path,
        show=True,
    )
    print(f"  Saved   : {out_path}")


if __name__ == "__main__":
    main()
