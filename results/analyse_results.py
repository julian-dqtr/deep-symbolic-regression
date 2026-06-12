"""
DSR Results Analysis — PMLB Feynman Benchmark (50 000 episodes)
================================================================
Classifies each task by quality level based on NMSE (Normalized Mean
Squared Error) and reward, then generates a full report with cumulative
success rates.

Quality tiers:
  Excellent  : NMSE < 0.01   (near-perfect)
  Very Good  : 0.01 <= NMSE < 0.05
  Good       : 0.05 <= NMSE < 0.15
  Moderate   : 0.15 <= NMSE < 0.50
  Poor       : NMSE >= 0.50  (or reward == -1)
"""

import csv
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────

CSV_PATH = Path(__file__).parent / "results_pmlb_feynman_all_50000.csv"

THRESHOLDS = {
    "Excellent":  0.01,
    "Very Good":  0.05,
    "Good":       0.15,
    "Moderate":   0.50,
}

# Ordered tiers (best to worst)
TIERS = ["Excellent", "Very Good", "Good", "Moderate", "Poor"]


def classify(nmse: float) -> str:
    if nmse < THRESHOLDS["Excellent"]:
        return "Excellent"
    elif nmse < THRESHOLDS["Very Good"]:
        return "Very Good"
    elif nmse < THRESHOLDS["Good"]:
        return "Good"
    elif nmse < THRESHOLDS["Moderate"]:
        return "Moderate"
    else:
        return "Poor"


# ── CSV Loading ────────────────────────────────────────────────────────────────

def load_results(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row["task_name"]:
                continue
            try:
                nmse   = float(row["best_train_nmse"])
                reward = float(row["best_train_reward"])
            except (ValueError, KeyError):
                continue
            rows.append({
                "task":    row["task_name"],
                "reward":  reward,
                "nmse":    nmse,
                "expr":    row.get("best_train_expr", "").strip(),
                "quality": classify(nmse),
            })
    return rows


# ── Statistics ─────────────────────────────────────────────────────────────────

def compute_stats(results: list[dict]) -> dict:
    n      = len(results)
    nmses   = [r["nmse"] for r in results]
    rewards = [r["reward"] for r in results]

    counts = {q: 0 for q in TIERS}
    for r in results:
        counts[r["quality"]] += 1

    # Cumulative rates: how many tasks reached AT LEAST this tier or better
    cumulative = {}
    running = 0
    for tier in TIERS[:-1]:  # Excellent → Moderate (skip Poor)
        running += counts[tier]
        cumulative[tier] = running / n * 100

    return {
        "total":      n,
        "mean_nmse":  sum(nmses) / n,
        "median_nmse": sorted(nmses)[n // 2],
        "min_nmse":   min(nmses),
        "max_nmse":   max(nmses),
        "mean_reward": sum(rewards) / n,
        "counts":     counts,
        "cumulative": cumulative,  # cumulative success rate up to each tier
    }


# ── Display helpers ────────────────────────────────────────────────────────────

COLORS = {
    "Excellent":  "\033[92m",   # green
    "Very Good":  "\033[96m",   # cyan
    "Good":       "\033[93m",   # yellow
    "Moderate":   "\033[95m",   # magenta
    "Poor":       "\033[91m",   # red
    "reset":      "\033[0m",
    "bold":       "\033[1m",
}


def colored(text: str, *codes: str) -> str:
    return "".join(COLORS.get(c, "") for c in codes) + str(text) + COLORS["reset"]


def bar(count: int, total: int, width: int = 30) -> str:
    filled = round(count / total * width) if total else 0
    return "█" * filled + "░" * (width - filled)


# ── Report sections ────────────────────────────────────────────────────────────

def print_summary(s: dict) -> None:
    n = s["total"]
    print(colored("\n══════════════════════════════════════════════════════════", "bold"))
    print(colored("  GLOBAL SUMMARY — PMLB Feynman (50 000 episodes)", "bold"))
    print(colored("══════════════════════════════════════════════════════════", "bold"))
    print(f"  Tasks analyzed : {n}")
    print(f"  Mean reward    : {s['mean_reward']:+.4f}")
    print(f"  Mean NMSE      : {s['mean_nmse']:.4f}")
    print(f"  Median NMSE    : {s['median_nmse']:.4f}")
    print(f"  Min / Max NMSE : {s['min_nmse']:.2e} / {s['max_nmse']:.4f}")

    print()
    print(colored("  Distribution by quality tier:", "bold"))
    header = f"  {'Tier':<12}  {'Bar':^30}  {'Count':>6}   {'%':>6}   {'Cumul. %':>9}"
    print(colored(header, "bold"))
    print(f"  {'─'*75}")
    for tier in TIERS:
        count = s["counts"][tier]
        pct   = count / n * 100
        b     = bar(count, n)
        label = f"{tier:<12}"
        cumul = s["cumulative"].get(tier)
        cumul_str = f"{cumul:6.1f}%" if cumul is not None else "      —"
        print(
            f"  {colored(label, tier)}  {b}  {count:>4} / {n}"
            f"  ({pct:5.1f}%)  cumul: {cumul_str}"
        )

    print()
    print(colored("  Cumulative success rate (best → Moderate):", "bold"))
    for tier in TIERS[:-1]:
        cumul = s["cumulative"][tier]
        b = bar(int(cumul), 100, width=40)
        print(f"  ≤ {colored(tier, tier):<22}  {b}  {cumul:5.1f}%")


def print_series_breakdown(results: list[dict]) -> None:
    """Summary table grouped by Feynman series (I, II, III, test)."""
    series: dict[str, list] = {}
    for r in results:
        t = r["task"]
        if "feynman_I_" in t and "II" not in t and "III" not in t:
            key = "Feynman I"
        elif "feynman_II_" in t and "III" not in t:
            key = "Feynman II"
        elif "feynman_III_" in t:
            key = "Feynman III"
        elif "feynman_test" in t:
            key = "Feynman Test"
        else:
            key = "Other"
        series.setdefault(key, []).append(r)

    print(colored(f"\n{'═'*65}", "bold"))
    print(colored("  BY SERIES", "bold"))
    print(colored(f"{'═'*65}", "bold"))
    header = f"  {'Series':<15} {'N':>4}  {'Mean NMSE':>10}  {'Excellent':>10}  {'Poor':>6}  {'Solve%':>7}"
    print(colored(header, "bold"))
    print(f"  {'─'*63}")
    for key in ["Feynman I", "Feynman II", "Feynman III", "Feynman Test"]:
        if key not in series:
            continue
        s = series[key]
        n = len(s)
        mean_nmse = sum(r["nmse"] for r in s) / n
        n_exc     = sum(1 for r in s if r["quality"] == "Excellent")
        n_poor    = sum(1 for r in s if r["quality"] == "Poor")
        solve_pct = (n - n_poor) / n * 100
        print(
            f"  {key:<15} {n:>4}  {mean_nmse:>10.4f}"
            f"  {n_exc:>10}  {n_poor:>6}  {solve_pct:>6.1f}%"
        )


def print_top_worst(results: list[dict], n: int = 5) -> None:
    sorted_asc = sorted(results, key=lambda x: x["nmse"])
    print(colored(f"\n{'═'*65}", "bold"))
    print(colored(f"  TOP {n} BEST TASKS", "bold"))
    print(colored(f"{'═'*65}", "bold"))
    for r in sorted_asc[:n]:
        print(f"  {colored(r['task'], 'Excellent'):<50}  NMSE={r['nmse']:.2e}")
        print(f"    └─ {r['expr']}")

    print(colored(f"\n{'═'*65}", "bold"))
    print(colored(f"  TOP {n} WORST TASKS", "bold"))
    print(colored(f"{'═'*65}", "bold"))
    for r in sorted_asc[-n:][::-1]:
        print(f"  {colored(r['task'], 'Poor'):<50}  NMSE={r['nmse']:.4f}")
        print(f"    └─ {r['expr'][:80]}")


def print_tier_detail(results: list[dict], tier: str) -> None:
    subset = [r for r in results if r["quality"] == tier]
    if not subset:
        return
    print(colored(f"\n{'─'*62}", "bold"))
    print(colored(f"  {tier.upper()} ({len(subset)} tasks)", tier, "bold"))
    print(colored(f"{'─'*62}", "bold"))
    for r in sorted(subset, key=lambda x: x["nmse"]):
        task_short = r["task"].replace("feynman_", "")
        print(
            f"  {colored(task_short, tier):<40}"
            f"  NMSE={r['nmse']:.4e}"
            f"  reward={r['reward']:+.4f}"
        )
        expr = r["expr"]
        if expr and len(expr) <= 80:
            print(f"    └─ {colored(expr, 'bold')}")
        elif expr:
            print(f"    └─ {colored(expr[:77] + '...', 'bold')}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print(colored("\nLoading results...", "bold"))
    results = load_results(CSV_PATH)

    if not results:
        print("No results found.")
        return

    s = compute_stats(results)

    print_summary(s)
    print_series_breakdown(results)
    print_top_worst(results, n=5)

    for tier in TIERS:
        print_tier_detail(results, tier)

    print(colored("\n══════════════════════════════════════════════════════════\n", "bold"))


if __name__ == "__main__":
    main()
