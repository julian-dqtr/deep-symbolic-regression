import csv
import os
import random
import numpy as np
import torch

from ..benchmarks.datasets import get_task_suite
from ..core.expression import safe_prefix_to_infix
from ..core.evaluator import PrefixEvaluator
from ..core.factory import build_grammar
from ..baselines.random_search import random_search
from ..baselines.beam_search import beam_search
from ..training.trainer import Trainer
from ..config import ENV_CONFIG
from ..analysis.visualizer import (
    plot_method_comparison,
    plot_training_history,
    plot_target_vs_prediction,
    save_summary_barplot,
    ensure_dir,
)


RESULTS_DIR = "results"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> str:
    device = device.lower().strip()

    if device == "gpu":
        device = "cuda"

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return "cpu"

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda'.")

    return device


def compute_reward(eval_result, complexity: int) -> float:
    if not eval_result["is_valid"]:
        return float(ENV_CONFIG["invalid_reward"])

    return float(
        -eval_result["nmse"] - ENV_CONFIG["complexity_penalty"] * complexity
    )


def evaluate_tokens(grammar, X, y, tokens):
    evaluator = PrefixEvaluator(grammar)
    eval_result = evaluator.evaluate(tokens=tokens, X=X, y=y)

    return {
        "tokens": tokens,
        "infix": safe_prefix_to_infix(tokens, grammar),
        "nmse": float(eval_result["nmse"]),
        "is_valid": bool(eval_result["is_valid"]),
        "complexity": len(tokens),
        "reward": compute_reward(eval_result, len(tokens)),
    }


def predict_tokens(grammar, X, y, tokens):
    evaluator = PrefixEvaluator(grammar)
    try:
        y_pred = evaluator._eval_prefix(tokens, X)
        return np.asarray(y_pred, dtype=np.float32)
    except Exception:
        return None


def run_random_search_baseline(grammar, X, y, num_trials=1000):
    results = random_search(
        grammar=grammar,
        X=X,
        y=y,
        num_trials=num_trials,
        top_k=1,
    )

    if not results["best_candidates"]:
        return {
            "reward": float("-inf"),
            "nmse": float("inf"),
            "tokens": None,
            "infix": None,
            "is_valid": False,
            "complexity": None,
        }

    best = results["best_candidates"][0]
    return {
        "reward": float(best["reward"]),
        "nmse": float(best["nmse"]),
        "tokens": best["tokens"],
        "infix": best["infix"],
        "is_valid": bool(best["is_valid"]),
        "complexity": int(best["complexity"]),
    }


def run_ppo_sampling(X, y, num_variables, device="cpu", num_episodes=1000):
    trainer = Trainer(
        X=X,
        y=y,
        num_variables=num_variables,
        device=device,
        optimizer_name="ppo",
    )
    trainer.num_episodes = num_episodes

    results = trainer.train()
    grammar = trainer.grammar

    if results["best_episode"] is None:
        return {
            "trainer": trainer,
            "reward": float("-inf"),
            "nmse": float("inf"),
            "tokens": None,
            "infix": None,
            "is_valid": False,
            "complexity": None,
        }

    best_tokens = results["best_episode"]["tokens"]
    evaluated = evaluate_tokens(grammar, X, y, best_tokens)

    return {
        "trainer": trainer,
        **evaluated,
    }


def evaluate_beam_candidates(grammar, X, y, candidates):
    evaluated = []

    for cand in candidates:
        tokens = cand["tokens"]
        eval_result = evaluate_tokens(grammar, X, y, tokens)

        enriched = {
            **eval_result,
            "logprob": float(cand["logprob"]),
            "complete": bool(cand["complete"]),
            "pending_slots": int(cand["pending_slots"]),
        }
        evaluated.append(enriched)

    return evaluated


def sort_candidates_by_reward(candidates):
    return sorted(
        candidates,
        key=lambda c: (
            c["is_valid"],
            c.get("complete", False),
            c["reward"],
            -c["complexity"],
            c.get("logprob", float("-inf")),
        ),
        reverse=True,
    )


def run_ppo_beam_reranked(
    trainer,
    X,
    y,
    beam_width=50,
    max_length=30,
    temperature=1.5,
    device="cpu",
):
    grammar = trainer.grammar
    policy = trainer.policy

    raw_candidates = beam_search(
        policy=policy,
        grammar=grammar,
        X=X,
        y=y,
        beam_width=beam_width,
        max_length=max_length,
        temperature=temperature,
        device=device,
    )

    evaluated_candidates = evaluate_beam_candidates(
        grammar=grammar,
        X=X,
        y=y,
        candidates=raw_candidates,
    )

    reranked = sort_candidates_by_reward(evaluated_candidates)

    if not reranked:
        return {
            "reward": float("-inf"),
            "nmse": float("inf"),
            "tokens": None,
            "infix": None,
            "is_valid": False,
            "complexity": None,
            "candidates": [],
        }

    top = reranked[0]
    return {
        "reward": float(top["reward"]),
        "nmse": float(top["nmse"]),
        "tokens": top["tokens"],
        "infix": top["infix"],
        "is_valid": bool(top["is_valid"]),
        "complexity": int(top["complexity"]),
        "candidates": reranked,
    }


def print_method_result(method_name, result):
    print(f"\n--- {method_name} ---")

    if result["tokens"] is None:
        print("No solution found.")
        return

    print(f"Reward:     {result['reward']:.6f}")
    print(f"NMSE:       {result['nmse']:.6f}")
    print(f"Valid:      {result['is_valid']}")
    print(f"Complexity: {result['complexity']}")
    print(f"Tokens:     {result['tokens']}")
    print(f"Expression: {result['infix']}")


def winner_name(random_result, ppo_result, beam_result):
    scores = {
        "RANDOM": random_result["reward"],
        "PPO_SAMPLING": ppo_result["reward"],
        "PPO_BEAM_RERANKED": beam_result["reward"],
    }
    return max(scores, key=scores.get)


def save_task_plots(task_name, grammar, X, y, random_result, ppo_result, beam_result, trainer):
    task_dir = os.path.join(RESULTS_DIR, "plots", task_name)
    ensure_dir(task_dir)

    # training curves
    if trainer is not None:
        plot_training_history(
            history=trainer.history,
            title=f"Training History - {task_name}",
            show=False,
            save_path=os.path.join(task_dir, "training_history.png"),
        )

    # target vs PPO best
    if ppo_result["tokens"] is not None:
        evaluator = PrefixEvaluator(grammar)
        plot_target_vs_prediction(
            grammar=grammar,
            evaluator=evaluator,
            tokens=ppo_result["tokens"],
            X=X,
            y=y,
            title=f"{task_name} - PPO Best",
            show=False,
            save_path=os.path.join(task_dir, "ppo_target_vs_prediction.png"),
        )

    # comparison plot
    predictions = {}

    if random_result["tokens"] is not None:
        y_pred_random = predict_tokens(grammar, X, y, random_result["tokens"])
        if y_pred_random is not None:
            predictions["Random"] = y_pred_random

    if ppo_result["tokens"] is not None:
        y_pred_ppo = predict_tokens(grammar, X, y, ppo_result["tokens"])
        if y_pred_ppo is not None:
            predictions["PPO"] = y_pred_ppo

    if beam_result["tokens"] is not None:
        y_pred_beam = predict_tokens(grammar, X, y, beam_result["tokens"])
        if y_pred_beam is not None:
            predictions["Beam"] = y_pred_beam

    if predictions:
        plot_method_comparison(
            X=X,
            y=y,
            predictions_dict=predictions,
            title=f"{task_name} - Method Comparison",
            save_path=os.path.join(task_dir, "method_comparison.png"),
            show=False,
        )


def compare_on_task(
    task,
    num_episodes=1000,
    random_trials=1000,
    beam_width=50,
    beam_temperature=1.5,
    device="cpu",
):
    X, y = task.generate()
    grammar = build_grammar(num_variables=task.num_variables)

    print("\n" + "=" * 100)
    print(f"TASK: {task.name}")
    print(f"TARGET: {task.target_expression}")
    print(f"NUM_VARIABLES: {task.num_variables}")
    print(f"NUM_SAMPLES: {len(X)}")
    print("=" * 100)

    random_result = run_random_search_baseline(
        grammar=grammar,
        X=X,
        y=y,
        num_trials=random_trials,
    )
    print_method_result("RANDOM SEARCH", random_result)

    ppo_result = run_ppo_sampling(
        X=X,
        y=y,
        num_variables=task.num_variables,
        device=device,
        num_episodes=num_episodes,
    )
    print_method_result("PPO SAMPLING", ppo_result)

    beam_result = run_ppo_beam_reranked(
        trainer=ppo_result["trainer"],
        X=X,
        y=y,
        beam_width=beam_width,
        max_length=ENV_CONFIG["max_length"],
        temperature=beam_temperature,
        device=device,
    )
    print_method_result("PPO + BEAM RERANKED", beam_result)

    winner = winner_name(random_result, ppo_result, beam_result)
    print("\n>>> WINNER:", winner)

    save_task_plots(
        task_name=task.name,
        grammar=grammar,
        X=X,
        y=y,
        random_result=random_result,
        ppo_result=ppo_result,
        beam_result=beam_result,
        trainer=ppo_result["trainer"],
    )

    return {
        "task": task.name,
        "target": task.target_expression,
        "random_reward": random_result["reward"],
        "random_nmse": random_result["nmse"],
        "random_expr": random_result["infix"],
        "ppo_reward": ppo_result["reward"],
        "ppo_nmse": ppo_result["nmse"],
        "ppo_expr": ppo_result["infix"],
        "beam_reward": beam_result["reward"],
        "beam_nmse": beam_result["nmse"],
        "beam_expr": beam_result["infix"],
        "winner": winner,
    }


def print_summary_table(rows):
    print("\n" + "=" * 140)
    print("FINAL SUMMARY TABLE")
    print("=" * 140)

    header = (
        f"{'Task':<20}"
        f"{'Random Reward':<18}"
        f"{'PPO Reward':<18}"
        f"{'Beam Reward':<18}"
        f"{'Winner':<20}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['task']:<20}"
            f"{row['random_reward']:<18.6f}"
            f"{row['ppo_reward']:<18.6f}"
            f"{row['beam_reward']:<18.6f}"
            f"{row['winner']:<20}"
        )

    print("=" * 140)

    print("\nBest expressions per method:")
    for row in rows:
        print(f"\nTask: {row['task']}")
        print(f"  Random: {row['random_expr']}")
        print(f"  PPO:    {row['ppo_expr']}")
        print(f"  Beam:   {row['beam_expr']}")


def export_results_csv(rows, filepath):
    ensure_dir(os.path.dirname(filepath))

    fieldnames = [
        "task",
        "target",
        "random_reward",
        "random_nmse",
        "random_expr",
        "ppo_reward",
        "ppo_nmse",
        "ppo_expr",
        "beam_reward",
        "beam_nmse",
        "beam_expr",
        "winner",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\nCSV exported to: {filepath}")


def run_benchmark_suite(
    suite_name="toy",
    task_indices=None,
    num_samples=100,
    num_episodes=1000,
    random_trials=1000,
    beam_width=50,
    beam_temperature=1.5,
    device="cpu",
):
    device = resolve_device(device)
    tasks = get_task_suite(name=suite_name, num_samples=num_samples)

    if task_indices is None:
        task_indices = list(range(len(tasks)))

    rows = []

    for idx in task_indices:
        if idx < 0 or idx >= len(tasks):
            raise ValueError(f"Invalid task index: {idx}")

        set_seed(42 + idx)

        row = compare_on_task(
            task=tasks[idx],
            num_episodes=num_episodes,
            random_trials=random_trials,
            beam_width=beam_width,
            beam_temperature=beam_temperature,
            device=device,
        )
        rows.append(row)

    print_summary_table(rows)

    csv_path = os.path.join(RESULTS_DIR, f"{suite_name}_benchmark_results.csv")
    fig_path = os.path.join(RESULTS_DIR, f"{suite_name}_benchmark_summary.png")

    export_results_csv(rows, csv_path)
    save_summary_barplot(rows, save_path=fig_path, show=False)

    print(f"Figure exported to: {fig_path}")

    return rows


def main():
    run_benchmark_suite(
        suite_name="nguyen",
        task_indices=[0, 1, 2, 3, 4, 5, 6],
        num_samples=100,
        num_episodes=1000,
        random_trials=1000,
        beam_width=50,
        beam_temperature=1.5,
        device="cuda",
    )


if __name__ == "__main__":
    main()