import random
import numpy as np
import torch

from ..core.factory import build_grammar
from ..core.expression import safe_prefix_to_infix
from ..benchmarks.datasets import load_dataset
from ..baselines.random_search import random_search
from ..training.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_random(grammar, X, y, num_trials: int = 1000):
    results = random_search(
        grammar=grammar,
        X=X,
        y=y,
        num_trials=num_trials,
        top_k=1,
    )

    best = results["best_candidates"][0]

    return {
        "reward": best["reward"],
        "expr_tokens": best["tokens"],
        "expr_infix": best["infix"],
        "nmse": best["nmse"],
        "raw_results": results,
    }


def run_rl(X, y, num_variables: int, device: str = "cpu", num_episodes: int = 1000):
    trainer = Trainer(
        X=X,
        y=y,
        num_variables=num_variables,
        device=device,
    )
    trainer.num_episodes = num_episodes

    results = trainer.train()

    best_episode = results["best_episode"]
    grammar = trainer.grammar

    if best_episode is None:
        return {
            "reward": float("-inf"),
            "expr_tokens": None,
            "expr_infix": None,
            "raw_results": results,
        }

    best_tokens = best_episode["tokens"]

    return {
        "reward": results["best_reward"],
        "expr_tokens": best_tokens,
        "expr_infix": safe_prefix_to_infix(best_tokens, grammar),
        "raw_results": results,
    }


def compare(task_name: str = "x_plus_1", n_samples: int = 100, device: str = "cpu"):
    print("\n" + "=" * 60)
    print("RUNNING RL vs RANDOM SEARCH")
    print("=" * 60)

    set_seed(42)

    dataset = load_dataset(task_name, num_samples=n_samples)
    X, y = dataset["X"], dataset["y"]
    num_vars = dataset["num_variables"]
    target = dataset["target"]

    grammar = build_grammar(num_variables=num_vars)

    print(f"\nTask: {task_name}")
    print(f"Target: {target}")
    print(f"Num variables: {num_vars}")
    print(f"Samples: {X.shape[0]}")
    print(f"Grammar size: {len(grammar)}")

    print("\n--- RANDOM SEARCH ---")
    random_results = run_random(grammar, X, y, num_trials=1000)
    print(f"Best reward: {random_results['reward']:.6f}")
    print(f"Best NMSE: {random_results['nmse']:.6f}")
    print(f"Best expr tokens: {random_results['expr_tokens']}")
    print(f"Best expr infix: {random_results['expr_infix']}")

    print("\n--- RL TRAINING ---")
    rl_results = run_rl(
        X=X,
        y=y,
        num_variables=num_vars,
        device=device,
        num_episodes=1000,
    )
    print(f"Best reward: {rl_results['reward']:.6f}")
    print(f"Best expr tokens: {rl_results['expr_tokens']}")
    print(f"Best expr infix: {rl_results['expr_infix']}")

    print("\n" + "=" * 60)
    print("COMPARISON RESULT")
    print("=" * 60)

    if rl_results["reward"] > random_results["reward"]:
        winner = "RL"
    elif rl_results["reward"] < random_results["reward"]:
        winner = "RANDOM"
    else:
        winner = "TIE"

    print(f"Winner: {winner}")
    print("=" * 60)

    return {
        "task_name": task_name,
        "target": target,
        "random": random_results,
        "rl": rl_results,
        "winner": winner,
    }


def main():
    compare(task_name="x_plus_1", n_samples=100, device="cpu")


if __name__ == "__main__":
    main()