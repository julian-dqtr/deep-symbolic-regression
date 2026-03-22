import random
import numpy as np
import torch

from ..benchmarks.datasets import get_task_suite
from ..core.factory import build_grammar
from ..analysis.visualizer import print_top_candidates
from ..baselines.random_search import random_search, print_random_search_summary


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_random_search(
    suite_name: str = "toy",
    task_index: int = 1,
    num_samples: int = 100,
    num_trials: int = 1000,
    top_k: int = 5,
):
    print("Loading task suite...")
    tasks = get_task_suite(name=suite_name, num_samples=num_samples)
    print(f"Loaded {len(tasks)} tasks")

    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(
            f"task_index={task_index} is out of range for suite '{suite_name}' "
            f"(available: 0..{len(tasks)-1})"
        )

    task = tasks[task_index]
    print(f"Selected task: {task.name}")

    X, y = task.generate()
    print(f"Generated dataset: X shape={X.shape}, y shape={y.shape}")

    grammar = build_grammar(num_variables=task.num_variables)
    print(f"Built grammar with vocab size={len(grammar)}")

    print("\n" + "=" * 80)
    print("Running Random Search Baseline")
    print(f"Task suite: {suite_name}")
    print(f"Task name: {task.name}")
    print(f"Target expression: {task.target_expression}")
    print(f"Num variables: {task.num_variables}")
    print(f"Num samples: {num_samples}")
    print(f"Num trials: {num_trials}")
    print("=" * 80)

    results = random_search(
        grammar=grammar,
        X=X,
        y=y,
        num_trials=num_trials,
        top_k=top_k,
    )

    print("Random search finished")
    print_random_search_summary(results)

    print("\nDetailed top candidates:")
    print_top_candidates(results["best_candidates"])

    return {
        "task": task,
        "X": X,
        "y": y,
        "grammar": grammar,
        "results": results,
    }


def main():
    print("run_random_search.py started")
    set_seed(42)
    print("Seed set")

    run_single_random_search(
        suite_name="toy",
        task_index=1,
        num_samples=100,
        num_trials=1000,
        top_k=5,
    )

    print("Script completed successfully")


if __name__ == "__main__":
    main()