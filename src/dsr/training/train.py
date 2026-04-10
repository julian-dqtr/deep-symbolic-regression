import argparse
import os
import torch
import numpy as np
import random
import csv
from datetime import datetime

from ..data.datasets import get_task_suite
from .trainer import Trainer
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_on_suite(suite_name: str, args):
    print(f"\n{'='*80}")
    print(f"Starting Training on {suite_name.upper()} suite")
    print(f"{'='*80}\n")

    tasks = get_task_suite(name=suite_name, num_samples=args.num_samples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    os.makedirs("results", exist_ok=True)
    results_file = (
        f"results/results_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    with open(results_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_name", "best_train_reward", "best_train_nmse", "best_train_expr",
        ])

    print(f"Metrics will be saved to: {results_file}\n")

    for idx, task in enumerate(tasks):
        print(f"--- Task {idx+1}/{len(tasks)}: {task.name} ---")
        X, y = task.generate()

        trainer = Trainer(
            X=X,
            y=y,
            num_variables=task.num_variables,
            device=device,
            optimizer_name="rspg",
        )
        trainer.num_episodes  = args.num_episodes
        trainer.batch_size    = 256
        trainer.learning_rate = args.learning_rate
        trainer.entropy_weight = args.entropy_weight

        # Rebuild Adam with the correct lr (Trainer.__init__ uses config default)
        trainer.rl_optimizer.optimizer = torch.optim.Adam(
            trainer.policy.parameters(), lr=args.learning_rate
        )

        results      = trainer.train()
        best_reward  = results["best_reward"]
        best_episode = results["best_episode"]

        print(f"\nTraining finished for {task.name}. Best reward: {best_reward:.6f}")

        # Default values in case no valid episode was found
        train_nmse = ""
        train_expr = ""

        if best_episode is not None:
            grammar      = trainer.grammar
            evaluator    = PrefixEvaluator(grammar)
            eval_result  = evaluator.evaluate(best_episode["tokens"], X, y)
            train_nmse   = eval_result["nmse"]
            train_expr   = safe_prefix_to_infix(
                best_episode["tokens"], grammar,
                eval_result.get("optimized_constants", []),
            )
            print(f"  Best expression : {train_expr}")
            print(f"  Best NMSE       : {train_nmse:.6f}")
            print(f"  Is valid        : {eval_result['is_valid']}")
        else:
            print("  No valid episode found.")

        with open(results_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([task.name, best_reward, train_nmse, train_expr])

        print()


def main():
    parser = argparse.ArgumentParser(
        description="Train RSPG on a Feynman task suite."
    )
    parser.add_argument(
        "--suite", type=str, default="pmlb_feynman_all",
        choices=["nguyen", "nguyen_univariate", "nguyen_bivariate", "pmlb_feynman_subset", "pmlb_feynman_all"],
    )
    parser.add_argument(
        "--num_episodes", type=int, default=3000,
        help="Training episodes per task (default: 3000)",
    )
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4,
                        help="Adam learning rate (Optuna best: 3.35e-4)")
    parser.add_argument("--entropy_weight", type=float, default=0.017,
                        help="Entropy regularisation weight (Optuna best: 0.017)")

    args = parser.parse_args()
    set_seed()
    train_on_suite(args.suite, args)


if __name__ == "__main__":
    main()