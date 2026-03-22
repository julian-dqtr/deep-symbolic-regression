import random
import numpy as np
import torch

from ..benchmarks.datasets import get_task_suite
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..analysis.visualizer import (
    ASTVisualizer,
    plot_training_history,
    plot_target_vs_prediction,
)
from ..training.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_task(
    suite_name: str = "toy",
    task_index: int = 1,
    num_samples: int = 100,
    num_episodes: int = 1000,
    device: str = "cpu",
    optimizer_name: str = "ppo",
    show_plots: bool = True,
):
    tasks = get_task_suite(name=suite_name, num_samples=num_samples)

    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(
            f"task_index={task_index} is out of range for suite '{suite_name}' "
            f"(available: 0..{len(tasks)-1})"
        )

    task = tasks[task_index]
    X, y = task.generate()

    print("\n" + "=" * 80)
    print(f"Task suite: {suite_name}")
    print(f"Task name: {task.name}")
    print(f"Target expression: {task.target_expression}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Num variables: {task.num_variables}")
    print(f"Num samples: {num_samples}")
    print("=" * 80)

    trainer = Trainer(
        X=X,
        y=y,
        num_variables=task.num_variables,
        device=device,
        optimizer_name=optimizer_name,
    )
    trainer.num_episodes = num_episodes

    results = trainer.train()

    best_reward = results["best_reward"]
    best_episode = results["best_episode"]

    print("\nTraining finished.")
    print(f"Best reward: {best_reward:.6f}")

    if best_episode is None:
        print("No best episode found.")
        return results

    best_tokens = best_episode["tokens"]
    grammar = trainer.grammar
    evaluator = PrefixEvaluator(grammar)

    best_expr_str = safe_prefix_to_infix(best_tokens, grammar)
    eval_result = evaluator.evaluate(best_tokens, X, y)

    print("Best tokens:", best_tokens)
    print("Best expression:", best_expr_str)
    print("Best complexity:", len(best_tokens))
    print("Best NMSE:", eval_result["nmse"])
    print("Is valid:", eval_result["is_valid"])

    if show_plots:
        plot_training_history(
            history=results["history"],
            title=f"Training History - {task.name} ({optimizer_name})",
            show=True,
        )

        plot_target_vs_prediction(
            grammar=grammar,
            evaluator=evaluator,
            tokens=best_tokens,
            X=X,
            y=y,
            title=f"Target vs Prediction - {task.name}",
            show=True,
        )

        viz = ASTVisualizer()
        viz.draw_tree(
            tokens=best_tokens,
            grammar=grammar,
            title=f"AST - {task.name}: {best_expr_str}",
            show=True,
        )

    return {
        "task": task,
        "X": X,
        "y": y,
        "trainer": trainer,
        "results": results,
        "best_expression": best_expr_str,
        "evaluation": eval_result,
    }


def main():
    set_seed(42)

    run_single_task(
        suite_name="toy",
        task_index=1,
        num_samples=100,
        num_episodes=1000,
        device="cuda",
        optimizer_name="ppo",
        show_plots=True,
    )


if __name__ == "__main__":
    main()