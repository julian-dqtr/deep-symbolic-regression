import random
import numpy as np
import torch

from ..benchmarks.datasets import get_task_suite
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..analysis.visualizer import ASTVisualizer, plot_target_vs_prediction, print_top_candidates
from ..baselines.beam_search import beam_search
from ..training.trainer import Trainer
from ..config import ENV_CONFIG


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


def compute_reward_from_eval(eval_result, complexity: int) -> float:
    if not eval_result["is_valid"]:
        return float(ENV_CONFIG["invalid_reward"])

    return float(
        -eval_result["nmse"] - ENV_CONFIG["complexity_penalty"] * complexity
    )


def evaluate_beam_candidates(grammar, X, y, candidates):
    """
    Evaluate raw beam candidates with the real task reward.
    """
    evaluator = PrefixEvaluator(grammar)
    evaluated = []

    for cand in candidates:
        tokens = cand["tokens"]
        eval_result = evaluator.evaluate(tokens=tokens, X=X, y=y)
        reward = compute_reward_from_eval(eval_result, complexity=len(tokens))

        enriched = {
            "tokens": tokens,
            "infix": safe_prefix_to_infix(tokens, grammar, eval_result.get("optimized_constants", [])),
            "logprob": cand["logprob"],
            "complete": cand["complete"],
            "pending_slots": cand["pending_slots"],
            "is_valid": eval_result["is_valid"],
            "nmse": eval_result["nmse"],
            "reward": reward,
            "complexity": len(tokens),
        }
        evaluated.append(enriched)

    return evaluated


def sort_candidates_by_reward(candidates):
    """
    Reward-aware reranking:
    1) valid before invalid
    2) complete before incomplete
    3) higher reward first
    4) shorter complexity preferred if tied
    5) higher logprob as final tiebreaker
    """
    return sorted(
        candidates,
        key=lambda c: (
            c["is_valid"],
            c["complete"],
            c["reward"],
            -c["complexity"],
            c["logprob"],
        ),
        reverse=True,
    )


def print_reward_reranked_candidates(candidates):
    print("\n" + "=" * 80)
    print("REWARD-RERANKED BEAM CANDIDATES")
    print("=" * 80)

    if not candidates:
        print("No candidates.")
        return

    for rank, cand in enumerate(candidates, start=1):
        print(f"Rank {rank}")
        print(f"  Reward: {cand['reward']:.6f}")
        print(f"  NMSE: {cand['nmse']:.6f}")
        print(f"  LogProb: {cand['logprob']:.6f}")
        print(f"  Valid: {cand['is_valid']}")
        print(f"  Complete: {cand['complete']}")
        print(f"  Complexity: {cand['complexity']}")
        print(f"  Tokens: {cand['tokens']}")
        print(f"  Expression: {cand['infix']}")
        print("  " + "-" * 40)


def run_decode_with_beam(
    suite_name: str = "toy",
    task_index: int = 1,
    num_samples: int = 100,
    num_episodes: int = 1000,
    beam_width: int = 10,
    max_length: int = 30,
    optimizer_name: str = "ppo",
    device: str = "cpu",
    show_plots: bool = True,
):
    device = resolve_device(device)

    tasks = get_task_suite(name=suite_name, num_samples=num_samples)

    if task_index < 0 or task_index >= len(tasks):
        raise ValueError(
            f"task_index={task_index} is out of range for suite '{suite_name}' "
            f"(available: 0..{len(tasks)-1})"
        )

    task = tasks[task_index]
    X, y = task.generate()

    print("\n" + "=" * 80)
    print("TRAIN + DECODE WITH BEAM SEARCH")
    print(f"Task suite: {suite_name}")
    print(f"Task name: {task.name}")
    print(f"Target expression: {task.target_expression}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Device: {device}")
    print(f"Beam width: {beam_width}")
    print("=" * 80)

    trainer = Trainer(
        X=X,
        y=y,
        num_variables=task.num_variables,
        device=device,
        optimizer_name=optimizer_name,
    )
    trainer.num_episodes = num_episodes

    train_results = trainer.train()

    best_episode = train_results["best_episode"]
    grammar = trainer.grammar
    policy = trainer.policy
    evaluator = PrefixEvaluator(grammar)

    print("\n" + "=" * 80)
    print("BEST EXPRESSION FROM TRAINING")
    print("=" * 80)

    if best_episode is not None:
        best_tokens = best_episode["tokens"]
        best_eval = evaluator.evaluate(best_tokens, X, y)
        best_expr = safe_prefix_to_infix(best_tokens, grammar, best_eval.get("optimized_constants", []))
        best_reward = train_results["best_reward"]

        print(f"Tokens: {best_tokens}")
        print(f"Expression: {best_expr}")
        print(f"Reward: {best_reward:.6f}")
        print(f"NMSE: {best_eval['nmse']:.6f}")
        print(f"Valid: {best_eval['is_valid']}")
    else:
        best_tokens = None
        print("No best episode found during training.")

    print("\n" + "=" * 80)
    print("RAW BEAM SEARCH DECODING")
    print("=" * 80)

    raw_candidates = beam_search(
        policy=policy,
        grammar=grammar,
        X=X,
        y=y,
        beam_width=beam_width,
        max_length=max_length,
        temperature=1.5,
        device=device,
    )

    evaluated_candidates = evaluate_beam_candidates(
        grammar=grammar,
        X=X,
        y=y,
        candidates=raw_candidates,
    )

    print_top_candidates(evaluated_candidates)

    reranked_candidates = sort_candidates_by_reward(evaluated_candidates)
    print_reward_reranked_candidates(reranked_candidates)

    if len(reranked_candidates) > 0:
        top_beam = reranked_candidates[0]

        print("\n" + "=" * 80)
        print("TOP RERANKED BEAM RESULT")
        print("=" * 80)
        print(f"Tokens: {top_beam['tokens']}")
        print(f"Expression: {top_beam['infix']}")
        print(f"Reward: {top_beam['reward']:.6f}")
        print(f"NMSE: {top_beam['nmse']:.6f}")
        print(f"Valid: {top_beam['is_valid']}")
        print(f"Complete: {top_beam['complete']}")
        print(f"LogProb: {top_beam['logprob']:.6f}")

        if show_plots and top_beam["is_valid"]:
            plot_target_vs_prediction(
                grammar=grammar,
                evaluator=evaluator,
                tokens=top_beam["tokens"],
                X=X,
                y=y,
                title=f"Reranked Beam Prediction - {task.name}",
                show=True,
            )

            viz = ASTVisualizer()
            viz.draw_tree(
                tokens=top_beam["tokens"],
                grammar=grammar,
                title=f"Reranked Beam AST - {task.name}: {top_beam['infix']}",
                show=True,
            )

    print("\n" + "=" * 80)
    print("TRAINING BEST vs RERANKED BEAM BEST")
    print("=" * 80)

    if best_episode is not None and len(reranked_candidates) > 0:
        top_beam = reranked_candidates[0]
        train_best_reward = train_results["best_reward"]
        beam_best_reward = top_beam["reward"]

        print(f"Training best reward: {train_best_reward:.6f}")
        print(f"Reranked beam best reward: {beam_best_reward:.6f}")

        if beam_best_reward > train_best_reward:
            print("Winner: RERANKED BEAM SEARCH")
        elif beam_best_reward < train_best_reward:
            print("Winner: TRAINING SAMPLING")
        else:
            print("Winner: TIE")

    return {
        "task": task,
        "X": X,
        "y": y,
        "trainer": trainer,
        "train_results": train_results,
        "raw_beam_candidates": evaluated_candidates,
        "reranked_beam_candidates": reranked_candidates,
    }


def main():
    set_seed(42)

    run_decode_with_beam(
        suite_name="toy",
        task_index=1,          # x_plus_1
        num_samples=100,
        num_episodes=1000,
        beam_width=50,
        max_length=30,
        optimizer_name="ppo",
        device="cuda",
        show_plots=True,
    )


if __name__ == "__main__":
    main()