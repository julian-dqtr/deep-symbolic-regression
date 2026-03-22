import heapq
import random
from typing import Dict, List

import numpy as np

from ..config import ENV_CONFIG
from ..core.evaluator import PrefixEvaluator
from ..core.expression import is_complete_prefix, safe_prefix_to_infix


def sample_random_prefix_expression(grammar, max_length: int) -> List[str]:
    """
    Sample a random prefix expression by following the pending_slots mechanism.
    """
    tokens = []
    pending_slots = 1

    while pending_slots > 0 and len(tokens) < max_length:
        token = random.choice(grammar.action_space)
        tokens.append(token)
        pending_slots = pending_slots - 1 + grammar.arity[token]

        if pending_slots < 0:
            break

    return tokens


def evaluate_candidate(tokens, grammar, evaluator, X, y) -> Dict:
    result = evaluator.evaluate(tokens=tokens, X=X, y=y)

    if result["is_valid"]:
        reward = -result["nmse"] - ENV_CONFIG["complexity_penalty"] * len(tokens)
    else:
        reward = ENV_CONFIG["invalid_reward"]

    return {
        "tokens": list(tokens),
        "infix": safe_prefix_to_infix(tokens, grammar),
        "reward": float(reward),
        "nmse": float(result["nmse"]),
        "is_valid": bool(result["is_valid"]),
        "complexity": len(tokens),
        "is_complete": is_complete_prefix(tokens, grammar),
    }


def random_search(
    grammar,
    X: np.ndarray,
    y: np.ndarray,
    num_trials: int = 1000,
    top_k: int = 5,
    max_length: int | None = None,
):
    if max_length is None:
        max_length = ENV_CONFIG["max_length"]

    evaluator = PrefixEvaluator(grammar)

    best_heap = []
    valid_count = 0
    complete_count = 0
    unique_id = 0

    for _ in range(num_trials):
        tokens = sample_random_prefix_expression(grammar=grammar, max_length=max_length)

        candidate = evaluate_candidate(
            tokens=tokens,
            grammar=grammar,
            evaluator=evaluator,
            X=X,
            y=y,
        )

        if candidate["is_valid"]:
            valid_count += 1
        if candidate["is_complete"]:
            complete_count += 1

        score = candidate["reward"]
        heap_item = (score, unique_id, candidate)
        unique_id += 1

        if len(best_heap) < top_k:
            heapq.heappush(best_heap, heap_item)
        else:
            if score > best_heap[0][0]:
                heapq.heapreplace(best_heap, heap_item)

    best_candidates = [
        item[2] for item in sorted(best_heap, key=lambda x: x[0], reverse=True)
    ]

    return {
        "best_candidates": best_candidates,
        "best_reward": best_candidates[0]["reward"] if best_candidates else float("-inf"),
        "valid_count": valid_count,
        "complete_count": complete_count,
        "num_trials": num_trials,
    }


def print_random_search_summary(results: Dict):
    print("\n--- Random Search Summary ---")
    print(f"Trials: {results['num_trials']}")
    print(f"Valid expressions: {results['valid_count']}")
    print(f"Complete expressions: {results['complete_count']}")
    print(f"Best reward: {results['best_reward']:.6f}")

    if results["best_candidates"]:
        print("\nTop candidates:")
        for rank, cand in enumerate(results["best_candidates"], start=1):
            print(f"Rank {rank}")
            print(f"  Reward: {cand['reward']:.6f}")
            print(f"  NMSE: {cand['nmse']:.6f}")
            print(f"  Valid: {cand['is_valid']}")
            print(f"  Complete: {cand['is_complete']}")
            print(f"  Complexity: {cand['complexity']}")
            print(f"  Tokens: {cand['tokens']}")
            print(f"  Expression: {cand['infix']}")
            print("  " + "-" * 40)