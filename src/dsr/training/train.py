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
    print(f"\\n{'='*80}")
    print(f"Starting Training on {suite_name.upper()} suite")
    print(f"{'='*80}\\n")
    
    tasks = get_task_suite(name=suite_name, num_samples=args.num_samples)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    os.makedirs("results", exist_ok=True)
    results_file = f"results/results_{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(results_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "task_name", "best_train_reward", "best_train_nmse", "best_train_expr", 
            "best_beam_reward", "best_beam_nmse", "best_beam_expr"
        ])
        
    print(f"Metrics will be safely saved to: {results_file}\n")
    
    for idx, task in enumerate(tasks):
        print(f"--- Task {idx+1}/{len(tasks)}: {task.name} ---")
        X, y = task.generate()
        
        trainer = Trainer(
            X=X,
            y=y,
            num_variables=task.num_variables,
            device=device,
            optimizer_name="rspg"
        )
        trainer.num_episodes = args.num_episodes
        trainer.batch_size = 256
        trainer.learning_rate = args.learning_rate
        trainer.entropy_weight = args.entropy_weight
        
        trainer.optimizer = torch.optim.Adam(
            trainer.policy.parameters(), 
            lr=args.learning_rate
        )
        
        results = trainer.train()
        
        best_reward = results["best_reward"]
        best_episode = results["best_episode"]
        
        print(f"\\nTraining finished for {task.name}. Best reward: {best_reward:.6f}")
        
        if best_episode is not None:
            best_tokens = best_episode["tokens"]
            grammar = trainer.grammar
            evaluator = PrefixEvaluator(grammar)
            
            eval_result = evaluator.evaluate(best_tokens, X, y)
            best_expr_str = safe_prefix_to_infix(best_tokens, grammar, eval_result.get("optimized_constants", []))
            
            print("Best expression:", best_expr_str)
            print("Best NMSE:", eval_result["nmse"])
            print("Is valid:", eval_result["is_valid"])
        else:
            print("No valid episode found.")
            
        
        # Beam search is intentionally disabled: Top-K Risk-Seeking sampling outperforms it.

        with open(results_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            train_r = best_reward
            train_nmse = eval_result["nmse"] if best_episode else ""
            train_e = best_expr_str if best_episode else ""
            
            beam_r = ""
            beam_nmse = ""
            beam_e = ""
            
            writer.writerow([task.name, train_r, train_nmse, train_e, beam_r, beam_nmse, beam_e])

        print("\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=str, default="pmlb_feynman_all", choices=["pmlb_feynman_all", "pmlb_feynman_subset"])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.000335)
    parser.add_argument("--entropy_weight", type=float, default=0.017)
    parser.add_argument("--beam_width", type=int, default=10)
    
    args = parser.parse_args()
    set_seed()
    
    train_on_suite(args.suite, args)

if __name__ == "__main__":
    main()
