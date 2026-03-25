import optuna
import numpy as np
import torch

from ..benchmarks.datasets import get_task_by_name
from ..training.trainer import Trainer
from ..core.evaluator import PrefixEvaluator

def objective(trial):
    # Hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    entropy_weight = trial.suggest_float("entropy_weight", 0.001, 0.1, log=True)
    
    # Increase to a higher value for better performance (e.g., 5000-10000)
    num_episodes = 5000 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # We test on 4 equations of varying difficulty
    # Easy: nguyen_x_plus_1 (equivalent to polynomial)
    # Medium: feynman_I_8_14 (distance, velocity)
    # Hard: feynman_I_10_7 (mass, velocity, etc.)
    # Very Hard: feynman_I_12_1 (force components)
    tasks_to_test = [
        "nguyen_x_plus_1",
        "feynman_I_8_14", 
        "feynman_I_10_7",
        "feynman_I_12_1"
    ]
    
    total_best_reward = 0.0
    
    for task_name in tasks_to_test:
        task = get_task_by_name(task_name, num_samples=100)
        X, y = task.generate()
        
        trainer = Trainer(
            X=X,
            y=y,
            num_variables=task.num_variables,
            device=device,
            optimizer_name="rspg"
        )
        
        # Override hyperparameters
        trainer.learning_rate = learning_rate
        trainer.entropy_weight = entropy_weight
        trainer.num_episodes = num_episodes
        trainer.batch_size = 256
        
        # Initialize optimizers with new learning rate
        trainer.optimizer = torch.optim.Adam(
            trainer.policy.parameters(), 
            lr=learning_rate
        )
        
        results = trainer.train()
        total_best_reward += results["best_reward"]
        
    avg_reward = total_best_reward / len(tasks_to_test)
    return avg_reward

def main():
    print("Starting Optuna Study for Hyperparameter Optimization...")
    study = optuna.create_study(direction="maximize")
    
    # We set 50 trials as a good starting point for real optimization
    # n_jobs=-1 will utilize all 20 CPU cores for massive parallelization
    study.optimize(objective, n_trials=50, n_jobs=-1)
    
    print("\\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Average Best Reward): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

if __name__ == "__main__":
    main()
