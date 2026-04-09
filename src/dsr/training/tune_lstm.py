"""
tune_lstm.py
============
Hyperparameter study for the LSTM policy network using Optuna.

Search space
------------
  hidden_dim            : {128, 256, 512}          (1024 removed — OOM on ≤6GB GPU)
  num_lstm_layers       : {1, 2, 3}
  token_embedding_dim   : {32, 64, 128}
  dataset_embedding_dim : {32, 64, 128}

Usage
-----
python -m dsr.training.tune_lstm \
    --tasks feynman_I_8_14 feynman_I_10_7 feynman_I_12_11 \
    --n_trials 20 --num_episodes 2000

python -m dsr.training.tune_lstm \
    --suite pmlb_feynman_subset \
    --n_trials 50 --num_episodes 3000

Output
------
results/
  lstm_tuning_<timestamp>.csv
  plots/
    lstm_importance.png
    lstm_parallel.png
    lstm_history.png
"""

import argparse
import csv
import os
import random
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise ImportError("Install optuna first:  pip install optuna")

from ..data.datasets import get_task_suite, get_task_by_name
from ..core.factory import build_grammar
from ..core.env import SymbolicRegressionEnv
from ..core.evaluator import PrefixEvaluator
from ..core.expression import safe_prefix_to_infix
from ..models.policy import SymbolicPolicy
from ..analysis.memory import TopKMemory
from .rollout import recompute_episode, collect_batched_episodes
from .risk_seeking_optimizer import RiskSeekingOptimizer


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Patched SymbolicPolicy with configurable architecture
# ---------------------------------------------------------------------------

class ConfigurablePolicy(SymbolicPolicy):
    """
    SymbolicPolicy with architecture dimensions passed explicitly,
    bypassing the global MODEL_CONFIG so Optuna can vary them freely.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_lstm_layers: int,
        token_embedding_dim: int,
        dataset_embedding_dim: int,
    ):
        nn.Module.__init__(self)

        self.vocab_size            = vocab_size
        self.hidden_dim            = hidden_dim
        self.num_lstm_layers       = num_lstm_layers
        self.token_embedding_dim   = token_embedding_dim
        self.dataset_embedding_dim = dataset_embedding_dim

        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=token_embedding_dim,
        )

        self.sequence_encoder = nn.LSTM(
            input_size=token_embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )

        self.dataset_encoder = None

        self.state_mlp = nn.Sequential(
            nn.Linear(hidden_dim + dataset_embedding_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.action_head = nn.Linear(hidden_dim, vocab_size)
        self.value_head  = nn.Linear(hidden_dim, 1)
        self.cached_dataset_embedding = None


# ---------------------------------------------------------------------------
# Single trial run
# ---------------------------------------------------------------------------

def run_trial_config(
    X, y, num_variables,
    hidden_dim, num_lstm_layers,
    token_embedding_dim, dataset_embedding_dim,
    learning_rate, entropy_weight,
    num_episodes, batch_size,
    device, seed,
) -> float:
    set_seed(seed)

    grammar   = build_grammar(num_variables=num_variables)
    env       = SymbolicRegressionEnv(X, y, grammar)
    evaluator = PrefixEvaluator(grammar)

    policy = ConfigurablePolicy(
        vocab_size=len(grammar),
        hidden_dim=hidden_dim,
        num_lstm_layers=num_lstm_layers,
        token_embedding_dim=token_embedding_dim,
        dataset_embedding_dim=dataset_embedding_dim,
    ).to(device)

    tensor_X = torch.tensor(X, dtype=torch.float32, device=device)
    tensor_y = torch.tensor(y, dtype=torch.float32, device=device)
    policy.set_dataset_embedding(tensor_X, tensor_y)

    rl_optimizer = RiskSeekingOptimizer(policy)
    rl_optimizer.learning_rate  = learning_rate
    rl_optimizer.entropy_weight = entropy_weight
    rl_optimizer.optimizer = torch.optim.Adam(
        policy.parameters(), lr=learning_rate
    )

    memory      = TopKMemory(capacity=20)
    best_reward = float("-inf")
    episode_idx = 0

    while episode_idx < num_episodes:
        current_batch = min(batch_size, num_episodes - episode_idx)

        batch_episodes = collect_batched_episodes(
            env_template=env,
            policy=policy,
            grammar=grammar,
            batch_size=current_batch,
            max_length=30,
            device=device,
        )

        for episode in batch_episodes:
            eval_result = evaluator.evaluate(tokens=episode["tokens"], X=X, y=y)
            reward = (
                -eval_result["nmse"] - 0.01 * len(episode["tokens"])
                if eval_result["is_valid"] else -1.0
            )
            L = len(episode["tokens"])
            episode["rewards"]      = [0.0] * L
            if L > 0:
                episode["rewards"][-1] = reward
            episode["final_reward"] = reward

            infix = safe_prefix_to_infix(
                episode["tokens"], grammar,
                eval_result.get("optimized_constants", []),
            )
            memory.add(
                tokens=episode["tokens"], infix=infix,
                reward=reward, nmse=eval_result["nmse"],
                complexity=L, source="sampling",
            )
            if reward > best_reward:
                best_reward = reward
            episode_idx += 1

        memory_episodes = []
        for item in memory.to_rows():
            try:
                ep = recompute_episode(
                    env=env, policy=policy, grammar=grammar,
                    tokens=item["tokens"], device=device,
                )
                memory_episodes.append(ep)
            except Exception:
                pass

        rl_optimizer.update(batch_episodes, memory_episodes=memory_episodes)

    return best_reward


# ---------------------------------------------------------------------------
# Optuna objective — CUDA OOM caught and pruned cleanly
# ---------------------------------------------------------------------------

def make_objective(tasks_data, args, device):
    def objective(trial: optuna.Trial) -> float:
        # 1024 hidden_dim removed — causes CUDA OOM on GPUs ≤ 6GB
        hidden_dim            = trial.suggest_categorical("hidden_dim",            [128, 256, 512])
        num_lstm_layers       = trial.suggest_categorical("num_lstm_layers",       [1, 2, 3])
        token_embedding_dim   = trial.suggest_categorical("token_embedding_dim",   [32, 64, 128])
        dataset_embedding_dim = trial.suggest_categorical("dataset_embedding_dim", [32, 64, 128])

        total_reward = 0.0
        for X, y, num_variables in tasks_data:
            try:
                reward = run_trial_config(
                    X=X, y=y,
                    num_variables=num_variables,
                    hidden_dim=hidden_dim,
                    num_lstm_layers=num_lstm_layers,
                    token_embedding_dim=token_embedding_dim,
                    dataset_embedding_dim=dataset_embedding_dim,
                    learning_rate=args.learning_rate,
                    entropy_weight=args.entropy_weight,
                    num_episodes=args.num_episodes,
                    batch_size=args.batch_size,
                    device=device,
                    seed=args.seed,
                )
                total_reward += reward

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Free GPU cache and skip this trial cleanly
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    raise optuna.exceptions.TrialPruned(
                        f"CUDA OOM — hidden_dim={hidden_dim}, "
                        f"num_lstm_layers={num_lstm_layers}"
                    )
                raise  # re-raise unexpected RuntimeErrors

        return total_reward / len(tasks_data)

    return objective


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_history(study, out_dir):
    values      = [t.value for t in study.trials if t.value is not None]
    best_so_far = np.maximum.accumulate(values)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(range(len(values)), values,
               color="#B5D4F4", s=25, zorder=2, label="Trial reward")
    ax.plot(range(len(best_so_far)), best_so_far,
            color="#D85A30", lw=2, zorder=3, label="Best so far")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Mean best reward")
    ax.set_title("LSTM hyperparameter search — history", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, "lstm_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


def plot_importance(study, out_dir):
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception:
        print("  [skip] importance — not enough completed trials")
        return

    params = list(importance.keys())
    values = [importance[p] for p in params]
    label_map = {
        "hidden_dim":            "Hidden dim",
        "num_lstm_layers":       "LSTM layers",
        "token_embedding_dim":   "Token emb. dim",
        "dataset_embedding_dim": "Dataset emb. dim",
    }
    labels = [label_map.get(p, p) for p in params]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color="#378ADD", alpha=0.85)
    ax.set_xlabel("Relative importance")
    ax.set_title("LSTM hyperparameter importance", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "lstm_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [plot] saved → {path}")


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "trial", "hidden_dim", "num_lstm_layers",
    "token_embedding_dim", "dataset_embedding_dim",
    "mean_best_reward", "status", "duration_sec",
]


def save_csv(study, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    completed = [t for t in study.trials if t.value is not None]
    completed.sort(key=lambda t: -(t.value or -9999))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for t in study.trials:
            duration = (
                (t.datetime_complete - t.datetime_start).total_seconds()
                if t.datetime_complete and t.datetime_start else ""
            )
            status = t.state.name  # COMPLETE / PRUNED / FAIL
            writer.writerow({
                "trial":                t.number,
                "hidden_dim":           t.params.get("hidden_dim", ""),
                "num_lstm_layers":      t.params.get("num_lstm_layers", ""),
                "token_embedding_dim":  t.params.get("token_embedding_dim", ""),
                "dataset_embedding_dim":t.params.get("dataset_embedding_dim", ""),
                "mean_best_reward":     f"{t.value:.6f}" if t.value is not None else "",
                "status":               status,
                "duration_sec":         f"{duration:.1f}" if duration else "",
            })
    print(f"CSV saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LSTM hyperparameter search with Optuna."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--tasks", nargs="+", default=None, metavar="TASK")
    src.add_argument("--suite", type=str, default="pmlb_feynman_subset",
                     choices=["pmlb_feynman_subset", "pmlb_feynman_all"])

    parser.add_argument("--n_trials",       type=int,   default=20)
    parser.add_argument("--num_episodes",   type=int,   default=2000)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--num_samples",    type=int,   default=100)
    parser.add_argument("--learning_rate",  type=float, default=3.35e-4)
    parser.add_argument("--entropy_weight", type=float, default=0.017)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--n_jobs",         type=int,   default=1)

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  n_trials: {args.n_trials}")

    if args.tasks:
        tasks = [get_task_by_name(n, num_samples=args.num_samples) for n in args.tasks]
    else:
        tasks = get_task_suite(name=args.suite, num_samples=args.num_samples)

    tasks_data = []
    for task in tasks:
        X, y = task.generate()
        tasks_data.append((X, y, task.num_variables))

    print(f"Tasks: {[t.name for t in tasks]}")

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results"
    plots_dir   = os.path.join(results_dir, "plots")
    csv_path    = os.path.join(results_dir, f"lstm_tuning_{timestamp}.csv")
    os.makedirs(plots_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        make_objective(tasks_data, args, device),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        # Catch pruned trials gracefully without stopping the study
        catch=(Exception,),
    )

    # Results
    completed = [t for t in study.trials if t.value is not None]
    if not completed:
        print("No completed trials. Try reducing num_episodes or batch_size.")
        return

    best = study.best_trial
    print("\n" + "=" * 60)
    print("BEST LSTM CONFIGURATION")
    print("=" * 60)
    print(f"  Mean best reward      : {best.value:.4f}")
    print(f"  hidden_dim            : {best.params['hidden_dim']}")
    print(f"  num_lstm_layers       : {best.params['num_lstm_layers']}")
    print(f"  token_embedding_dim   : {best.params['token_embedding_dim']}")
    print(f"  dataset_embedding_dim : {best.params['dataset_embedding_dim']}")
    print()
    print("Paste into MODEL_CONFIG in config.py:")
    print(f"  MODEL_CONFIG = {{")
    print(f"      'token_embedding_dim':   {best.params['token_embedding_dim']},")
    print(f"      'hidden_dim':            {best.params['hidden_dim']},")
    print(f"      'dataset_embedding_dim': {best.params['dataset_embedding_dim']},")
    print(f"      'num_lstm_layers':       {best.params['num_lstm_layers']},")
    print(f"  }}")
    print("=" * 60)

    pruned = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED)
    failed = sum(1 for t in study.trials
                 if t.state == optuna.trial.TrialState.FAIL)
    print(f"\nTrials: {len(completed)} completed, {pruned} pruned, {failed} failed")

    save_csv(study, csv_path)
    plot_history(study, plots_dir)
    plot_importance(study, plots_dir)

    print(f"\nCSV   → {csv_path}")
    print(f"Plots → {plots_dir}/")


if __name__ == "__main__":
    main()