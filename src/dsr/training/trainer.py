import torch
import numpy as np

from ..config import TRAINING_CONFIG
from ..core.factory import build_grammar
from ..core.env import SymbolicRegressionEnv
from ..core.expression import safe_prefix_to_infix
from ..core.evaluator import PrefixEvaluator
from ..models.policy import SymbolicPolicy
from ..analysis.memory import TopKMemory
from ..models.policy import SymbolicPolicy
from ..analysis.memory import TopKMemory
from .rollout import collect_episode, recompute_episode, collect_batched_episodes
from .policy_optimizer import ReinforceOptimizer
from .ppo_optimizer import PPOOptimizer
from .risk_seeking_optimizer import RiskSeekingOptimizer


def normalize_device(device: str) -> str:
    device = device.lower().strip()

    if device == "gpu":
        device = "cuda"

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}. Use 'cpu' or 'cuda'.")

    return device


class Trainer:
    def __init__(self, X, y, num_variables: int, device: str = "cpu", optimizer_name: str = "reinforce"):
        self.device = normalize_device(device)

        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)

        self.grammar = build_grammar(num_variables=num_variables)
        self.env = SymbolicRegressionEnv(self.X, self.y, self.grammar)
        self.evaluator = PrefixEvaluator(self.grammar)

        self.policy = SymbolicPolicy(vocab_size=len(self.grammar)).to(self.device)

        self.tensor_X = torch.tensor(self.X, dtype=torch.float32, device=self.device)
        self.tensor_y = torch.tensor(self.y, dtype=torch.float32, device=self.device)
        self.policy.set_dataset_embedding(self.tensor_X, self.tensor_y)

        if optimizer_name.lower() == "ppo":
            self.rl_optimizer = PPOOptimizer(self.policy)
        elif optimizer_name.lower() == "rspg":
            self.rl_optimizer = RiskSeekingOptimizer(self.policy)
        else:
            self.rl_optimizer = ReinforceOptimizer(self.policy)

        self.optimizer_name = optimizer_name.lower()
        self.num_episodes = TRAINING_CONFIG["num_episodes"]

        self.history = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "final_reward": [],
            "episode_length": [],
        }

        self.best_reward = float("-inf")
        self.best_episode = None
        self.memory = TopKMemory(capacity=20)

    def train(self):
        episode_idx = 0
        while episode_idx < self.num_episodes:
            current_batch_size = min(self.batch_size, self.num_episodes - episode_idx)
            
            batch_episodes = collect_batched_episodes(
                env_template=self.env,
                policy=self.policy,
                grammar=self.grammar,
                batch_size=current_batch_size,
                max_length=30,
                device=self.device
            )
            
            for episode in batch_episodes:
                eval_result = self.evaluator.evaluate(tokens=episode["tokens"], X=self.X, y=self.y)
                reward = -eval_result["nmse"] - 0.01 * len(episode["tokens"]) if eval_result["is_valid"] else -1.0
                
                L = len(episode["tokens"])
                episode["rewards"] = [0.0] * L
                if L > 0:
                    episode["rewards"][-1] = reward
                episode["final_reward"] = reward
                
                infix = safe_prefix_to_infix(episode["tokens"], self.grammar, eval_result.get("optimized_constants", []))
                
                self.memory.add(
                    tokens=episode["tokens"],
                    infix=infix,
                    reward=reward,
                    nmse=eval_result["nmse"],
                    complexity=L,
                    source="sampling",
                )

                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_episode = episode
                    print("New best expression:", episode["tokens"])
                    
                episode_idx += 1

            # Inject Top-K Memory Episodes 
            memory_items = self.memory.to_rows()
            memory_episodes = []
            
            # Recompute gradients natively for the historical memories using the current network parameters 
            for item in memory_items:
                try:
                    ep = recompute_episode(
                        env=self.env,
                        policy=self.policy,
                        grammar=self.grammar,
                        tokens=item["tokens"],
                        device=self.device,
                    )
                    memory_episodes.append(ep)
                except Exception as e:
                    pass # Safety net in case an old sequence causes an unexpected PyTorch grid error 

            if self.optimizer_name == "rspg":
                stats = self.rl_optimizer.update(batch_episodes, memory_episodes=memory_episodes)
            else:
                stats = self.rl_optimizer.update(batch_episodes)

            # Record stats
            self.history["loss"].append(stats["loss"])
            self.history["policy_loss"].append(stats["policy_loss"])
            self.history["value_loss"].append(stats.get("value_loss", 0.0))
            self.history["entropy"].append(stats["entropy"])
            self.history["final_reward"].append(stats["final_reward"])

            if episode_idx % max(100, self.batch_size) == 0 or episode_idx == current_batch_size:
                print(
                    f"[Episode {episode_idx}/{self.num_episodes}] "
                    f"optimizer={self.optimizer_name} | "
                    f"batch_reward={stats['final_reward']:.4f} | "
                    f"best={self.best_reward:.4f} | "
                    f"entropy={stats['entropy']:.4f}"
                )

        return {
            "history": self.history,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "memory": self.memory,
        }