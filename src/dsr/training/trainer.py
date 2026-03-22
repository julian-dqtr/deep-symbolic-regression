import numpy as np

from ..config import TRAINING_CONFIG
from ..core.factory import build_grammar
from ..core.env import SymbolicRegressionEnv
from ..core.expression import safe_prefix_to_infix
from ..core.evaluator import PrefixEvaluator
from ..models.policy import SymbolicPolicy
from ..analysis.memory import TopKMemory
from .rollout import collect_episode
from .policy_optimizer import ReinforceOptimizer
from .ppo_optimizer import PPOOptimizer


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

        if optimizer_name.lower() == "ppo":
            self.rl_optimizer = PPOOptimizer(self.policy)
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

    def _update_memory(self, tokens, source="sampling"):
        eval_result = self.evaluator.evaluate(tokens=tokens, X=self.X, y=self.y)
        reward = -eval_result["nmse"] - 0.01 * len(tokens) if eval_result["is_valid"] else -1.0
        infix = safe_prefix_to_infix(tokens, self.grammar)

        self.memory.add(
            tokens=tokens,
            infix=infix,
            reward=reward,
            nmse=eval_result["nmse"],
            complexity=len(tokens),
            source=source,
        )

    def train(self):
        for episode_idx in range(1, self.num_episodes + 1):
            episode = collect_episode(
                env=self.env,
                policy=self.policy,
                grammar=self.grammar,
                device=self.device,
            )

            stats = self.rl_optimizer.update(episode)

            self.history["loss"].append(stats["loss"])
            self.history["policy_loss"].append(stats["policy_loss"])
            self.history["value_loss"].append(stats.get("value_loss", 0.0))
            self.history["entropy"].append(stats["entropy"])
            self.history["final_reward"].append(stats["final_reward"])
            self.history["episode_length"].append(len(episode["tokens"]))

            self._update_memory(episode["tokens"], source="sampling")

            if stats["final_reward"] > self.best_reward:
                self.best_reward = stats["final_reward"]
                self.best_episode = episode
                print("New best expression:", episode["tokens"])

            if episode_idx % 100 == 0 or episode_idx == 1:
                print(
                    f"[Episode {episode_idx}/{self.num_episodes}] "
                    f"optimizer={self.optimizer_name} | "
                    f"reward={stats['final_reward']:.4f} | "
                    f"best={self.best_reward:.4f} | "
                    f"len={len(episode['tokens'])} | "
                    f"entropy={stats['entropy']:.4f}"
                )

        return {
            "history": self.history,
            "best_reward": self.best_reward,
            "best_episode": self.best_episode,
            "memory": self.memory,
        }