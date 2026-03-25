import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from ..config import TRAINING_CONFIG

class RiskSeekingOptimizer:
    def __init__(self, policy, epsilon=0.05):
        self.policy = policy
        self.epsilon = epsilon

        self.learning_rate = TRAINING_CONFIG["learning_rate"]
        self.entropy_weight = TRAINING_CONFIG["entropy_weight"]
        self.grad_clip_norm = TRAINING_CONFIG["grad_clip_norm"]

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def compute_returns(self, rewards):
        final_reward = rewards[-1] if len(rewards) > 0 else 0.0
        returns = [final_reward for _ in rewards]
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, batch_episodes, memory_episodes=None):
        rewards = [ep["final_reward"] for ep in batch_episodes]
        threshold = np.quantile(rewards, 1.0 - self.epsilon)

        elite_episodes = [ep for ep in batch_episodes if ep["final_reward"] >= threshold]

        if not elite_episodes:
            elite_episodes = batch_episodes

        # Inject historical best equations from memory!
        if memory_episodes:
            elite_episodes.extend(memory_episodes)

        all_log_probs = []
        all_entropies = []
        all_returns = []

        for episode in elite_episodes:
            if len(episode["log_probs"]) == 0:
                continue
            device = episode["log_probs"][0].device

            log_probs = torch.stack(episode["log_probs"])
            entropies = torch.stack(episode["entropies"])
            returns = self.compute_returns(episode["rewards"]).to(device)

            all_log_probs.append(log_probs)
            all_entropies.append(entropies)
            all_returns.append(returns)

        if not all_log_probs:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "entropy": 0.0,
                "final_reward": float(np.mean(rewards)),
            }

        log_probs = torch.cat(all_log_probs)
        entropies = torch.cat(all_entropies)
        returns = torch.cat(all_returns)

        # In Risk-Seeking Policy Gradient, we simply maximize the probability of the elite episodes
        policy_loss = -log_probs.sum()
        entropy_bonus = entropies.sum()

        loss = policy_loss - self.entropy_weight * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy_bonus.mean().item()),
            "final_reward": float(np.mean(rewards)), # Mean reward of the whole batch
        }
