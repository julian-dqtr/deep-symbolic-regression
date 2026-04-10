import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ..core.config import TRAINING_CONFIG


class ReinforceOptimizer:
    def __init__(self, policy):
        self.policy = policy

        self.learning_rate = TRAINING_CONFIG["learning_rate"]
        self.entropy_weight = TRAINING_CONFIG["entropy_weight"]
        self.grad_clip_norm = TRAINING_CONFIG["grad_clip_norm"]

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        # Moving-average baseline to reduce variance
        self.baseline = 0.0
        self.baseline_momentum = 0.9

    def compute_returns(self, rewards):
        """
        Since the environment gives zero intermediate rewards and only a final reward,
        we propagate the final reward to all time steps.
        """
        final_reward = rewards[-1] if len(rewards) > 0 else 0.0
        returns = [final_reward for _ in rewards]
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, batch_episodes):
        """
        Perform one REINFORCE update from a batch of collected episodes.

        Args:
            batch_episodes (list): list of episode dicts

        Returns:
            dict: training statistics
        """
        all_log_probs = []
        all_entropies = []
        all_returns = []

        for episode in batch_episodes:
            if len(episode["log_probs"]) == 0:
                continue

            device = episode["log_probs"][0].device

            log_probs = torch.stack(episode["log_probs"])   # (T,)
            entropies = torch.stack(episode["entropies"])   # (T,)
            returns = self.compute_returns(episode["rewards"]).to(device)

            all_log_probs.append(log_probs)
            all_entropies.append(entropies)
            all_returns.append(returns)

        if not all_log_probs:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "entropy": 0.0,
                "baseline": float(self.baseline),
                "final_reward": float(np.mean([ep["final_reward"] for ep in batch_episodes])),
            }

        log_probs = torch.cat(all_log_probs)
        entropies = torch.cat(all_entropies)
        returns = torch.cat(all_returns)

        # Update moving-average baseline
        mean_return = returns.mean().item()
        self.baseline = (
            self.baseline_momentum * self.baseline
            + (1.0 - self.baseline_momentum) * mean_return
        )

        advantages = returns - self.baseline

        policy_loss = -(log_probs * advantages).sum() / len(batch_episodes)
        entropy_bonus = entropies.sum() / len(batch_episodes)
        loss = policy_loss - self.entropy_weight * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy": float(entropy_bonus.item()),
            "baseline": float(self.baseline),
            "final_reward": float(np.mean([ep["final_reward"] for ep in batch_episodes])),
        }